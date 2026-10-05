"""
WPTTDataset：缓存切分结果 + 支持多进程/单进程 + 缓存页码映射 + DropSegment 开关。

关键修复：
- threshold=3（原 threshold=180 在 ×0.1 缩放后的坐标上失效，检测不到任何拐点）
- render_trajectory_to_bitmap 按 -1 分段绘制（见 path_signature.py）
- __getitem__ 里签名计算前过滤 -1（避免跳变点污染积分路径）
"""

import os
import gc
import numpy as np
import cv2
cv2.setNumThreads(0)  # fork 模式下禁用 OpenCV 内部多线程
import torch
import signatory
from torch.utils.data import Dataset
from multiprocessing import Pool

from src.data.loader import load_wptt_page
from src.preprocessing.corner import detect_corners
from src.preprocessing.segmentation import pseudo_segment
from src.features.path_signature import render_trajectory_to_bitmap
from src.augmentation.drop_segment import get_segments, apply_drop_segment_from_segments


def _load_one_page(file_path):
    return load_wptt_page(file_path)


def _segment_one_page(args):
    page_idx, points = args
    corners = detect_corners(points, k=2, threshold=3)
    chars = pseudo_segment(points, corners)
    return page_idx, chars


class WPTTDataset(Dataset):
    def __init__(self, metadata_file, split='Train', transform=None,
                 page_workers=8, segment_workers=8, apply_drop=True):
        """
        Args:
            metadata_file: metadata.csv 路径
            split: 'Train' 或 'Test'
            transform: 图像变换（如 _pad_to_96）
            page_workers: 页面加载的进程数，0 表示单进程
            segment_workers: 伪字符切分的进程数，0 表示单进程
            apply_drop: 是否在 __getitem__ 里做 DropSegment
                - True：训练集、20-test 评估使用
                - False：训练每轮评估使用（保证测试内容一致）
        """
        import pandas as pd
        self.df = pd.read_csv(metadata_file, encoding='utf-8-sig')
        self.df['writer_id'] = self.df['writer_id'].astype(str)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        self.writer_ids = sorted(self.df['writer_id'].unique())
        self.label_map = {wid: idx for idx, wid in enumerate(self.writer_ids)}
        self.num_classes = len(self.writer_ids)
        self.transform = transform
        self.apply_drop = apply_drop

        # ========== 页面加载 ==========
        file_list = self.df['file'].tolist()
        if page_workers > 0:
            print(f"正在加载 {split} 集的所有页面轨迹（{page_workers} 进程）...")
            with Pool(processes=page_workers) as pool:
                self.pages = pool.map(_load_one_page, file_list)
        else:
            print(f"正在加载 {split} 集的所有页面轨迹（单进程）...")
            self.pages = [_load_one_page(f) for f in file_list]
        print(f"✅ 加载了 {len(self.pages)} 个页面。")

        # ========== 伪字符切分 ==========
        page_args = [(i, points) for i, points in enumerate(self.pages)]
        if segment_workers > 0:
            print(f"正在预切分伪字符并建立索引（{segment_workers} 进程）...")
            with Pool(processes=segment_workers) as pool:
                results = pool.map(_segment_one_page, page_args)
        else:
            print(f"正在预切分伪字符并建立索引（单进程）...")
            results = [_segment_one_page(args) for args in page_args]

        # ========== 缓存切分后的字符坐标 ==========
        self.samples = []
        self.char_points_list = []
        for page_idx, chars in results:
            if len(chars) > 0:
                writer_id = self.df.iloc[page_idx]['writer_id']
                for char_idx in range(len(chars)):
                    self.samples.append((page_idx, char_idx, writer_id))
                    pts = chars[char_idx]
                    if pts.ndim == 1:
                        pts = pts.reshape(-1, 2)
                    self.char_points_list.append(pts)
        print(f"✅ 共生成 {len(self.samples)} 个伪字符样本，坐标已缓存。")

        # ========== 缓存页码映射 ==========
        self.page_names_for_samples = []
        for page_idx, _, _ in self.samples:
            self.page_names_for_samples.append(self.df.iloc[page_idx]['page'])
        print(f"✅ 已缓存样本到页码的映射。")

        # ========== 释放页面级轨迹 ==========
        self.pages = None
        gc.collect()
        print(f"🧹 已释放页面级轨迹缓存，节省约 5-10GB 内存。")

        # ========== 缓存 DropSegment 片段索引 ==========
        print(f"正在缓存 DropSegment 片段索引（{len(self.char_points_list)} 个样本）...")
        self.segments_list = []
        for char_pts in self.char_points_list:
            self.segments_list.append(get_segments(char_pts))
        print(f"✅ 已缓存 {len(self.segments_list)} 个样本的 DropSegment 片段索引。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        page_idx, char_idx, writer_id = self.samples[idx]
        char_points = self.char_points_list[idx]

        # ========== DropSegment 开关 ==========
        if self.apply_drop:
            drop_points = apply_drop_segment_from_segments(
                char_points, self.segments_list[idx]
            )
        else:
            drop_points = char_points

        new_corners = detect_corners(drop_points, k=2, threshold=3)
        chars_drop = pseudo_segment(drop_points, new_corners)

        if len(chars_drop) == 0:
            feat = np.zeros((54, 54, 64), dtype=np.float32)
        else:
            final_points = chars_drop[0]
            if final_points.ndim == 1:
                final_points = final_points.reshape(-1, 2)

            # 过滤掉 -1（抬笔分隔符），用于签名计算
            down_points = final_points[final_points[:, 0] != -1.0]

            if len(down_points) < 3:
                feat = np.zeros((54, 54, 64), dtype=np.float32)
            else:
                # 位图（内部按 -1 分段绘制）
                bitmap, norm_points = render_trajectory_to_bitmap(final_points)

                # 用过滤后的有效点计算路径签名
                traj_tensor = torch.from_numpy(down_points.astype(np.float32)).unsqueeze(0)
                try:
                    sig_tensor = signatory.signature(traj_tensor, 5).squeeze(0)
                except Exception:
                    feat = np.zeros((54, 54, 64), dtype=np.float32)
                    return torch.from_numpy(feat).permute(2, 0, 1).float(), self.label_map[writer_id]

                sig_np = sig_tensor.numpy()
                n_rows = len(norm_points) - 1
                if n_rows <= 0:
                    feat = np.zeros((54, 54, 64), dtype=np.float32)
                else:
                    flat_sig = sig_np.flatten()
                    target_len = n_rows * 63
                    if len(flat_sig) < target_len:
                        flat_sig = np.pad(flat_sig, (0, target_len - len(flat_sig)), 'constant')
                    else:
                        flat_sig = flat_sig[:target_len]
                    sig_np = flat_sig.reshape(n_rows, 63)

                    sig_maps = np.zeros((54, 54, 63), dtype=np.float32)
                    for i in range(1, len(norm_points)):
                        if i - 1 < sig_np.shape[0]:
                            x, y = norm_points[i]
                            if 0 <= x < 54 and 0 <= y < 54:
                                sig_maps[y, x, :] = sig_np[i - 1, :]

                    for c in range(63):
                        channel = sig_maps[:, :, c]
                        if np.max(channel) - np.min(channel) > 1e-5:
                            norm_ch = ((channel - np.min(channel)) / (np.max(channel) - np.min(channel)) * 255).astype(np.uint8)
                            eq_ch = cv2.equalizeHist(norm_ch)
                            sig_maps[:, :, c] = eq_ch.astype(np.float32) / 255.0

                    bitmap_3d = np.expand_dims(bitmap, axis=-1)
                    feat = np.concatenate([bitmap_3d, sig_maps], axis=-1).astype(np.float32)

        feat_tensor = torch.from_numpy(feat).permute(2, 0, 1).float()
        if self.transform:
            feat_tensor = self.transform(feat_tensor)
        label = self.label_map[writer_id]
        return feat_tensor, label