import os
import sys
import torch
import numpy as np
import cv2
import signatory
from torch.utils.data import Dataset

from src.data.loader import load_wptt_page
from src.preprocessing.corner import detect_corners
from src.preprocessing.segmentation import pseudo_segment
from src.features.path_signature import render_trajectory_to_bitmap, generate_path_signature_features
from src.augmentation.drop_segment import apply_drop_segment  # 🟢 导入增强模块


class WPTTDataset(Dataset):
    def __init__(self, metadata_file, split='Train', transform=None):
        import pandas as pd
        self.df = pd.read_csv(metadata_file, encoding='utf-8-sig')
        self.df['writer_id'] = self.df['writer_id'].astype(str)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        self.writer_ids = sorted(self.df['writer_id'].unique())
        self.label_map = {wid: idx for idx, wid in enumerate(self.writer_ids)}
        self.num_classes = len(self.writer_ids)
        self.transform = transform

        print(f"正在加载 {split} 集的所有页面轨迹...")
        self.pages = []
        for _, row in self.df.iterrows():
            self.pages.append(load_wptt_page(row['file']))
        print(f"✅ 加载了 {len(self.pages)} 个页面。")

        print(f"正在预切分伪字符并建立索引...")
        self.samples = []
        for page_idx, points in enumerate(self.pages):
            corners = detect_corners(points, k=2, threshold=180)
            chars = pseudo_segment(points, corners)
            if len(chars) > 0:
                writer_id = self.df.iloc[page_idx]['writer_id']
                for char_idx in range(len(chars)):
                    self.samples.append((page_idx, char_idx, writer_id))
        print(f"✅ 共生成 {len(self.samples)} 个伪字符样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        page_idx, char_idx, writer_id = self.samples[idx]
        points = self.pages[page_idx]
        corners = detect_corners(points, k=2, threshold=180)
        chars = pseudo_segment(points, corners)
        char_points = chars[char_idx]
        if char_points.ndim == 1:
            char_points = char_points.reshape(-1, 2)

        # 🟢 调用独立的 DropSegment 增强模块
        drop_points = apply_drop_segment(char_points)

        # 重新切分（通常还是同一个字，但偶尔可能出现多个片段）
        new_corners = detect_corners(drop_points, k=2, threshold=180)
        chars_drop = pseudo_segment(drop_points, new_corners)

        if len(chars_drop) == 0:
            feat = np.zeros((54, 54, 64), dtype=np.float32)
        else:
            final_points = chars_drop[0]
            if final_points.ndim == 1:
                final_points = final_points.reshape(-1, 2)
            if len(final_points) < 2:
                feat = np.zeros((54, 54, 64), dtype=np.float32)
            else:
                # 渲染位图（CPU）
                bitmap, norm_points = render_trajectory_to_bitmap(final_points)

                # 路径签名（CPU，使用 signatory）
                traj_tensor = torch.from_numpy(final_points.astype(np.float32)).unsqueeze(0)
                try:
                    sig_tensor = signatory.signature(traj_tensor, 5).squeeze(0)
                except:
                    feat = np.zeros((54, 54, 64), dtype=np.float32)
                    return torch.from_numpy(feat).permute(2, 0, 1).float(), self.label_map[writer_id]

                sig_np = sig_tensor.numpy()
                n_rows = len(norm_points) - 1
                if n_rows <= 0:
                    feat = np.zeros((54, 54, 64), dtype=np.float32)
                else:
                    # 强制补零到 63 列（防止 62 维度问题）
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

                    # 直方图均衡化（只在非零通道执行）
                    for c in range(63):
                        channel = sig_maps[:, :, c]
                        if np.max(channel) - np.min(channel) > 1e-5:
                            norm_ch = ((channel - np.min(channel)) / (np.max(channel) - np.min(channel)) * 255).astype(
                                np.uint8)
                            eq_ch = cv2.equalizeHist(norm_ch)
                            sig_maps[:, :, c] = eq_ch.astype(np.float32) / 255.0

                    bitmap_3d = np.expand_dims(bitmap, axis=-1)
                    feat = np.concatenate([bitmap_3d, sig_maps], axis=-1).astype(np.float32)

        feat_tensor = torch.from_numpy(feat).permute(2, 0, 1).float()
        if self.transform:
            feat_tensor = self.transform(feat_tensor)
        label = self.label_map[writer_id]
        return feat_tensor, label