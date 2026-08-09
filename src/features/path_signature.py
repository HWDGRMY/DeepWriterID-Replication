import numpy as np
import cv2
import torch
import signatory

def render_trajectory_to_bitmap(points, target_size=54):
    points = np.array(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(-1, 2)
    down_points = points[points[:, 0] != -1]
    if len(down_points) < 3:
        return np.zeros((target_size, target_size), dtype=np.float32), down_points

    min_x, max_x = np.min(down_points[:, 0]), np.max(down_points[:, 0])
    min_y, max_y = np.min(down_points[:, 1]), np.max(down_points[:, 1])
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    scale = min((target_size - 4) / width, (target_size - 4) / height)
    norm_points = down_points.astype(np.float32)
    norm_points[:, 0] = (norm_points[:, 0] - min_x) * scale + 2
    norm_points[:, 1] = (norm_points[:, 1] - min_y) * scale + 2
    norm_points = norm_points.astype(int)

    img = np.full((target_size, target_size), 255, dtype=np.uint8)
    cv2.polylines(img, [norm_points], isClosed=False, color=0, thickness=1)
    return img.astype(np.float32) / 255.0, norm_points

def generate_path_signature_features(points, max_level=5, target_size=54):
    points = np.array(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(-1, 2)
    down_points = points[points[:, 0] != -1]
    if len(down_points) < 3:
        return np.zeros((target_size, target_size, 63), dtype=np.float32)

    # 计算路径签名（在 CPU 上，避免多进程 CUDA 问题）
    traj_tensor = torch.from_numpy(down_points.astype(np.float32)).unsqueeze(0)
    try:
        sig_tensor = signatory.signature(traj_tensor, max_level).squeeze(0)
    except:
        # 如果签名计算失败，返回全零
        return np.zeros((target_size, target_size, 63), dtype=np.float32)

    sig_np = sig_tensor.numpy()
    # 计算点数，用于确定签名行数
    min_x, max_x = np.min(down_points[:, 0]), np.max(down_points[:, 0])
    min_y, max_y = np.min(down_points[:, 1]), np.max(down_points[:, 1])
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    scale = min((target_size - 4) / width, (target_size - 4) / height)
    norm_points = down_points.astype(np.float32)
    norm_points[:, 0] = (norm_points[:, 0] - min_x) * scale + 2
    norm_points[:, 1] = (norm_points[:, 1] - min_y) * scale + 2
    norm_points = norm_points.astype(int)

    n_rows = len(norm_points) - 1
    if n_rows <= 0:
        return np.zeros((target_size, target_size, 63), dtype=np.float32)

    # 🟢 核心修复：强制填充/截断到 n_rows × 63
    flat_sig = sig_np.flatten()
    target_len = n_rows * 63
    if len(flat_sig) < target_len:
        flat_sig = np.pad(flat_sig, (0, target_len - len(flat_sig)), 'constant')
    else:
        flat_sig = flat_sig[:target_len]
    sig_np = flat_sig.reshape(n_rows, 63)

    feature_maps = np.zeros((target_size, target_size, 63), dtype=np.float32)
    for i in range(1, len(norm_points)):
        if i-1 < sig_np.shape[0]:
            x, y = norm_points[i]
            if 0 <= x < target_size and 0 <= y < target_size:
                feature_maps[y, x, :] = sig_np[i-1, :]

    # 直方图均衡化
    for c in range(63):
        channel = feature_maps[:, :, c]
        if np.max(channel) - np.min(channel) > 1e-5:
            norm_ch = ((channel - np.min(channel)) / (np.max(channel) - np.min(channel)) * 255).astype(np.uint8)
            eq_ch = cv2.equalizeHist(norm_ch)
            feature_maps[:, :, c] = eq_ch.astype(np.float32) / 255.0
        else:
            feature_maps[:, :, c] = 0

    return feature_maps