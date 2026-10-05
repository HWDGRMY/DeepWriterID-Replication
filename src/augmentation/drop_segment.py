import numpy as np
import random
from src.preprocessing.corner import detect_corners


def get_segments(points, threshold=3):
    """
    检测拐点并切分片段，返回片段索引列表 [(start, end), ...]。
    只做一次，可缓存。片段判定与原 apply_drop_segment 一致：
    过滤掉 (-1, 0) 后，有效点数 >= 3 才保留。
    """
    corners = detect_corners(points, k=2, threshold=threshold)
    all_indices = sorted([0] + list(corners) + [len(points) - 1])
    segments = []
    for i in range(len(all_indices) - 1):
        start, end = all_indices[i], all_indices[i + 1]
        seg = points[start:end + 1]
        if seg.ndim == 1:
            seg = seg.reshape(-1, 2)
        seg = seg[seg[:, 0] != -1]
        if len(seg) >= 3:
            segments.append((start, end))
    return segments


def apply_drop_segment_from_segments(points, segments, max_remove_ratio=0.5):
    """
    基于预计算的片段索引，随机删除部分片段并重组轨迹。
    """
    if len(segments) <= 3:
        return points

    max_remove = max(1, int(len(segments) * max_remove_ratio))
    remove_count = random.randint(0, max_remove)
    remove_indices = set(random.sample(range(len(segments)), remove_count))

    new_points = []
    for i, (start, end) in enumerate(segments):
        if i not in remove_indices:
            seg = points[start:end + 1]
            if seg.ndim == 1:
                seg = seg.reshape(-1, 2)
            seg = seg[seg[:, 0] != -1]
            new_points.extend(seg)
            if i < len(segments) - 1:
                new_points.append((-1.0, 0.0))

    if len(new_points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(new_points, dtype=np.float32)


def apply_drop_segment(points, max_remove_ratio=0.5):
    """兼容旧接口：内部先检测片段，再删除。"""
    segments = get_segments(points)
    return apply_drop_segment_from_segments(points, segments, max_remove_ratio)