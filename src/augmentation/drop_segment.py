import numpy as np
import random
from src.preprocessing.corner import detect_corners
from src.preprocessing.segmentation import pseudo_segment

def apply_drop_segment(points, max_remove_ratio=0.5):
    """
    对输入的轨迹坐标 points 执行 DropSegment：
    1. 检测拐点
    2. 切分成片段
    3. 随机删除部分片段
    4. 重组并返回新的轨迹
    """
    # 1. 找拐点，切片段
    corners = detect_corners(points, k=2, threshold=180)
    segments = []
    all_indices = sorted([0] + list(corners) + [len(points) - 1])
    for i in range(len(all_indices) - 1):
        seg = points[all_indices[i]:all_indices[i+1]+1]
        seg = seg[seg[:, 0] != -1]
        if len(seg) >= 3:
            segments.append(seg)

    # 2. 如果片段太少（字太短），直接跳过丢弃
    if len(segments) <= 3:
        return points

    # 3. 计算要删除的片段数量，并随机选取
    max_remove = max(1, int(len(segments) * max_remove_ratio))
    remove_count = random.randint(0, max_remove)
    remove_indices = random.sample(range(len(segments)), remove_count)

    # 4. 重组片段
    new_points = []
    for i, seg in enumerate(segments):
        if i not in remove_indices:
            new_points.extend(seg)
            if i < len(segments) - 1:
                new_points.append((-1.0, 0.0))

    return np.array(new_points, dtype=np.float32)