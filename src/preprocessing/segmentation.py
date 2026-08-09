import numpy as np

def pseudo_segment(points, corners_idx):
    segs = []
    all_indices = sorted([0] + list(corners_idx) + [len(points) - 1])
    for i in range(len(all_indices) - 1):
        start = all_indices[i]
        end = all_indices[i+1] + 1
        seg = points[start:end]
        seg = seg[seg[:, 0] != -1]
        if len(seg) >= 3:
            segs.append(seg)

    if len(segs) == 0:
        return []

    all_y = points[points[:, 0] != -1][:, 1]
    page_height = np.max(all_y) - np.min(all_y) if len(all_y) > 0 else 100.0
    avg_height = page_height / 10.0
    if avg_height < 10.0:
        avg_height = 50.0

    pseudo_chars = []
    current_char = []
    current_width = 0.0

    for seg in segs:
        seg_width = np.max(seg[:, 0]) - np.min(seg[:, 0])
        if current_width + seg_width > avg_height:
            if len(current_char) > 0:
                char_points = np.vstack(current_char)
                pseudo_chars.append(char_points)
            current_char = [seg]
            current_width = seg_width
        else:
            current_char.append(seg)
            current_width += seg_width

    if len(current_char) > 0:
        char_points = np.vstack(current_char)
        pseudo_chars.append(char_points)

    if len(pseudo_chars) == 0:
        for seg in segs:
            pseudo_chars.append(seg)

    return pseudo_chars