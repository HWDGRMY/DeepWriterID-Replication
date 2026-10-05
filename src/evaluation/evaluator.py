"""
页面级投票评估。

- evaluate_page_level：1-test，训练时每轮调用，默认不开 DropSegment
- evaluate_page_level_20test：20-test，训练完成后调用，默认开 DropSegment
"""

import os
import gc
import csv
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader

from src.data.dataset import WPTTDataset


def _pad_to_96(tensor):
    pad = (21, 21, 21, 21)
    return torch.nn.functional.pad(tensor, pad, mode='constant', value=0)


def _compute_page_acc(probs, sample_pages, page_to_true_label):
    page_votes = defaultdict(list)
    for i, page in enumerate(sample_pages):
        page_votes[page].append(probs[i])

    correct = 0
    for page, char_probs_list in page_votes.items():
        page_avg_prob = np.mean(char_probs_list, axis=0)
        pred_label = np.argmax(page_avg_prob)
        if pred_label == page_to_true_label[page]:
            correct += 1
    return correct / len(page_votes)


# ============================================================
# 1-test：训练时每轮调用，不开 DropSegment
# ============================================================

def evaluate_page_level(model, metadata_file, global_label_map, device,
                        batch_size=128, num_workers=8,
                        page_workers=8, segment_workers=8,
                        apply_drop=False):
    test_dataset = WPTTDataset(
        metadata_file, split='Test', transform=_pad_to_96,
        page_workers=page_workers,
        segment_workers=segment_workers,
        apply_drop=apply_drop,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        multiprocessing_context='fork' if num_workers > 0 else None,
    )

    model.eval()
    print(f"\n📊 正在并行推理测试集（apply_drop={apply_drop}）...")

    df_test = pd.read_csv(metadata_file, encoding='utf-8-sig')
    df_test['writer_id'] = df_test['writer_id'].astype(str)
    test_df = df_test[df_test['split'] == 'Test'].reset_index(drop=True)

    page_names = list(test_dataset.page_names_for_samples)

    all_probs = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="并行推理"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.detach().cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)

    page_votes = defaultdict(list)
    for prob, page in zip(all_probs, page_names):
        page_votes[page].append(prob)

    correct = 0
    for page, probs in page_votes.items():
        avg_prob = np.mean(probs, axis=0)
        pred_label = np.argmax(avg_prob)
        true_writer = test_df[test_df['page'] == page].iloc[0]['writer_id']
        true_label = global_label_map[true_writer]
        if pred_label == true_label:
            correct += 1

    acc = correct / len(page_votes)
    print(f"\n✅ 页面级准确率: {acc * 100:.2f}% ({correct}/{len(page_votes)})")
    return acc


# ============================================================
# 20-test：训练完成后调用，开 DropSegment
# ============================================================

def evaluate_page_level_20test(model, metadata_file, global_label_map, device,
                                batch_size=256, num_workers=8, n_test=20,
                                page_workers=8, segment_workers=8,
                                log_name='20test_records.csv',
                                apply_drop=True):
    model.eval()

    print(f"\n📋 初始化测试集元信息（apply_drop={apply_drop}）...")
    base_dataset = WPTTDataset(
        metadata_file, split='Test', transform=_pad_to_96,
        page_workers=page_workers,
        segment_workers=segment_workers,
        apply_drop=apply_drop,
    )
    sample_pages = list(base_dataset.page_names_for_samples)
    n_samples = len(sample_pages)
    unique_pages = sorted(set(sample_pages))
    print(f"✅ 测试集共 {n_samples} 个伪字符，分布在 {len(unique_pages)} 个页面。")

    df_test = pd.read_csv(metadata_file, encoding='utf-8-sig')
    df_test['writer_id'] = df_test['writer_id'].astype(str)
    test_df = df_test[df_test['split'] == 'Test'].reset_index(drop=True)
    page_to_true_label = {}
    for page in unique_pages:
        wid = test_df[test_df['page'] == page].iloc[0]['writer_id']
        page_to_true_label[page] = global_label_map[wid]

    del base_dataset
    gc.collect()

    accumulated_probs = None
    per_round_records = []

    for r in range(n_test):
        print(f"\n🔄 第 {r+1}/{n_test} 轮增强")
        dataset = WPTTDataset(
            metadata_file, split='Test', transform=_pad_to_96,
            page_workers=page_workers,
            segment_workers=segment_workers,
            apply_drop=apply_drop,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            multiprocessing_context='fork' if num_workers > 0 else None,
        )

        round_probs = []
        with torch.no_grad():
            for inputs, _ in tqdm(loader, desc=f"Round {r+1}"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                round_probs.append(probs)
        round_probs = np.concatenate(round_probs, axis=0)

        round_page_acc = _compute_page_acc(round_probs, sample_pages, page_to_true_label)

        if accumulated_probs is None:
            accumulated_probs = round_probs.copy()
        else:
            accumulated_probs += round_probs

        cum_avg_probs = accumulated_probs / (r + 1)
        cumulative_page_acc = _compute_page_acc(cum_avg_probs, sample_pages, page_to_true_label)

        per_round_records.append((r + 1, round_page_acc, cumulative_page_acc))
        print(f"  → 第 {r+1} 轮 page acc: {round_page_acc*100:.2f}%  "
              f"| 累积 page acc: {cumulative_page_acc*100:.2f}%")

        del dataset, loader, round_probs
        gc.collect()
        torch.cuda.empty_cache()

    log_dir = 'outputs/logs'
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, log_name)
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'round_page_acc', 'cumulative_page_acc'])
        for rec in per_round_records:
            writer.writerow([rec[0], f"{rec[1]:.4f}", f"{rec[2]:.4f}"])
    print(f"\n📝 每轮记录已保存至 {csv_path}")

    final_acc = per_round_records[-1][2]
    print(f"\n✅ {n_test}-test 最终页面级准确率: {final_acc * 100:.2f}%")
    return final_acc