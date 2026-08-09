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

def evaluate_page_level(model, metadata_file, global_label_map, device, batch_size=256, num_workers=0):
    """
    页面级准确率评估函数。
    新增 num_workers 参数，默认 0（单线程），保证 Windows 本地绝对稳定。
    """
    test_dataset = WPTTDataset(metadata_file, split='Test', transform=_pad_to_96)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,     # 恢复成可配置参数，默认 0
        pin_memory=False
    )

    model.eval()
    print(f"\n📊 正在并行推理测试集...")

    df_test = pd.read_csv(metadata_file, encoding='utf-8-sig')
    df_test['writer_id'] = df_test['writer_id'].astype(str)
    test_df = df_test[df_test['split'] == 'Test'].reset_index(drop=True)
    page_names = []
    for page_idx, _, _ in test_dataset.samples:
        page_names.append(test_df.iloc[page_idx]['page'])

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