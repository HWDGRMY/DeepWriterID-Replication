"""
DCNN 评估入口。

用法：
    # 1-test（不开 DropSegment，可复现）
    python scripts/evaluate.py

    # 20-test（开 DropSegment，每轮随机删片段）
    python scripts/evaluate.py --n_test 20
"""

import os
import sys
import argparse
import torch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.dcnn import DCNN
from src.evaluation.evaluator import evaluate_page_level, evaluate_page_level_20test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='outputs/checkpoints/dcnn_best.pth')
    parser.add_argument('--metadata', type=str, default='data/features/metadata.csv')
    parser.add_argument('--n_test', type=int, default=1,
                        help='1 表示单次评估（不开 DropSegment），>1 表示多次评估（开 DropSegment）')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--log_name', type=str, default='20test_records.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 设备: {device}")

    df = pd.read_csv(args.metadata, encoding='utf-8-sig')
    df['writer_id'] = df['writer_id'].astype(str)
    all_writer_ids = sorted(df['writer_id'].unique())
    global_label_map = {wid: idx for idx, wid in enumerate(all_writer_ids)}
    num_classes = len(global_label_map)
    print(f"📊 类别数: {num_classes}")

    model = DCNN(num_classes=num_classes).to(device)

    if not os.path.exists(args.ckpt):
        print(f"❌ 找不到权重文件: {args.ckpt}")
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"✅ 已加载权重: {args.ckpt}")
        if 'epoch' in ckpt:
            print(f"   来源 epoch: {ckpt['epoch']}, best_acc: {ckpt.get('best_acc', '?')}")
    else:
        model.load_state_dict(ckpt)
        print(f"✅ 已加载权重（裸权重）: {args.ckpt}")

    if args.n_test <= 1:
        acc = evaluate_page_level(
            model, args.metadata, global_label_map, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            apply_drop=False,
        )
    else:
        acc = evaluate_page_level_20test(
            model, args.metadata, global_label_map, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_test=args.n_test,
            log_name=args.log_name,
            apply_drop=True,
        )

    print(f"\n{'='*60}")
    print(f"最终页面级准确率: {acc*100:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()