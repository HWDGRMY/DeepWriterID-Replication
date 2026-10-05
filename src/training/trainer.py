"""
DCNN 训练器（支持从 95.09% 模型微调）。

关键设计：
- 微调模式：加载预训练模型权重，optimizer/scheduler 重新初始化
- 每个 epoch 新建 train_loader（fork 多进程），epoch 结束自动销毁
- 训练阶段与评估阶段的 worker 数分离，避免进程冲突
- 断点续训：完整保存/恢复模型、优化器、调度器状态
- 每轮指标写入 CSV
"""

import os
import sys
import csv
import time
import gc
import random as _random
import numpy as _np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.dataset import WPTTDataset
from src.models.dcnn import DCNN
from src.evaluation.evaluator import evaluate_page_level


def _worker_init_fn(worker_id):
    """每个 DataLoader worker 独立随机种子，保证 DropSegment 真随机。"""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    _random.seed(worker_seed)
    _np.random.seed(worker_seed)


def _pad_to_96(tensor):
    """54×54 → 96×96。"""
    pad = (21, 21, 21, 21)
    return torch.nn.functional.pad(tensor, pad, mode='constant', value=0)


def _init_log_file(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_char_acc',
                'test_page_acc', 'lr', 'epoch_time_sec'
            ])
        print(f"📝 已创建训练日志: {log_path}")


def _append_log(log_path, row):
    with open(log_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def train(config):
    # ========== 读取超参数 ==========
    BATCH_SIZE = config.get('batch_size', 192)
    EPOCHS = config.get('epochs', 50)
    INITIAL_LR = config.get('initial_lr', 0.0005)
    EVAL_FREQUENCY = config.get('eval_frequency', 1)
    NUM_WORKERS = config.get('num_workers', 16)
    PAGE_WORKERS = config.get('page_workers', 8)
    SEGMENT_WORKERS = config.get('segment_workers', 8)

    PRETRAINED_PATH = config.get('pretrained_path', None)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print(f"🔧 配置: batch={BATCH_SIZE}, lr={INITIAL_LR}, "
          f"eval_freq={EVAL_FREQUENCY}, workers={NUM_WORKERS}")
    if PRETRAINED_PATH:
        print(f"🔧 微调模式：从 {PRETRAINED_PATH} 加载预训练权重")

    # ========== 数据路径 ==========
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    metadata_file = os.path.join(BASE_DIR, 'data', 'features', 'metadata.csv')

    print("正在加载数据集 ...")
    full_df = pd.read_csv(metadata_file, encoding='utf-8-sig')
    full_df['writer_id'] = full_df['writer_id'].astype(str)
    all_writer_ids = sorted(full_df['writer_id'].unique())
    global_label_map = {wid: idx for idx, wid in enumerate(all_writer_ids)}
    num_classes = len(global_label_map)

    # ========== 数据增强与 Dataset ==========
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        _pad_to_96
    ])

    train_dataset = WPTTDataset(
        metadata_file, split='Train', transform=train_transform,
        page_workers=PAGE_WORKERS, segment_workers=SEGMENT_WORKERS,
        apply_drop=True,
    )
    print(f"📊 训练集样本数: {len(train_dataset)}")
    print(f"📊 作者（类别）总数: {num_classes}")

    # ========== 模型 ==========
    model = DCNN(num_classes=num_classes).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型参数量: {total_params / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()

    # ========== 优化器与调度器 ==========
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ========== 路径与状态 ==========
    log_path = os.path.join(BASE_DIR, 'outputs', 'logs', 'training_log_finetune.csv')
    latest_model_path = os.path.join(BASE_DIR, 'outputs', 'checkpoints', 'dcnn_finetune_latest.pth')
    best_model_path = os.path.join(BASE_DIR, 'outputs', 'checkpoints', 'dcnn_finetune_best.pth')
    os.makedirs(os.path.dirname(latest_model_path), exist_ok=True)
    _init_log_file(log_path)

    start_epoch = 1
    best_page_acc = 0.0

    # ========== 加载预训练权重（微调模式） ==========
    if PRETRAINED_PATH and os.path.exists(PRETRAINED_PATH):
        print(f"📦 正在加载预训练权重: {PRETRAINED_PATH}")
        ckpt = torch.load(PRETRAINED_PATH, map_location=DEVICE)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            pretrained_acc = ckpt.get('best_acc', 0.0)
            print(f"✅ 加载成功（原模型 best_acc: {pretrained_acc:.4f}），optimizer/scheduler 重新初始化")
        else:
            model.load_state_dict(ckpt, strict=False)
            print(f"✅ 加载成功（裸权重），optimizer/scheduler 重新初始化")
    else:
        print("🆕 未指定预训练权重，从零开始训练。")

    # ========== 断点续训 ==========
    if os.path.exists(latest_model_path):
        checkpoint = torch.load(latest_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_page_acc = checkpoint.get('best_acc', 0.0)
        if 'optimizer_state_dict' in checkpoint and 'scheduler_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"✅ 完整断点恢复: Epoch {start_epoch}, Best Page Acc: {best_page_acc:.4f}")
        else:
            print(f"✅ 仅模型权重恢复: Epoch {start_epoch}, Best Page Acc: {best_page_acc:.4f}")
    else:
        print("🆕 未找到续训检查点，从 Epoch 1 开始。")

    # ========== 训练主循环 ==========
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start_time = time.time()

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=2,
            multiprocessing_context='fork',
            worker_init_fn=_worker_init_fn,
        )

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, pred = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (pred == labels).sum().item()

            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct_train/total_train:.4f}",
            })

        del train_loader
        gc.collect()

        epoch_loss = running_loss / total_train
        epoch_acc = correct_train / total_train
        print(f"Epoch {epoch} Train | Loss: {epoch_loss:.4f} | Char Acc: {epoch_acc:.4f}")

        # 保存 latest（评估前）
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_page_acc,
        }, latest_model_path)

        # 页面级评估
        test_page_acc = None
        if epoch % EVAL_FREQUENCY == 0:
            test_page_acc = evaluate_page_level(
                model, metadata_file, global_label_map, DEVICE,
                num_workers=8, page_workers=8, segment_workers=8,
            )
            print(f"Epoch {epoch} Test | 页面级准确率: {test_page_acc:.4f}")

            if test_page_acc > best_page_acc:
                best_page_acc = test_page_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_page_acc,
                }, best_model_path)
                print(f"🎉 新最佳模型保存 (页面 Acc: {best_page_acc:.4f})")

        # 写日志
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        _append_log(log_path, [
            epoch,
            f"{epoch_loss:.4f}",
            f"{epoch_acc:.4f}",
            f"{test_page_acc:.4f}" if test_page_acc is not None else "",
            f"{current_lr:.8f}",
            f"{epoch_time:.1f}",
        ])

        scheduler.step()
        print("-" * 50)