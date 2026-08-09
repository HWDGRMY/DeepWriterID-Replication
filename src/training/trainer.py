import os
import sys
import csv
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

def _pad_to_96(tensor):
    pad = (21, 21, 21, 21)
    return torch.nn.functional.pad(tensor, pad, mode='constant', value=0)

def train(config):
    # 从配置文件读取参数（兼容云端和本地）
    BATCH_SIZE = config.get('batch_size', 256)
    EPOCHS = config.get('epochs', 50)
    INITIAL_LR = config.get('initial_lr', 0.0005)
    EVAL_FREQUENCY = config.get('eval_frequency', 5)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 绝对定位项目根目录，确保云端和本地路径一致
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    metadata_file = os.path.join(BASE_DIR, 'data', 'features', 'metadata.csv')

    if not os.path.exists(metadata_file):
        print(f"❌ 错误：找不到 {metadata_file}。请确认是否已运行预处理脚本。")
        return

    print("正在加载数据集 ...")
    full_df = pd.read_csv(metadata_file, encoding='utf-8-sig')
    full_df['writer_id'] = full_df['writer_id'].astype(str)
    all_writer_ids = sorted(full_df['writer_id'].unique())
    global_label_map = {wid: idx for idx, wid in enumerate(all_writer_ids)}
    num_classes = len(global_label_map)

    train_transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        _pad_to_96
    ])

    train_dataset = WPTTDataset(metadata_file, split='Train', transform=train_transform)
    print(f"📊 训练集样本数: {len(train_dataset)}")
    print(f"📊 作者（类别）总数: {num_classes}")

    # 读取配置中的并行参数（本地设为0可防卡死，云端可设16）
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=config.get('num_workers', 16),
        pin_memory=config.get('pin_memory', False),
        prefetch_factor=config.get('prefetch_factor', 1),
        multiprocessing_context='spawn'
    )

    model = DCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 创建输出目录
    log_path = os.path.join(BASE_DIR, 'outputs', 'logs', 'training_log.csv')
    latest_model_path = os.path.join(BASE_DIR, 'outputs', 'checkpoints', 'dcnn_latest.pth')
    best_model_path = os.path.join(BASE_DIR, 'outputs', 'checkpoints', 'dcnn_best.pth')
    os.makedirs(os.path.dirname(latest_model_path), exist_ok=True)

    start_epoch = 1
    best_test_acc = 0.0

    # 断点续训（仅加载权重，优化器状态重置以防震荡）
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            start_epoch = checkpoint['epoch'] + 1
            best_test_acc = checkpoint.get('best_acc', 0.0)
    elif os.path.exists(latest_model_path):
        checkpoint = torch.load(latest_model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            start_epoch = checkpoint['epoch'] + 1
            best_test_acc = checkpoint.get('best_acc', 0.0)

    for param_group in optimizer.param_groups:
        param_group['lr'] = INITIAL_LR
    remaining_epochs = EPOCHS - start_epoch + 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)
    print(f"✅ 将从 Epoch {start_epoch} 继续，初始学习率 {INITIAL_LR}")

    for epoch in range(start_epoch, EPOCHS + 1):
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
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct_train/total_train:.4f}"})

        epoch_loss = running_loss / total_train
        epoch_acc = correct_train / total_train
        print(f"Epoch {epoch} Train | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        # 评估逻辑
        test_page_acc = 0.0
        if epoch % EVAL_FREQUENCY == 0:
            test_page_acc = evaluate_page_level(model, metadata_file, global_label_map, DEVICE)
            print(f"Epoch {epoch} Test | 页面级准确率: {test_page_acc:.4f}")
            if test_page_acc > best_test_acc:
                best_test_acc = test_page_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"🎉 新最佳模型保存 (Acc: {best_test_acc:.4f})")
        else:
            print(f"Epoch {epoch} Test | (已跳过测试，下次评估在 Epoch {epoch + (EVAL_FREQUENCY - (epoch % EVAL_FREQUENCY))})")

        # 写入 CSV 日志
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{epoch_loss:.4f}", f"{epoch_acc:.4f}", f"{test_page_acc:.4f}"])

        # 保存最新状态
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'best_acc': best_test_acc}, latest_model_path)
        scheduler.step()
        print("-" * 50)