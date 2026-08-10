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
    BATCH_SIZE = config.get('batch_size', 256)
    EPOCHS = config.get('epochs', 50)
    INITIAL_LR = config.get('initial_lr', 0.0005)
    EVAL_FREQUENCY = config.get('eval_frequency', 5)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

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
    # 调度器的总轮次固定为 EPOCHS（这样内部会自动计算剩余衰减）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    log_path = os.path.join(BASE_DIR, 'outputs', 'logs', 'training_log.csv')
    latest_model_path = os.path.join(BASE_DIR, 'outputs', 'checkpoints', 'dcnn_latest.pth')
    best_model_path = os.path.join(BASE_DIR, 'outputs', 'checkpoints', 'dcnn_best.pth')
    os.makedirs(os.path.dirname(latest_model_path), exist_ok=True)

    start_epoch = 1
    best_test_acc = 0.0

    # ========== 真正的断点续训（恢复优化器和调度器状态） ==========
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
    elif os.path.exists(latest_model_path):
        checkpoint = torch.load(latest_model_path, map_location=DEVICE)
    else:
        checkpoint = None

    if checkpoint is not None:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            start_epoch = checkpoint['epoch'] + 1
            best_test_acc = checkpoint.get('best_acc', 0.0)
            # 如果检查点中包含了优化器和调度器的状态，则恢复它们
            if 'optimizer_state_dict' in checkpoint and 'scheduler_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✅ 成功恢复优化器和调度器状态，学习率将从中断处继续下降。")
            else:
                # 若没有保存优化器/调度器，则重置（学习率回到 INITIAL_LR）
                for param_group in optimizer.param_groups:
                    param_group['lr'] = INITIAL_LR
                # 调度器需要重新初始化，但保持 T_max = EPOCHS，内部会重新计数
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
                print("⚠️ 检查点中未包含优化器/调度器状态，已重置学习率。建议使用最新完整检查点。")
        else:
            # 旧格式只存了权重，没有 epoch 信息，默认为从头开始
            model.load_state_dict(checkpoint, strict=False)
            print("⚠️ 检测到旧格式检查点，仅加载权重，训练将从 Epoch 1 重新开始。")
    else:
        print("🆕 未找到检查点，将从 Epoch 1 开始全新训练。")

    print(f"✅ 将从 Epoch {start_epoch} 继续训练。")
    # =============================================================

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

        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{epoch_loss:.4f}", f"{epoch_acc:.4f}", f"{test_page_acc:.4f}"])

        # 🟢 保存完整检查点（模型、优化器、调度器）
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_test_acc
        }, latest_model_path)

        scheduler.step()
        print("-" * 50)