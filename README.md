# DeepWriterID Replication on CASIA-OLHWDB

## 项目简介
本项目是对论文《DeepWriterID: An End-to-end Online Text-independent Writer Identification System》的完整复刻，并基于 **CASIA-OLHWDB 2.0-2.2 在线手写文本数据集（1019 位书写者）** 进行了验证。

## 核心成绩
在 1019 个类别（远超原论文 187 类）的分类任务下，通过 DropSegment、路径签名特征提取与页面级投票策略，最终测试集页面级准确率达到 **95.09%**。
- 原论文在 NLPR (187类) 上的成绩为 95.72%
- 本项目在 1019 类分类任务上实现了极具竞争力的泛化性能。

## 硬件与软件环境
本项目在以下云端环境跑通，并取得最终成绩：
- **GPU**: NVIDIA RTX 4090 D (24GB VRAM)
- **CPU**: AMD EPYC 9754 (18 vCPU)
- **RAM**: 60 GB
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.12
- **CUDA**: 12.4
- **PyTorch**: 2.5.1

## 项目目录结构
```text
DeepWriterID-Replication/
├── configs/                            # 参数配置文件
│   └── default.yaml                    # 全局超参数
├── data/                               # 数据目录
│   ├── raw/                            # 存放原始 .wptt 轨迹文件
│   └── features/                       # 存放 metadata.csv
├── src/                                # 核心源码包
│   ├── data/                           # 数据加载与处理
│   │   ├── loader.py                   # 解析 .wptt
│   │   └── dataset.py                  # WPTTDataset (含 DropSegment)
│   ├── models/                         # 模型定义
│   │   └── dcnn.py                     # DCNN 网络架构
│   ├── preprocessing/                  # 预处理
│   │   ├── corner.py                   # 拐点检测
│   │   └── segmentation.py             # 伪字符切分
│   ├── features/                       # 特征提取
│   │   └── path_signature.py           # 路径签名
│   ├── augmentation/                   # 数据增强
│   │   └── drop_segment.py             # 核心 DropSegment 逻辑
│   ├── training/                       # 训练引擎
│   │   └── trainer.py                  # 核心训练循环
│   ├── evaluation/                     # 评估引擎
│   │   └── evaluator.py                # 页面级投票准确率计算
│   └── utils/                          # 工具函数
│       └── visualizer.py               # 可视化工具
├── scripts/                            # 可执行入口脚本
│   ├── preprocess.py                   # 预处理 (生成 metadata.csv)
│   ├── train.py                        # 开始训练
│   └── evaluate.py                     # 单独评估模型
├── outputs/                            # 产出物（不上传 GitHub）
│   ├── logs/                           # 训练日志 CSV
│   ├── checkpoints/                    # 模型权重 .pth
│   └── results/                        # 测试可视化图片
├── environment.yml                     # Conda 环境配置
├── setup.py                            # 包安装配置
├── README.md                           # 项目说明
└── LICENSE                             # 开源协议
```

## 快速开始

### 1. 环境配置
建议使用 Conda 创建并激活环境：
```bash
conda env create -f environment.yml
conda activate deepwriterid
```
或使用 pip 安装（确保已安装 PyTorch 与 CUDA 环境）：
```bash
pip install -e .
```

### 2. 数据准备
- 向中科院自动化所申请获取 CASIA-OLHWDB 在线文本数据（`.wptt` 文件）。官方网站申请地址：[https://nlpr.ia.ac.cn/databases/handwriting/Home.html](https://nlpr.ia.ac.cn/databases/handwriting/Home.html)
- 将解压后的数据放在 `data/raw/` 目录下，保持文件夹结构为：
  ```text
  data/raw/WPTT2.x-Train/
  data/raw/WPTT2.x-Test/
  ```
- 运行预处理脚本生成 `metadata.csv`：
  ```bash
  python scripts/preprocess.py
  ```

### 3. 训练模型
修改 `configs/default.yaml` 中的参数（如 `batch_size`, `epochs` 等），然后运行：
```bash
python scripts/train.py
```
- 训练日志将保存在 `outputs/logs/training_log.csv`。
- 每轮结束后，最新模型将保存至 `outputs/checkpoints/dcnn_latest.pth`。
- 如果该轮测试集准确率刷新记录，将额外保存至 `outputs/checkpoints/dcnn_best.pth`。

### 4. 评估模型
加载最佳模型并进行完整的页面级测试：
```bash
python scripts/evaluate.py
```

## 🎲 随机种子与数据划分说明

本项目在数据预处理阶段（`scripts/preprocess.py`）采用了“按作者分组，随机打乱，每名作者取 1 页作测试集，其余 4 页作训练集”的划分策略。

### 1. 随机种子状态（默认）
当前开源版本的 `preprocess.py` **没有设置固定的随机种子**。这意味着每次运行 `python scripts/preprocess.py`，都会对每个作者的 5 个页面进行一次全新的随机划分，测试页的选取可能不同，最终跑出的准确率可能会在 **94.5%~95.5%** 之间轻微浮动。

### 2. 如何固定划分，保证结果可复现
如果你希望每次运行划分后的结果完全一致，请在 `preprocess.py` 的 `main()` 函数开头手动添加一行代码：
```python
random.seed(42)  # 数字 42 可换成任意你喜欢的整数
```
但请注意，由于不同操作系统（Linux vs Windows）下文件遍历顺序不同，即使种子相同，生成的 `metadata.csv` 也可能不同，因此无法保证准确率的完全一致性。

### 3. 严格复现 95.09% 成绩的唯一方法
本项目对外宣传的 **95.09% (969/1019)** 是基于**一份特定的 `metadata.csv`**（即作者在云端训练 50 轮时随机选中并固定的那份测试集划分）得出的。

**如果你想严格复现 95.09% 这个数字，唯一可靠的方法是：**
1. 直接使用作者提供的 `metadata.csv` 文件。
2. 将该文件中的路径修改为适合你本地的路径（可使用脚本一键替换，或手动修改）。
3. 将修改后的 `metadata.csv` 放入你的 `data/features/` 目录。
4. 运行 `python scripts/evaluate.py`。

> ⚠️ **重要提示**：如果在本地重新运行 `preprocess.py` 生成新的 `metadata.csv`，测试集分布会发生改变，将无法复现 95.09% 的准确率。

## 致谢
- 原始论文：Weixin Yang, Lianwen Jin, et al. *DeepWriterID: An End-to-end Online Text-independent Writer Identification System*.
- 数据集：中国科学院自动化研究所 CASIA-OLHWDB 手写数据库。

## 许可证
本项目采用 MIT 许可证。
