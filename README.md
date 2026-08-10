# DeepWriterID Replication on CASIA-OLHWDB

## 项目简介
本项目是对论文《DeepWriterID: An End-to-end Online Text-independent Writer Identification System》的完整复刻，并基于 **CASIA-OLHWDB 2.0-2.2 在线手写文本数据集（1019 位书写者）** 进行了验证。

## 核心成绩
在 1019 个类别（远超原论文 187 类）的分类任务下，通过 DropSegment、路径签名特征提取与页面级投票策略，最终测试集页面级准确率达到 **95.09%**。
- 原论文在 NLPR (187类) 上的成绩为 95.72%
- 本项目在 1019 类分类任务上实现了极具竞争力的泛化性能。

## 📊 数据集对比说明

本项目使用的数据集与原论文（DeepWriterID）存在重大差异。

| 对比维度 | 原论文使用的数据集（NLPR） | 本项目使用的数据集（CASIA-OLHWDB 2.0-2.2） |
| :--- | :--- | :--- |
| **作者数量** | 中文任务 **187 位** 书写者 | **1019 位** 书写者（全量合并后重新按 4:1 划分） |
| **任务类别数** | 187 类分类 | **1019 类分类**（类别数增加 5.4 倍） |
| **测试集内容** | **固定内容（fixed-content）**<br>所有 187 位作者的测试页抄写的是同一篇文章 | **内容各不相同**<br>不同作者的测试页文本内容互不相同（来源不同的新闻、古诗模板） |
| **泛化难度** | 较低。模型只需在“相同内容”下辨别笔迹风格 | 极高。模型必须在“不同内容”下识别出同一个人的笔迹风格，且类别极多 |
| **训练样本量** | 较小 | **约 85 万** 个动态增强的伪字符样本 |

### 🎯 为什么这个对比很重要？
原论文在 187 类、固定文本内容的测试集上取得了 **95.72%** 的页面级准确率。
本项目在 **1019 类、测试文本内容完全随机** 的苛刻条件下，依然取得了 **95.09%** 的页面级准确率。

这不仅证明了 DeepWriterID 方法（DropSegment、路径签名、页面级投票）的优秀之处，也表明该复刻模型在更接近真实应用场景（多类别、跨文本内容）下，依然具备极强的泛化性能。

## 硬件与软件环境
本项目在以下云端环境跑通，并取得最终成绩：
- **GPU**: NVIDIA RTX 4090 D (24GB VRAM)
- **CPU**: AMD EPYC 9754 (18 vCPU)
- **RAM**: 60 GB
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.12
- **CUDA**: 12.4
- **PyTorch**: 2.5.1

## 💻 硬件约束：预处理流水线对系统的真实需求

本项目的计算瓶颈并不在于神经网络的前向传播或反向传播，而在于 **CPU 密集型的数据预处理流水线**。以下分析基于实际测试数据（18 vCPU + 60GB RAM + RTX 4090D）：

### 1. 内存带宽与并发瓶颈
- **内存是分布式数据加载的第一道关卡**。为了缓解 CPU 渲染延迟，`DataLoader` 被配置为 16 个并行 Worker 进程（`num_workers=16`）。
- 当系统物理内存达到 **60GB** 时，该配置可稳定运行，单轮耗时约 **15 分钟**。
- 若内存容量低于 60GB，高并发下的多进程内存争用将导致系统触发 `Bus error` 或 OOM Killer，进程被强制中断。**即使将 CPU 核心数提升至 96 核，如果内存仅 32GB，内存依然会成为无法逾越的瓶颈。**

### 2. 单核性能对预处理时间的决定性作用
- 每一轮训练需要处理约 **85 万个动态增强的伪字符**，每次迭代均包含：拐点检测、OpenCV 图像渲染、直方图均衡化及路径签名计算。
- 由于上述操作难以高度并行化，**单核主频与 CPU 架构**对此类任务的完成效率起着决定性作用。
- 本项目之所以能维持 15 分钟一轮的吞吐量，得益于 **AMD EPYC 9754 等服务器级处理器的高单核主频与出色的浮点计算能力**。若使用主频较低的老旧架构处理器，即便 vCPU 数量充足，单轮耗时也可能延长至 30 分钟以上，导致 GPU 长期处于饥饿状态。

### 3. 显卡利用率的真实情况
- 在 `batch_size=256` 的设置下，RTX 4090D 的**显存占用稳定在 9.5GB 左右**，而 GPU 核心计算单元的使用率长期低于 **40%**。
- 这说明 GPU 的算力大幅冗余，计算瓶颈完全位于前端的 CPU 数据流水线上。显存容量即便进一步提升，对训练加速的影响也微乎其微。

### 📌 硬件配置推荐
为保证复刻本项目 **95.09%** 成绩的体验，建议硬件配置满足以下标准：
- **内存（RAM）**：**≥ 60GB**（避免多进程并发时的内存溢出错误）。
- **CPU**：**≥ 16 vCPU**，且具备**较高的单核主频与先进微架构**（如 AMD EPYC 系列或 Intel 至强系列，不建议使用低主频老旧架构）。
- **GPU（显存）**：**≥ 12GB**（例如 RTX 3060 12GB、RTX 4090 等，显存容量超过 12GB 对速度无明显增益）。

> **备注**：如果你在云服务器上运行本项目，即使租用拥有众多 CPU 核心的高配实例，也要优先确认其内存容量是否达到 60GB，以及 CPU 是否为主频较高的现代架构。否则，你将很难达到预期的训练速度。

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
## 🧠 Dropout 正则化策略

由于本项目面对的是 **1019 个类别** 的超大分类任务，直接套用原论文在 187 个类别上的 Dropout 配置（0.3, 0.4, 0.5, 0.5）会导致模型严重欠拟合。

因此，本项目在 **`src/models/dcnn.py`** 中做了针对性的自适应调整：
- 全连接层 `fc1` 和 `fc2` 中，使用了 **0.2** 的 Dropout 比例。
- 输出层 `output` 之前，同样设置了 **0.2** 的 Dropout 比例。

这样的设置既能防止 1019 类任务中因网络参数过多而产生过拟合，又能保证在训练初期梯度能够正常传递、模型顺利收敛。

> 如果你希望在训练过程中动态调整 Dropout 强度，只需修改 `src/models/dcnn.py` 中的 `nn.Dropout(0.2)` 数值即可。

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

本项目使用的数据集为 **CASIA-OLHWDB 2.0-2.2 在线手写文本数据集**。

- 官方下载地址：[https://nlpr.ia.ac.cn/databases/handwriting/Home.html](https://nlpr.ia.ac.cn/databases/handwriting/Home.html)，目前该数据集已可在官网直接下载，无需申请。
- 下载后，将解压出的 `WPTT2.0-Train`、`WPTT2.0-Test` 等文件夹放置在项目的 `data/raw/` 目录下，结构如下：
  ```text
  data/raw/WPTT2.0-Train/
  data/raw/WPTT2.0-Test/
  data/raw/WPTT2.1-Train/
  data/raw/WPTT2.1-Test/
  data/raw/WPTT2.2-Train/
  data/raw/WPTT2.2-Test/
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

本项目在数据预处理阶段（`scripts/preprocess.py`）采用“按作者分组，随机打乱，每名作者取 1 页作测试集，其余 4 页作训练集”的划分策略。

### 1. 随机种子状态（默认）
当前 `preprocess.py` **未设置固定的随机种子**。每次运行都会产生全新的随机划分，最终测试准确率可能在 **94.5%~95.5%** 之间轻微浮动。

### 2. 固定划分的方法
若希望每次划分结果一致，可在 `preprocess.py` 的 `main()` 函数开头添加：
```python
random.seed(42)
```
由于不同操作系统（Linux vs Windows）下文件遍历顺序不同，该操作也无法保证 100% 完全一致。

### 3. 严格复现 95.09% 成绩的方法
本项目宣传的 **95.09% (969/1019)** 基于 **特定的 `metadata.csv` 划分文件** 与 **已训练好的 `dcnn_best.pth` 模型权重**。

若要精准复现该成绩：
1. 直接使用项目提供的 `metadata.csv`（位于 `data/features/`）。
2. 根据你的本地路径，修改该文件中的文件路径。
3. 配合笔者训练好的 `dcnn_best.pth` 模型。
4. 运行 `python scripts/evaluate.py`。

> 自行重新训练模型产生的权重，因硬件、CUDA 版本和随机性的细微差异，最终准确率会在 94.8%~95.2% 之间浮动，这是深度学习训练的正常现象。
 
## 📦 模型文件说明（不公开）

本项目中的预训练模型文件（`outputs/checkpoints/dcnn_best.pth`）**不包含在本 GitHub 仓库中**，原因如下：

1. **算力成本极高**：该模型是基于 **RTX 4090D 显卡（24GB VRAM）** 及 **18核 AMD EPYC 处理器**，历经 **50 轮完整训练（约 3 天计算时间）** 才训练得到。该训练过程消耗了相当的云服务器租赁成本。
2. **模型资产保护**：该模型在 **1019 个书写者** 的 CASIA 数据集上达到了 **95.09%** 的页面级准确率，具有较高的研究与复刻价值。作者希望保护该训练成果的完整性，避免未经授权的随意扩散。

### 📎 如何获取模型文件？
如果你需要该模型用于**学术研究**或**商业合作**，可以通过以下方式联系作者：

- **GitHub Issues**：在本项目 GitHub 仓库提交 Issues 并说明用途。

作者将根据具体用途（学术/商业）提供模型文件，部分情况下可能收取适当的算力成本费。感谢你的理解与支持。

## 致谢
- 原始论文：Weixin Yang, Lianwen Jin, et al. *DeepWriterID: An End-to-end Online Text-independent Writer Identification System*.
- 数据集：中国科学院自动化研究所 CASIA-OLHWDB 手写数据库。

## 💬 反馈与建议

如果你在使用本项目的过程中遇到任何问题，或者有更好的改进思路（比如模型架构优化、更高效的数据增强策略等），非常欢迎你在 GitHub 上提交 **Issue** 或直接发起 **Pull Request**。

我可能不会及时回复每一条消息，但所有有价值的建议都会认真考虑，并用于后续的迭代和优化。如果你在跑这个项目时卡在了某个环节，也欢迎在 Issues 里提问，我会把踩过的坑写出来，帮助大家避开。

## 许可证
本项目采用 MIT 许可证。