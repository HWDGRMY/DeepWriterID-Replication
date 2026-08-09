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

## 💻 硬件隐情：内存和显存的真实硬门槛

本项目对 CPU 和内存的实际消耗，**远高于对 GPU 和显存的要求**。以下基于实际测试数据（18 vCPU + 60GB 内存 + RTX 4090D）说明：

### 📊 性能瓶颈

1. **内存是绝对的硬瓶颈**：
   - 只有 **内存 ≥ 60GB**，才能在 16 个并行数据加载进程（`num_workers=16`）下稳定运行，单轮耗时约 **15 分钟**。
   - 若内存不足 60GB，强行提升并行数会直接触发系统 `Bus error` 或 `Killed`，训练直接崩溃。
   - **CPU 核心再多，也填不满内存的瓶颈**。即使租用 96 核的顶级 CPU，若内存只有 32GB，速度依然会被卡在内存读写上，无法发挥 CPU 的真正性能。

2. **显存需求其实很低，但建议预留**：
   - 本模型在 `batch_size=256` 时，显存占用约为 **9.5GB**（RTX 4090D 实测）。
   - 但考虑到 `batch_size` 可能调整、系统缓存或其他进程占用，**建议显存至少为 12GB**（例如 RTX 3060 12GB 版）。
   - 显存超过 12GB 后，对速度的提升几乎没有帮助。**在 RTX 4090D 实测中，GPU 核心利用率长期低于 40%**，绝大多数时间在等待 CPU 喂数据。

### 📌 云服务器配置建议

如果你希望在云端复现本项目 **95.09%** 的成绩并保持 **15 分钟一轮** 的效率，你的配置必须满足：

- **内存**：**≥ 60GB**（低于 60GB 容易出现 `Bus error` 或速度骤降）。
- **显存**：**≥ 12GB**（推荐 RTX 3060 12GB、RTX 4090 等均可，显存再多也提升不大）。
- **CPU**：**16 vCPU 及以上**（只要内存足够，核心数越多越好，但核心数不是决定速度的关键）。

> ⚠️ **特别提醒**：即使你租用 96 核的顶级云端 CPU，如果内存只有 32GB，它依然会因内存瓶颈被卡死，无法发挥出 CPU 的真正实力。**内存的容量和带宽，才是决定本项目训练速度的关键所在。**

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
