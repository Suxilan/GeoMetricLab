# GeoMetricLab

**GeoMetricLab** 是一个专为视觉位置识别 (VPR) 和图像检索任务设计的模块化、可扩展框架。它利用现代深度学习实践，旨在促进度量学习模型的快速实验和部署。

## 🚀 主要特性

- **PyTorch Lightning**: 简化的训练循环，支持多 GPU，代码结构清晰。
- **PyTorch Metric Learning**: 轻松集成最先进的度量学习损失函数、挖掘器 (miners) 和归约器 (reducers)。
- **WandB 集成**: 使用 Weights & Biases 无缝跟踪实验和可视化。
- **模块化设计**: 数据集、骨干网络 (backbones)、聚合器 (aggregators) 和损失函数组件解耦，方便混合搭配实验。
- **可配置性**: 灵活的配置管理，确保实验可复现。

## 📂 项目结构

仓库组织结构如下：

```text
GeoMetricLab/
├── config/         # 配置文件 (实验配置, 消融实验等)
├── data/           # 原始数据存储 (通常被 git 忽略)
├── datasets/       # PyTorch Datasets 和 Lightning DataModules
│   ├── GL3D/       # GL3D 数据集实现
│   └── oxford/     # Oxford RobotCar 数据集实现
├── eval/           # 评估脚本和指标
├── losses/         # 自定义损失函数和包装器
├── models/         # 模型架构
│   ├── backbone/   # 特征提取器 (ResNet, ViT 等)
│   └── aggregator/ # 聚合层 (NetVLAD, GeM 等)
├── scripts/        # 数据处理或启动任务的实用脚本
├── tests/          # 单元测试
├── train/       # LightningModules 和训练逻辑
└── utils/          # 通用工具函数 (I/O, 日志等)
```

## 🛠️ 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/GeoMetricLab.git
cd GeoMetricLab

# 安装依赖
pip install -r requirements.txt

# 以编辑模式安装包
pip install -e .
```

## 🏃 使用方法

### 训练

要开始训练实验，请使用指定的配置文件运行训练脚本：

```bash
# 运行基线实验
python -m training.train --config config/experiments/baseline.yaml

# 运行带有特定覆盖参数的实验
python -m training.train model.backbone=resnet50 training.batch_size=32
```

### 测试 / 评估

在测试集上评估已训练的模型：

```bash
python -m eval.test --checkpoint checkpoints/best_model.ckpt --dataset oxford
```

## 🤝 贡献

欢迎贡献！请确保您的代码遵循项目的编码规范：
- 所有函数参数和返回值必须使用 **类型提示 (Type Hints)**。
- 类和函数需包含 **Google 风格的文档字符串**。
- 提交 PR 前请运行测试。

## 📄 许可证

本项目采用 MIT 许可证。
