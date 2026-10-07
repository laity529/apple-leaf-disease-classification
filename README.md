# 果园病虫害发病症状识别与分类模型设计

> 基于 PyTorch 与迁移学习的苹果叶片病害四分类模型，验证集准确率 **99.06%**。

## 项目背景

苹果叶片病害早期症状相似，人工判读依赖经验且效率低。本项目用深度学习做叶片图像自动分类，覆盖苹果常见的 3 种病害与健康状态，用于辅助果园病害快速筛查。

## 效果展示

| 训练曲线 | 混淆矩阵 |
|---|---|
| ![训练曲线](assets/Figure_1.png) | ![混淆矩阵](assets/Figure_2.png) |

| 预测结果可视化 | 样本示例 |
|---|---|
| ![预测可视化](assets/test.png) | ![样本](assets/train.png) |

> 图片来自 `result/` 目录，请在上传前确认 `Figure_1` / `Figure_2` 分别对应哪张（训练曲线 / 混淆矩阵），对不上就调换顺序。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据集
#    下载 PlantVillage 数据集中的苹果叶片部分，解压到 ./apple_source/
#    目录结构：
#      apple_source/
#        ├── Apple___Apple_scab/
#        ├── Apple___Black_rot/
#        ├── Apple___Cedar_apple_rust/
#        └── Apple___healthy/

# 3. 划分训练集 / 验证集（按类别 8:2）
python prepare_data.py

# 4. 训练（50 epoch，RTX 4060 Laptop 约 9 分钟）
python train.py

# 5. 评估（输出准确率、混淆矩阵、分类报告）
python test.py
```

## 数据集

使用公开数据集 **PlantVillage** 的苹果叶片部分，共 4 类 3171 张图像。

| 类别 | 训练集 | 验证集 | 合计 |
|---|---|---|---|
| Apple___Apple_scab（疮痂病） | 504 | 126 | 630 |
| Apple___Black_rot（黑腐病） | 496 | 125 | 621 |
| Apple___Cedar_apple_rust（锈病） | 220 | 55 | 275 |
| Apple___healthy（健康） | 1316 | 329 | 1645 |
| **合计** | **2536** | **635** | **3171** |

下载方式：（填写你的来源，如 Kaggle PlantVillage 链接或网盘）

**划分方式**：`prepare_data.py` 逐类随机打乱后按 8:2 切分，各类别独立划分，因此训练集与验证集的类别比例一致。

**注意**：本项目只划分了训练集与验证集，未设置独立测试集，报告中的准确率均为**验证集**指标。

## 方法

- **骨干网络**：ResNet18（ImageNet 预训练权重 `ResNet18_Weights.IMAGENET1K_V1`）
- **迁移策略**：冻结全部预训练层参数，仅替换最后的全连接层为 4 分类输出并训练
- **数据增强**（仅训练集）：随机水平翻转、随机旋转 ±10°、颜色抖动（亮度/对比度/饱和度 0.2）
- **预处理**：Resize 到 224×224，按 ImageNet 均值方差归一化
- **训练配置**：Adam（lr=1e-3）、CrossEntropyLoss、batch size 32、50 个 epoch
- **评估指标**：准确率、混淆矩阵、分类报告（Precision / Recall / F1）

## 实验结果

| 指标 | 结果 |
|---|---|
| 验证集准确率（第 50 轮） | 99.06% |
| 验证集最高准确率（第 46/47/49 轮） | 99.53% |

**各类别 F1**：（跑一次 `python test.py`，把分类报告结果填在这里）

**错误分析**：（把混淆矩阵里的主要误判填在这里，例如某两类之间存在混淆，原因是早期病斑面积小、特征相似）

**已知问题**：
1. **类别不平衡**：健康样本 1645 张，锈病仅 275 张（约 6:1）。改进方向：类别加权损失、过采样 / 欠采样。
2. **权重保存的是最后一轮**：`train.py` 在训练循环结束后保存模型，未取验证集最优的一轮。改进方向：按验证集准确率保存最优权重。
3. **无独立测试集**：应划分 train / val / test 三部分，用测试集报告最终指标。

## 项目结构

```
├── model.py          # ResNet18 迁移学习模型定义
├── dataset.py        # 数据加载与增强
├── prepare_data.py   # 数据集 8:2 划分
├── train.py          # 训练主循环与日志
├── test.py           # 评估、混淆矩阵、预测可视化
├── train_log.txt     # 50 轮训练日志
├── assets/           # 结果图
├── apple_source/     # 原始数据集（不上传）
├── data/             # 划分后的数据集（不上传）
└── best.pth          # 模型权重（不上传，可用 Git LFS 或 Release 发布）
```

## 技术栈

Python · PyTorch · torchvision · scikit-learn · matplotlib · seaborn

## TODO

- [ ] 划分独立测试集，用测试集报告最终指标
- [ ] 保存验证集最优权重而非最后一轮
- [ ] 用类别加权损失或采样策略缓解类别不平衡
- [ ] 导出 ONNX 做推理加速
- [ ] Gradio / Streamlit 搭建在线 demo
