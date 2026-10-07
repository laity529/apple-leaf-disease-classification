# test.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import get_model
from dataset import get_data_loaders
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import random  # 导入random模块，用于随机抽样

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def evaluate_model(model, val_loader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算准确率
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"测试集准确率: {accuracy:.4f}")

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("混淆矩阵:")
    print(cm)

    # 打印分类报告（精确率、召回率、F1分数）
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('混淆矩阵')
    plt.show()


def visualize_predictions(model, val_loader, class_names, num_images=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 获取验证集的dataset（而非固定批次）
    val_dataset = val_loader.dataset

    # 随机选择num_images个索引（从验证集总样本中抽样）
    total_samples = len(val_dataset)
    indices = random.sample(range(total_samples), num_images)

    # 根据索引获取对应的图像和标签
    images = torch.stack([val_dataset[i][0] for i in indices])
    labels = torch.tensor([val_dataset[i][1] for i in indices])

    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        ax = plt.subplot(2, 3, i + 1)
        img = images[i].cpu().numpy().transpose((1, 2, 0))  # 转换为HWC格式
        img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])  # 反归一化
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        plt.title(f"真实: {true_label}\n预测: {pred_label}")
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # 获取数据加载器
    _, val_loader, class_names = get_data_loaders(data_dir='./data', batch_size=32)

    # 加载训练好的模型
    model = get_model(num_classes=len(class_names))
    model.load_state_dict(torch.load('best.pth', weights_only=True))

    # 评估模型
    evaluate_model(model, val_loader, class_names)
    # 预测结果可视化
    visualize_predictions(model, val_loader, class_names, num_images=6)