# dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_data_loaders(data_dir, batch_size=32, img_size=224, val_split=0.2):
    """
    获取训练集和验证集的数据加载器
    :param data_dir: 数据集根目录，包含train和val文件夹
    :param batch_size: 批次大小
    :param img_size: 图像缩放大小
    :param val_split: 验证集比例（如果数据未预先划分）
    :return: train_loader, val_loader, class_names
    """
    # 定义数据预处理步骤
    # 训练集使用数据增强：随机翻转、旋转、颜色抖动等，以提高模型鲁棒性
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),      # 随机旋转±10度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
        transforms.ToTensor(),              # 转为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 验证集只进行常规预处理
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    # 假设数据已经划分为train和val文件夹
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=val_transform)

    # 如果数据没有预先划分，可以使用 random_split
    # full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    # train_size = int((1 - val_split) * len(full_dataset))
    # val_size = len(full_dataset) - train_size
    # train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    print(f"类别: {class_names}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(val_dataset)}")

    return train_loader, val_loader, class_names

if __name__ == '__main__':
    # 测试数据加载器
    data_dir = './data'  # 替换为你的数据集路径
    train_loader, val_loader, class_names = get_data_loaders(data_dir)
    images, labels = next(iter(train_loader))
    print(f"一个批次图像的形状: {images.shape}")  # [batch_size, 3, img_size, img_size]
    print(f"一个批次标签的形状: {labels.shape}")  # [batch_size]
