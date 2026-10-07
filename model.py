# model.py
import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes=4):
    """
    加载预训练的 ResNet18 模型，并修改最后一层以适应分类任务
    :param num_classes: 分类类别数（苹果病害类别数，默认4类）
    :return: 修改后的模型
    """
    # 加载预训练的 ResNet18 模型
    model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')

    # 冻结所有层的参数（可选，如果数据量小，可以只训练最后一层）
    for param in model.parameters():
        param.requires_grad = False

    # 修改最后一层（全连接层）以匹配你的类别数
    # ResNet18 的最后一层是 model.fc，输入特征维度为 512
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


if __name__ == '__main__':
    # 测试模型
    model = get_model(num_classes=4)
    print(model)
