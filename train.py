# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import get_model
from dataset import get_data_loaders
import time  # 导入time模块，用于记录训练时长


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 记录训练日志的文件
    log_file = open('train_log.txt', 'a')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()  # 记录本轮开始时间

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 验证集评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 计算本轮指标
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
        end_time = time.time()
        epoch_time = end_time - start_time  # 本轮训练时长

        # 获取硬件信息（GPU型号）
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        # 打印并保存日志
        log_entry = (
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {epoch_loss:.4f}, "
            f"Accuracy: {epoch_acc:.2f}%, "
            f"LR: {epoch_lr}, "
            f"GPU: {gpu_name}, "
            f"Time: {epoch_time:.2f}s\n"
        )
        print(log_entry, end='')  # 控制台输出
        log_file.write(log_entry)  # 保存到文件

    log_file.close()  # 关闭日志文件
    torch.save(model.state_dict(), 'best.pth')
    print("模型已保存为 best.pth")


if __name__ == '__main__':
    train_loader, val_loader, class_names = get_data_loaders(data_dir='./data', batch_size=32)
    model = get_model(num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)
