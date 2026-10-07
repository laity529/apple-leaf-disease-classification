# prepare_data.py
import os
import shutil
import random
from math import ceil

# ================= 配置区域 =================
# 原始数据集解压后的路径
source_dir = './apple_source'
# 整理后的目标路径
target_dir = './data'
# 验证集比例（20%的数据用来测试）
val_split_ratio = 0.2


# ===========================================

def reorganize_dataset():
    # 1. 确保目标文件夹存在
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')

    if os.path.exists(train_dir):
        print("数据集已整理，跳过此步骤。如需重新整理，请删除 './data' 文件夹。")
        return

    print("开始整理数据集...")

    # 2. 遍历原始文件夹中的每一个类别文件夹
    classes = os.listdir(source_dir)
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)

        # 确保是文件夹
        if not os.path.isdir(class_path):
            continue

        print(f"正在处理类别: {class_name}")

        # 获取该类别下所有图片
        images = os.listdir(class_path)
        random.shuffle(images)  # 打乱顺序

        # 计算切分点
        split_index = int(len(images) * (1 - val_split_ratio))
        train_images = images[:split_index]
        val_images = images[split_index:]

        # 创建对应的文件夹
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # 3. 复制图片
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copy(src, dst)

        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_class_dir, img)
            shutil.copy(src, dst)

    print("数据集整理完成！")
    print(f"训练集位置: {os.path.abspath(train_dir)}")
    print(f"验证集位置: {os.path.abspath(val_dir)}")


if __name__ == '__main__':
    reorganize_dataset()
