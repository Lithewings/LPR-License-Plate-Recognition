
import matplotlib
import pandas as pd




from tensorflow.keras import layers,models,optimizers,callbacks

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.font_manager import FontProperties
import seaborn as sns
import datetime
import json
import sys

# 禁用 TensorBoard 回调以解决日志目录问题
USE_TENSORBOARD = False
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']




# 设置中文字体支持
def setup_chinese_font():
    """配置中文字体支持"""
    try:
        # 尝试使用系统支持的中文字体
        font_path = None

        # 常见中文字体路径
        possible_fonts = [
            'C:/Windows/Fonts/simhei.ttf',  # Windows 黑体
            'C:/Windows/Fonts/simkai.ttf',  # Windows 楷体
            'C:/Windows/Fonts/simsun.ttc',  # Windows 宋体
            '/System/Library/Fonts/PingFang.ttc',  # macOS 苹方
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'  # Linux
        ]

        for font in possible_fonts:
            if os.path.exists(font):
                font_path = font
                break

        if font_path:
            # 创建字体属性
            chinese_font = FontProperties(fname=font_path)
            plt.rcParams['font.family'] = chinese_font.get_name()
            print(f"使用中文字体: {chinese_font.get_name()}")
        else:
            # 回退到支持中文的字体
            plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'sans-serif']
            print("使用默认中文字体支持")

        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"字体设置错误: {str(e)}")
        print("图表可能无法正确显示中文")


# 解决中文路径问题
def safe_image_dataset_from_directory(directory, class_names, **kwargs):
    """解决中文路径问题的替代函数，支持数字文件夹名映射到字母类别"""
    print(f"\n{'=' * 40}")
    print(f"正在加载数据集: {directory}")
    print(f"期望类别: {class_names}")
    print(f"{'=' * 40}")

    # 创建文件夹名到类别名的映射
    # 数字 0-9 直接映射
    folder_to_class = {}

    # 数字类别 0-9
    for i in range(10):
        folder_to_class[str(i)] = str(i)

    # 字母类别：文件夹 10-33 对应 A-Z (跳过I和O)
    letter_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                      'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
                      'U', 'V', 'W', 'X', 'Y', 'Z', '京', '闽', '粤', '苏', '沪', '浙']

    # 文件夹 10 对应第一个字母 'A'，11 对应 'B'，...，33 对应最后一个字母 'Z'
    # 34-39 对应省份
    for idx, cls in enumerate(letter_classes):
        folder_num = 10 + idx
        folder_to_class[str(folder_num)] = cls

    print(f"文件夹到类别的映射: {folder_to_class}")

    # 获取实际存在的目录
    actual_dirs = [d for d in os.listdir(directory)
                   if os.path.isdir(os.path.join(directory, d))]
    print(f"实际存在的目录: {actual_dirs}")

    # 手动遍历目录结构
    image_paths = []
    labels = []
    label_to_index = {name: idx for idx, name in enumerate(class_names)}

    found_classes = set()
    for folder_name in actual_dirs:
        # 检查文件夹是否在映射中
        if folder_name not in folder_to_class:
            print(f"警告: 文件夹 '{folder_name}' 没有对应的类别映射，跳过")
            continue

        class_name = folder_to_class[folder_name]

        # 确保类别在期望的类别列表中
        if class_name not in class_names:
            print(f"警告: 映射类别 '{class_name}' 不在期望类别列表中，跳过")
            continue

        class_dir = os.path.join(directory, folder_name)
        print(f"\n处理文件夹 '{folder_name}' -> 类别 '{class_name}'")

        class_image_count = 0
        for root, _, files in os.walk(class_dir):
            for file_name in files:
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_path = os.path.join(root, file_name)
                    image_paths.append(image_path)
                    labels.append(label_to_index[class_name])
                    class_image_count += 1

        found_classes.add(class_name)
        print(f"  找到 {class_image_count} 张图片")

    # 检查缺失类别
    missing_classes = set(class_names) - found_classes
    if missing_classes:
        print(f"\n警告: 以下类别未找到图片: {', '.join(missing_classes)}")

    print(f"\n总计找到 {len(image_paths)} 张图片")
    print(f"来自 {len(found_classes)} 个类别")
    print('=' * 40)

    # 创建数据集
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.keras.utils.to_categorical(labels, num_classes=len(class_names)))

    def load_and_preprocess_image(path):
        """加载和预处理图像 - 更健壮的版本"""
        path = tf.cast(path, tf.string)
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(img_bytes, channels=0, expand_animations=False)

        # 转换为灰度
        if img.shape[-1] == 3:
            img = tf.image.rgb_to_grayscale(img)
        elif img.shape[-1] == 4:
            img = tf.image.rgba_to_grayscale(img)

        img = tf.image.resize(img, IMG_SIZE)
        return img

    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = tf.data.Dataset.zip((image_ds, label_ds))

    if kwargs.get('shuffle', True):
        dataset = dataset.shuffle(buffer_size=len(image_paths))

    dataset = dataset.batch(kwargs.get('batch_size', 32))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def visualize_results(model, history, val_ds, class_names, model_name):
    # 设置中文字体
    setup_chinese_font()

    # 确保输出目录存在
    os.makedirs("results", exist_ok=True)

    # 提前计算关键指标
    best_epoch = np.argmax(history.history['val_accuracy'])
    best_acc = history.history['val_accuracy'][best_epoch]
    best_loss_epoch = np.argmin(history.history['val_loss'])
    best_loss = history.history['val_loss'][best_loss_epoch]
    final_val_acc = history.history['val_accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]

    # 1. 训练历史可视化 - 使用双坐标轴
    plt.figure(figsize=(16, 8))

    # 创建双坐标轴
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # 准确率曲线（左侧Y轴）
    ax1.plot(history.history['accuracy'], 'b-', linewidth=2, label='训练准确率')
    ax1.plot(history.history['val_accuracy'], 'b--', linewidth=2, label='验证准确率')

    # 损失曲线（右侧Y轴）
    ax2.plot(history.history['loss'], 'r-', linewidth=2, label='训练损失')
    ax2.plot(history.history['val_loss'], 'r--', linewidth=2, label='验证损失')

    # 标记最佳验证准确率
    ax1.plot(best_epoch, best_acc, 'bo', markersize=10, markeredgewidth=2,
             fillstyle='none', label='最佳验证点')

    # 标记最低验证损失
    ax2.plot(best_loss_epoch, best_loss, 'ro', markersize=10, markeredgewidth=2,
             fillstyle='none', label='最低损失点')

    # 设置坐标轴标签和标题
    ax1.set_xlabel('训练轮次 (Epoch)', fontsize=12)
    ax1.set_ylabel('准确率', fontsize=12, color='b')
    ax2.set_ylabel('损失值', fontsize=12, color='r')
    plt.title(f'{model_name} - 训练过程', fontsize=16, pad=20)

    # 设置网格和刻度
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center right',
               fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f'results/{model_name}_training_history.png', dpi=120)
    plt.close()

    # 2. 混淆矩阵 - 专业热力图风格
    y_true = []
    y_pred = []
    all_labels = np.arange(len(class_names))

    for images, labels in val_ds:
        true_labels = np.argmax(labels.numpy(), axis=1)
        y_true.extend(true_labels)

        predictions = model.predict(images, verbose=0)
        pred_labels = np.argmax(predictions, axis=1)
        y_pred.extend(pred_labels)

    # 生成混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    # 计算归一化混淆矩阵（按行）
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(max(10, len(class_names) * 0.8), max(8, len(class_names) * 0.7)))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                cbar_kws={'label': '准确率比例'},
                linewidths=0.5, linecolor='lightgray',
                annot_kws={"size": 10})

    # 设置标签和标题
    plt.title(f'{model_name} - 混淆矩阵', fontsize=16, pad=15)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)

    # 设置刻度
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks + 0.5, class_names, rotation=45, ha='right', fontsize=10)
    plt.yticks(tick_marks + 0.5, class_names, rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(f'results/{model_name}_confusion_matrix.png', dpi=120, bbox_inches='tight')
    plt.close()

    # 3. 类别性能指标 - 水平条形图
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # 提取每个类别的性能指标
    metrics = ['precision', 'recall', 'f1-score']
    performance_data = []
    for cls in class_names:
        if cls in report:  # 只处理存在的类别
            row = [report[cls][metric] for metric in metrics]
            performance_data.append(row)

    # 创建DataFrame
    performance_df = pd.DataFrame(performance_data,
                                  index=class_names,
                                  columns=[m.capitalize() for m in metrics])

    # 添加加权平均值
    weighted_avg = {
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1-score': report['weighted avg']['f1-score']
    }

    # 绘制水平条形图
    fig, ax = plt.subplots(figsize=(12, max(6, len(class_names) * 0.5)))

    # 设置位置和宽度
    y_pos = np.arange(len(class_names))
    bar_width = 0.25

    # 绘制每个指标
    for i, metric in enumerate(performance_df.columns):
        ax.barh(y_pos + i * bar_width, performance_df[metric],
                height=bar_width, alpha=0.8, label=metric)

    # 添加加权平均值线
    ax.axvline(weighted_avg['Precision'], color='b', linestyle='--', alpha=0.5,
               label=f'加权精确率 ({weighted_avg["Precision"]:.2f})')
    ax.axvline(weighted_avg['Recall'], color='g', linestyle='--', alpha=0.5,
               label=f'加权召回率 ({weighted_avg["Recall"]:.2f})')
    ax.axvline(weighted_avg['F1-score'], color='r', linestyle='--', alpha=0.5,
               label=f'加权F1值 ({weighted_avg["F1-score"]:.2f})')

    # 设置标签和标题
    ax.set_yticks(y_pos + bar_width)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel('分数', fontsize=12)
    ax.set_title(f'{model_name} - 类别性能指标', fontsize=16, pad=15)

    # 添加图例
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

    # 添加数值标签
    for i, row in enumerate(performance_df.itertuples()):
        for j, value in enumerate(row[1:]):
            ax.text(value + 0.01, i + j * bar_width, f'{value:.2f}',
                    va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'results/{model_name}_class_performance.png', dpi=120)
    plt.close()

    # 4. 关键指标总结 - 精美表格
    # 创建表格数据
    summary_data = [
        ["模型名称", model_name],
        ["训练轮次", len(history.history['accuracy'])],
        ["最终验证准确率", f"{final_val_acc:.2%}"],
        ["最佳验证准确率", f"{best_acc:.2%} (第{best_epoch + 1}轮)"],
        ["最终验证损失", f"{final_val_loss:.4f}"],
        ["最低验证损失", f"{best_loss:.4f} (第{best_loss_epoch + 1}轮)"],
        ["加权F1值", f"{weighted_avg['F1-score']:.4f}"]
    ]

    # 创建表格图
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    table = ax.table(cellText=summary_data,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.7])

    # 美化表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # 设置标题单元格样式
    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == 0:
            cell.set_facecolor('#f0f0f0')
            cell.set_text_props(weight='bold')

    plt.title(f'{model_name} - 关键指标总结', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_summary_table.png', dpi=120, bbox_inches='tight')
    plt.close()

    # 5. 打印分类报告
    print(f"\n{'=' * 60}")
    print(f"{model_name} 详细分类报告")
    print('=' * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 返回报告数据
    return report


# 1. 数据集配置

TRAIN_PATH = "tf_car_license_dataset/train_images/training-set"
VAL_PATH = "tf_car_license_dataset/train_images/validation-set"

IMG_SIZE = (40, 32)  # 图像尺寸
BATCH_SIZE = 64  # 批量大小

# 2. 创建字符类别映射
# 主模型（字母数字）：34类

main_class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z', '京', '闽', '粤', '苏', '沪', '浙']


# 4. 数据集验证和统计函数
def validate_and_count_dataset(dataset, dataset_name, class_names):
    """
    验证数据集并统计样本数量和类别分布

    :param dataset: 要验证的数据集
    :param dataset_name: 数据集名称（用于打印）
    :param class_names: 类别名称列表
    """
    print(f"\n{'=' * 50}")
    print(f"验证 {dataset_name} 数据集")
    print(f"{'=' * 50}")

    # 统计总样本数
    total_samples = 0
    class_counts = {class_name: 0 for class_name in class_names}

    # 遍历数据集进行统计
    for images, labels in dataset:
        batch_size = images.shape[0]
        total_samples += batch_size

        # 统计每个类别的样本数
        label_indices = tf.argmax(labels, axis=1).numpy()
        for idx in label_indices:
            class_name = class_names[idx]
            class_counts[class_name] += 1

    # 打印统计结果
    print(f"总样本数: {total_samples}")
    print("\n类别分布:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} 样本 ({count / total_samples:.2%})")

    # 可视化类别分布
    plt.figure(figsize=(15, 8))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.title(f"{dataset_name} 类别分布")
    plt.xlabel('类别')
    plt.ylabel('样本数量')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_class_distribution.png')
    plt.close()

    # 显示几张示例图像
    # print("\n随机示例图像:")
    # plt.figure(figsize=(15, 10))
    #
    # # 获取一个批次
    # for images, labels in dataset.take(1):
    #     batch_size = images.shape[0]
    #     num_samples = min(12, batch_size)  # 最多显示12张
    #
    #     for i in range(num_samples):
    #         plt.subplot(3, 4, i + 1)
    #         # 如果是单通道图像，转换为灰度显示
    #         if images[i].shape[-1] == 1:
    #             plt.imshow(images[i].numpy().squeeze(), cmap='gray')
    #         else:
    #             plt.imshow(images[i].numpy().astype("uint8"))
    #
    #         label_idx = tf.argmax(labels[i]).numpy()
    #         plt.title(f"{class_names[label_idx]}")
    #         plt.axis('off')
    #
    # plt.tight_layout()
    # plt.savefig(f'{dataset_name}_sample_images.png')
    # plt.close()
    # print(f"示例图像已保存为 {dataset_name}_sample_images.png")

    return total_samples, class_counts


# 5. 创建主模型（字母数字识别）
def create_main_model():
    inputs = layers.Input(shape=(40, 32, 1))  # 修改为单通道输入

    # 特征提取
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # 分类头
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(main_class_names), activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model



def create_enhanced_lenet():
    """
    基于LeNet-5改进的卷积神经网络，专门用于车牌字符识别（含汉字）。
    改进点：
        - 输入尺寸增大至48x48，保留更多细节。
        - 卷积核数量增加（32-64-128）。
        - 加入批归一化(BN)和Dropout，加速收敛并防止过拟合。
        - 使用全局平均池化(GAP)替代Flatten，减少参数量并增强平移不变性。
        - 全连接层后加入Dropout进一步正则化。
    """

    inputs = layers.Input(shape=(40, 32, 1))

    # 第一卷积块：Conv + BN + ReLU + MaxPool
    x = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # 第二卷积块：Conv + BN + ReLU + MaxPool
    x = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # 第三卷积块：Conv + BN + ReLU（LeNet-5中为C5，此处用卷积代替全连接）
    x = layers.Conv2D(128, (5, 5), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)







    # 全局平均池化（GAP）替代Flatten，减少参数并保留空间信息
    x = layers.GlobalAvgPool2D()(x)

    # 全连接层（F6）
    x = layers.Dense(84, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # 输出层（Softmax）
    outputs = layers.Dense(len(main_class_names), activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model



# 6. 数据预处理函数 - 简化版本
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


# 7. 训练函数
def train_model(model, train_ds, val_ds, model_name, class_names, epochs=50):
    # 编译模型
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 回调函数
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=f'best_{model_name}.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]



    # 训练
    print(f"\n{'=' * 60}")
    print(f"开始训练 {model_name} 模型")
    print(f"类别: {class_names}")
    print(f"训练样本数: {train_samples}")
    print(f"验证样本数: {val_samples}")
    print(f"{'=' * 60}")

    # 训练模型
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=2  # 更简洁的训练输出
    )

    # 评估
    model = models.load_model(f'best_{model_name}.h5')
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\n{model_name} 验证准确率: {val_acc:.4f}")

    return model, history


# ————————————————————————————————————————
# def preprocess_image(image_path, img_size=(40, 32)):
#     """
#     预处理二值化黑白图片，使其符合模型输入要求
#     :param image_path: 图片路径
#     :param img_size: 图像目标尺寸 (高度, 宽度) -> (40, 32)
#     :return: 预处理后的图像数组
#     """
#     # 读取图像 (直接以灰度模式读取)
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#     if img is None:
#         raise ValueError(f"无法读取图片: {image_path}")
#
#     # 二值化处理 (确保是黑白二值图)
#     _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#
#     # 调整尺寸到模型期望的大小 (高度, 宽度) = (40, 32)
#     resized_img = cv2.resize(binary_img, (img_size[1], img_size[0]))
#
#     # 归一化 (将像素值缩放到0-1范围)
#     normalized_img = resized_img / 255.0
#
#     # 添加通道维度 (灰度图只有1个通道)
#     processed_img = np.expand_dims(normalized_img, axis=-1)  # 形状变为 (32, 40, 1)
#
#     # 添加批次维度
#     batched_img = np.expand_dims(processed_img, axis=0)  # 形状变为 (1, 32, 40, 1)
#
#     return batched_img, resized_img


# # 3. 加载训练好的模型
# def load_model(model_path):
#     """加载保存的模型"""
#     return tf.keras.models.load_model(model_path)
#
#
# # 4. 预测并显示结果
# def predict_and_display(model, image_path):
#     """
#     预测单张图片并可视化结果
#     :param model: 加载的模型
#     :param image_path: 要预测的图片路径
#     """
#     # 预处理图像
#     input_img, display_img = preprocess_image(image_path)
#
#     # 进行预测
#     predictions = model.predict(input_img)
#     predicted_class = np.argmax(predictions[0])
#     confidence = np.max(predictions[0])
#
#     # 获取预测结果
#     predicted_label = main_class_names[predicted_class]
#
#     # 获取所有预测结果
#     top34_indices = np.argsort(predictions[0])[::-1][:34]
#     top34_labels = [main_class_names[i] for i in top34_indices]
#     top34_confidences = [predictions[0][i] for i in top34_indices]
#
#     # 可视化结果
#     plt.figure(figsize=(10, 6))
#
#     # 显示原始图像
#     plt.subplot(1, 2, 1)
#     plt.imshow(display_img, cmap='gray')
#     plt.title(f'输入图像 ({display_img.shape[1]}x{display_img.shape[0]})')
#     plt.axis('off')
#
#     # 显示预测结果
#     plt.subplot(1, 2, 2)
#     plt.barh(top34_labels, top34_confidences, color='skyblue')
#     plt.xlabel('置信度')
#     plt.title('预测结果 (Top 3)')
#     plt.xlim(0, 1)
#
#     # 添加置信度文本
#     for i, conf in enumerate(top34_confidences):
#         plt.text(conf + 0.02, i, f'{conf:.2%}', va='center')
#
#     # 添加主预测结果
#     plt.figtext(0.5, 0.05,
#                 f"预测结果: {predicted_label} (置信度: {confidence:.2%})",
#                 ha="center", fontsize=14,
#                 bbox=dict(facecolor='lightgreen', alpha=0.5))
#
#     plt.tight_layout()
#     plt.show()
#
#     # 打印详细预测结果
#     print("\n" + "=" * 50)
#     print(f"预测结果: {predicted_label}")
#     print(f"置信度: {confidence:.2%}")
#     print("\n 所有预测:")
#     for label, conf in zip(top34_labels, top34_confidences):
#         print(f"  {label}: {conf:.2%}")
#     print("=" * 50)
#
#
# # 5. 主函数
# def main():
#     # 加载模型 (替换为您的模型路径)
#     MODEL_PATH = "models/best_main_model.h5"  # 例如: "models/char_classifier.h5"
#     model = load_model(MODEL_PATH)
#
#     # 打印模型摘要
#     print("模型加载成功!")
#     model.summary()
#
#     # 打印输入形状信息
#     input_shape = model.input_shape
#     print(f"\n模型期望输入形状: {input_shape}")
#     print(f"高度: {input_shape[1]}, 宽度: {input_shape[2]}, 通道数: {input_shape[3]}")
#
#     # 测试单张图片 (替换为您的图片路径)
#     while True:
#         image_path = input("\n请输入要识别的图片路径 (或输入 'exit' 退出): ")
#         if image_path.lower() == 'exit':
#             break
#
#         try:
#             predict_and_display(model, image_path)
#         except Exception as e:
#             print(f"错误: {str(e)}")
#             print("请检查图片路径是否正确，图片是否为二值化黑白图像")


# ————————————————————————————————————————————


# 主程序流程
if __name__ == "__main__":
    # main()

    # 创建数据集
    train_ds = safe_image_dataset_from_directory(
        TRAIN_PATH,
        main_class_names
    )
    val_ds = safe_image_dataset_from_directory(
        VAL_PATH,
        main_class_names
    )
    # 预处理数据集
    train_ds = train_ds.map(preprocess)
    val_ds = val_ds.map(preprocess)

    # 验证和统计数据集
    train_samples, train_class_counts = validate_and_count_dataset(train_ds, "训练集", main_class_names)
    val_samples, val_class_counts = validate_and_count_dataset(val_ds, "验证集", main_class_names)

    # 创建模型
    # main_model = create_main_model()
    main_model = create_enhanced_lenet()
    # 打印模型摘要
    print("\n模型架构:")
    main_model.summary()

    # 训练模型
    trained_model, history = train_model(
        main_model,
        train_ds,
        val_ds,
        "new_model",
        main_class_names
    )

    # 可视化训练结果
    visualize_results(trained_model, history, val_ds, main_class_names, "new_model")
