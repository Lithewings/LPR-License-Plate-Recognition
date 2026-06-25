# 使用说明

本文档面向想快速运行、验证和复现实验的人。

## 1. 推理流程

推理入口文件：

```bash
python recognization.py
```

默认流程：

1. 读取 `test6.png`
2. 在 HSV 空间提取蓝色区域
3. 通过形态学操作连接车牌字符区域
4. 检测轮廓并筛选候选车牌框
5. 裁剪车牌并保存为 `located_plate.jpg`
6. 对车牌图像进行增强、二值化和字符分割
7. 将分割结果保存到 `characters/`
8. 加载 `models/best_new_model.h5`
9. 对每个字符进行分类并输出最终字符串

## 2. 更换输入图片

打开 `车牌定位分割.py`，找到：

```python
image_path = "test6.png"
```

替换为你的图片路径，例如：

```python
image_path = "test8.png"
```

建议图片满足：

- 车牌区域清晰可见
- 车牌倾斜角度不要过大
- 光照不过曝或过暗
- 蓝牌场景优先

## 3. 更换模型

脚本底部默认使用：

```python
main("model/model_legacy.h5")
```

可切换为：

```python
main("new_model_20260625_190227")
```

## 4. 查看中间结果

| 输出                                   | 说明 |
|--------------------------------------| --- |
| `located_plate.jpg`                  | 定位后裁剪出的车牌 |
| `valid_recognized/plate_located.png` | 候选框筛选可视化 |
| `characters/char_*.png`              | 分割出的单字符 |
| 终端输出                                 | 每个字符的预测类别和置信度 |

## 5. 训练流程


1. 读取训练集和验证集
2. 将数字文件夹映射到数字类别
3. 将字母/省份简称文件夹映射到字符类别
4. 统一输入尺寸为 `40x32`
5. 构建增强版 LeNet/CNN 模型
6. 使用 Adam 优化器训练
7. 保存验证集表现最好的模型
8. 输出训练和评估可视化图表

## 6. 数据集目录格式

训练脚本默认读取：

```text
tf_car_license_dataset/
└── train_images/
    ├── training-set/
    │   ├── 0/
    │   ├── 1/
    │   └── ...
    └── validation-set/
        ├── 0/
        ├── 1/
        └── ...
```

文件夹名使用数字编码，脚本内部会映射为实际类别，例如数字、字母和部分省份简称。

## 7. 评估结果解读

`results/` 下的图表可以用于项目展示：

- `*_training_history.png`：训练/验证准确率和损失曲线，用于观察收敛和过拟合。
- `*_confusion_matrix.png`：混淆矩阵，用于发现易混字符，例如 `0/O`、`1/I`、`8/B` 等。
- `*_class_performance.png`：按类别展示 precision、recall、F1-score。
- `*_summary_table.png`：汇总最佳验证准确率、最低验证损失和最终验证表现。

## 8. 推荐演示顺序

对外展示时，可以按下面顺序讲解：

1. 展示原始车牌图片。
2. 展示 `final_result2/plate_located.png`，说明如何定位候选区域。
3. 展示 `located_plate.jpg`，说明车牌裁剪结果。
4. 展示 `characters/`，说明字符分割结果。
5. 展示终端识别结果，说明模型逐字符预测。
6. 展示 `results/`，说明模型训练和评估闭环。
