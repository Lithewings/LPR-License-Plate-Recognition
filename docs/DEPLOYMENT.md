# 部署指南

本文档说明如何在本地或服务器环境中部署并运行车牌识别项目。

## 1. 环境要求

推荐环境：

- Python 3.10 或 3.11
- Windows 10/11、macOS 或 Linux
- CPU 可运行；如需加速训练，可使用支持 TensorFlow 的 GPU 环境
- 本地桌面运行推荐，因为当前脚本会调用 OpenCV/Matplotlib 图形窗口

## 2. 获取项目

```bash
git clone <your-repo-url>
cd <repo-name>
```

如果没有上传完整数据集或模型文件，请手动放置：

```text
models/best_new_model.h5
tf_car_license_dataset/train_images/training-set
tf_car_license_dataset/train_images/validation-set
```

## 3. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. 推理部署

推理入口：

```bash
python recognization.py
```

默认配置位于脚本底部：

```python
image_path = "test6.png"
main("models/model_legacy.h5")
```

更换测试图片时，修改 `image_path`。更换模型时，修改 `main()` 的模型路径。

运行成功后会输出：

- `located_plate.jpg`：裁剪出的车牌区域
- `characters/`：分割后的字符图片
- `final_result2/plate_located.png`：候选车牌区域可视化
- 终端识别结果：字符序列和平均置信度

## 5. 训练部署

训练入口：

```bash
python model_training.py
```

默认数据路径：

```python
TRAIN_PATH = "tf_car_license_dataset/train_images/training-set"
VAL_PATH = "tf_car_license_dataset/train_images/validation-set"
```

训练输出：

- `best_new_model.h5`
- `results/*_training_history.png`
- `results/*_confusion_matrix.png`
- `results/*_class_performance.png`
- `results/*_summary_table.png`

## 6. 无 GUI 环境部署建议

当前 `车牌定位分割.py` 中的 `visualize_step()` 会调用 `cv2.imshow()`，`predict_characters()` 会调用 `plt.show()`。如果部署到服务器、Docker、CI 或远程终端，可能出现窗口无法打开的问题。

推荐改造方向：

- 将 `cv2.imshow()` 替换为 `cv2.imwrite()`。
- 将 `plt.show()` 替换为 `plt.savefig()`。
- 将中间结果统一输出到 `outputs/`。
- 为脚本增加命令行参数，例如 `--image`、`--model`、`--no-gui`。

## 7. GitHub 部署展示建议

GitHub 单个文件限制为 100 MB。当前模型文件体积较小，可以直接展示；数据集和压缩包建议按需处理：

- 小型样例数据可以保留在仓库中，方便评审者快速运行。
- 完整数据集建议放到 Release、网盘、Kaggle、Hugging Face Dataset 或 Git LFS。
- 本地虚拟环境 `python/` 不应上传。

## 8. 常见问题

### 找不到 TensorFlow

确认已激活虚拟环境，并执行：

```bash
pip install -r requirements.txt
```

### OpenCV 窗口乱码或无法弹出

Windows 中文窗口标题可能受终端编码影响。识别逻辑不依赖窗口标题；如果影响运行，可将可视化窗口改为保存图片。

### 定位不到车牌

当前算法主要针对蓝牌。可尝试：

- 更换光照更清晰的图片
- 调整 HSV 蓝色阈值
- 放宽候选框长宽比和面积筛选条件
- 对黄牌、绿牌增加独立颜色阈值
