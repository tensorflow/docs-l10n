# TensorFlow Lite Model Maker

## 概览

借助 TensorFlow Lite Model Maker 库，可以简化使用自定义数据集训练 TensorFlow Lite 模型的过程。该库使用迁移学习来减少所需的训练数据量并缩短训练时间。

## 支持的任务

目前，Model Maker 库支持以下 ML 任务。点击以下链接可获取有关如何训练模型的指南。

支持的任务 | 任务效用
--- | ---
图像分类[指南](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification) | 将图像分成预定义类别。
文字分类[指南](https://www.tensorflow.org/lite/tutorials/model_maker_text_classification) | 将文字分成预定义类别。
BERT 问答[指南](https://www.tensorflow.org/lite/tutorials/model_maker_question_answer) | 使用 BERT 在特定上下文中查找给定问题的答案。

## 端到端示例

借助 Model Maker，仅仅通过几行代码即可使用自定义数据集训练 TensorFlow Lite 模型。例如，以下就是训练图像分类模型的步骤。

```python
# Load input data specific to an on-device ML app.
data = ImageClassifierDataLoader.from_folder('flower_photos/')
train_data, test_data = data.split(0.9)

# Customize the TensorFlow model.
model = image_classifier.create(train_data)

# Evaluate the model.
loss, accuracy = model.evaluate(test_data)

# Export to Tensorflow Lite model and label file in `export_dir`.
model.export(export_dir='/tmp/')
```

有关详情，请参阅[图像分类指南](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification)。

## 安装

可以通过两种方式安装 Model Maker。

- 安装预构建的 pip 软件包。

```shell
pip install tflite-model-maker
```

如果想安装 Nightly 版本，请使用以下命令：

```shell
pip install tflite-model-maker-nightly
```

- 从 GitHub 克隆源代码并安装。

```shell
git clone https://github.com/tensorflow/examples
cd examples/tensorflow_examples/lite/model_maker/pip_package
pip install -e .
```
