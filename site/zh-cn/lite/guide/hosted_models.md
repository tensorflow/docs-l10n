# 托管模型

下面是一个不完整列表，其中包括为兼容 TensorFlow Lite 而进行优化的预训练模型。

要开始选择模型，请访问带有端到端示例的<a href="../models">模型</a>页面，或者[从 TensorFlow Hub 中选择 TensorFlow Lite 模型](https://tfhub.dev/s?deployment-format=lite)。

注：适用于某个给定应用的最佳模型取决于您的要求。例如，某些应用可能受益于较高的准确率，而另一些应用则需要较小的模型大小。您应该使用各种模型来测试您的应用，在大小、性能和准确率之间找到最佳平衡。

## 图像分类

有关图像分类的更多信息，请参阅<a href="../models/image_classification/overview.md">图像分类</a>。探索 TensorFlow Lite Task 库，以获取有关如何在短短几行代码中[集成图像分类模型](../inference_with_metadata/task_library/image_classifier)的说明。

### 量化模型

<a href="../performance/post_training_quantization">量化</a>图像分类模型可以提供最小的模型大小和最快的性能，但以牺牲准确率为代价。性能值在运行 Android 10 的 Pixel 3 上测得。

您可以在 TensorFlow Hub 中找到许多[量化模型](https://tfhub.dev/s?deployment-format=lite&module-type=image-classification&q=quantized)，并获取更多模型信息。

模型名称 | 论文和模型 | 模型大小 | Top-1 准确率 | Top-5 准确率 | CPU（4 线程） | NNAPI
--- | :-: | --: | --: | --: | --: | --:
Mobilenet_V1_0.25_128_quant | [论文](https://arxiv.org/pdf/1712.05877.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128_quant.tgz) | 0.5 Mb | 39.5% | 64.4% | 0.8 毫秒 | 2 毫秒
Mobilenet_V1_0.25_160_quant | [论文](https://arxiv.org/pdf/1712.05877.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_160_quant.tgz) | 0.5 Mb | 42.8% | 68.1% | 1.3 毫秒 | 2.4 毫秒
Mobilenet_V1_0.25_192_quant | [论文](https://arxiv.org/pdf/1712.05877.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_192_quant.tgz) | 0.5 Mb | 45.7% | 70.8% | 1.8 毫秒 | 2.6 毫秒
Mobilenet_V1_0.25_224_quant | [论文](https://arxiv.org/pdf/1712.05877.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224_quant.tgz) | 0.5 Mb | 48.2% | 72.8% | 2.3 毫秒 | 2.9 毫秒
Mobilenet_V1_0.50_128_quant | [论文](https://arxiv.org/pdf/1712.05877.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_128_quant.tgz) | 1.4 Mb | 54.9% | 78.1% | 1.7 毫秒 | 2.6 毫秒
Mobilenet_V1_0.50_160_quant | [论文](https://arxiv.org/pdf/1712.05877.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_160_quant.tgz) | 1.4 Mb | 57.2% | 80.5% | 2.6 毫秒 | 2.9 毫秒
Mobilenet_V1_0.50_192_quant | [论文](https://arxiv.org/pdf/1712.05877.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_192_quant.tgz) | 1.4 Mb | 59.9% | 82.1% | 3.6 毫秒 | 3.3 毫秒
Mobilenet_V1_0.50_224_quant | [论文](https://arxiv.org/pdf/1712.05877.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_224_quant.tgz) | 1.4 Mb | 61.2% | 83.2% | 4.7 毫秒 | 3.6 毫秒
Mobilenet_V1_0.75_128_quant | [论文](https://arxiv.org/pdf/1712.05877.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_128_quant.tgz) | 2.6 Mb | 55.9% | 79.1% | 3.1 毫秒 | 3.2 毫秒
Mobilenet_V1_0.75_160_quant | [论文](https://arxiv.org/pdf/1712.05877.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_160_quant.tgz) | 2.6 Mb | 62.4% | 83.7% | 4.7 毫秒 | 3.8 毫秒
Mobilenet_V1_0.75_192_quant | [论文](https://arxiv.org/pdf/1712.05877.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_192_quant.tgz) | 2.6 Mb | 66.1% | 86.2% | 6.4 毫秒 | 4.2 毫秒
Mobilenet_V1_0.75_224_quant | [论文](https://arxiv.org/pdf/1712.05877.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_224_quant.tgz) | 2.6 Mb | 66.9% | 86.9% | 8.5 毫秒 | 4.8 毫秒
Mobilenet_V1_1.0_128_quant | [论文](https://arxiv.org/pdf/1712.05877.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_128_quant.tgz) | 4.3 Mb | 63.3% | 84.1% | 4.8 毫秒 | 3.8 毫秒
Mobilenet_V1_1.0_160_quant | [论文](https://arxiv.org/pdf/1712.05877.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_160_quant.tgz) | 4.3 Mb | 66.9% | 86.7% | 7.3 毫秒 | 4.6 毫秒
Mobilenet_V1_1.0_192_quant | [论文](https://arxiv.org/pdf/1712.05877.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_192_quant.tgz) | 4.3 Mb | 69.1% | 88.1% | 9.9 毫秒 | 5.2 毫秒
Mobilenet_V1_1.0_224_quant | [论文](https://arxiv.org/pdf/1712.05877.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz) | 4.3 Mb | 70.0% | 89.0% | 13 毫秒 | 6.0 毫秒
Mobilenet_V2_1.0_224_quant | [论文](https://arxiv.org/abs/1806.08342)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz) | 3.4 Mb | 70.8% | 89.9% | 12 毫秒 | 6.9 毫秒
Inception_V1_quant | [论文](https://arxiv.org/abs/1409.4842)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_224_quant_20181026.tgz) | 6.4 Mb | 70.1% | 89.8% | 39 毫秒 | 36 毫秒
Inception_V2_quant | [论文](https://arxiv.org/abs/1512.00567)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/inception_v2_224_quant_20181026.tgz) | 11 Mb | 73.5% | 91.4% | 59 毫秒 | 18 毫秒
Inception_V3_quant | [论文](https://arxiv.org/abs/1806.08342)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/inception_v3_quant.tgz) | 23 Mb | 77.5% | 93.7% | 148 毫秒 | 74 毫秒
Inception_V4_quant | [论文](https://arxiv.org/abs/1602.07261)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/inception_v4_299_quant_20181026.tgz) | 41 Mb | 79.5% | 93.9% | 268 毫秒 | 155 毫秒

注：模型文件包括 TF Lite FlatBuffer 和 Tensorflow 冻结计算图。

注：性能数值来自在 Pixel-3 (Android 10) 上进行的基准测试。准确率数值使用 [TFLite 图像分类评估工具](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification)计算得出。

### 浮点模型

浮点模型可以提供最佳的准确率，但以牺牲模型的大小和性能为代价。<a href="../performance/gpu">GPU 加速</a>需要使用浮点模型。性能值在运行 Android 10 的 Pixel 3 上测得。

您可以在 TensorFlow Hub 中找到许多[图像分类模型](https://tfhub.dev/s?deployment-format=lite&module-type=image-classification)，并获取更多模型信息。

模型名称 | 论文和模型 | 模型大小 | Top-1 准确率 | Top-5 准确率 | CPU（4 线程） | GPU | NNAPI
--- | :-: | --: | --: | --: | --: | --: | --:
DenseNet | [论文](https://arxiv.org/abs/1608.06993)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/densenet_2018_04_27.tgz) | 43.6 Mb | 64.2% | 85.6% | 195 毫秒 | 60 毫秒 | 1656 毫秒
SqueezeNet | [论文](https://arxiv.org/abs/1602.07360)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz) | 5.0 Mb | 49.0% | 72.9% | 36 毫秒 | 9.5 毫秒 | 18.5 毫秒
NASNet mobile | [论文](https://arxiv.org/abs/1707.07012)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz) | 21.4 Mb | 73.9% | 91.5% | 56 毫秒 | --- | 102 毫秒
NASNet large | [论文](https://arxiv.org/abs/1707.07012)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_large_2018_04_27.tgz) | 355.3 Mb | 82.6% | 96.1% | 1170 毫秒 | --- | 648 毫秒
ResNet_V2_101 | [论文](https://arxiv.org/abs/1603.05027)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/resnet_v2_101.tgz) | 178.3 Mb | 76.8% | 93.6% | 526 毫秒 | 92 毫秒 | 1572 毫秒
Inception_V3 | [论文](http://arxiv.org/abs/1512.00567)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz) | 95.3 Mb | 77.9% | 93.8% | 249 毫秒 | 56 毫秒 | 148 毫秒
Inception_V4 | [论文](http://arxiv.org/abs/1602.07261)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz) | 170.7 Mb | 80.1% | 95.1% | 486 毫秒 | 93 毫秒 | 291 毫秒
Inception_ResNet_V2 | [论文](https://arxiv.org/abs/1602.07261)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz) | 121.0 Mb | 77.5% | 94.0% | 422 毫秒 | 100 毫秒 | 201 毫秒
Mobilenet_V1_0.25_128 | [论文](https://arxiv.org/pdf/1704.04861.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz) | 1.9 Mb | 41.4% | 66.2% | 1.2 毫秒 | 1.6 毫秒 | 3 毫秒
Mobilenet_V1_0.25_160 | [论文](https://arxiv.org/pdf/1704.04861.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_160.tgz) | 1.9 Mb | 45.4% | 70.2% | 1.7 毫秒 | 1.7 毫秒 | 3.2 毫秒
Mobilenet_V1_0.25_192 | [论文](https://arxiv.org/pdf/1704.04861.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_192.tgz) | 1.9 Mb | 47.1% | 72.0% | 2.4 毫秒 | 1.8 毫秒 | 3.0 毫秒
Mobilenet_V1_0.25_224 | [论文](https://arxiv.org/pdf/1704.04861.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_224.tgz) | 1.9 Mb | 49.7% | 74.1% | 3.3 毫秒 | 1.8 毫秒 | 3.6 毫秒
Mobilenet_V1_0.50_128 | [论文](https://arxiv.org/pdf/1704.04861.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_128.tgz) | 5.3 Mb | 56.2% | 79.3% | 3.0 毫秒 | 1.7 毫秒 | 3.2 毫秒
Mobilenet_V1_0.50_160 | [论文](https://arxiv.org/pdf/1704.04861.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_160.tgz) | 5.3 Mb | 59.0% | 81.8% | 4.4 毫秒 | 2.0 毫秒 | 4.0 毫秒
Mobilenet_V1_0.50_192 | [论文](https://arxiv.org/pdf/1704.04861.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_192.tgz) | 5.3 Mb | 61.7% | 83.5% | 6.0 毫秒 | 2.5 毫秒 | 4.8 毫秒
Mobilenet_V1_0.50_224 | [论文](https://arxiv.org/pdf/1704.04861.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_224.tgz) | 5.3 Mb | 63.2% | 84.9% | 7.9 毫秒 | 2.8 毫秒 | 6.1 毫秒
Mobilenet_V1_0.75_128 | [论文](https://arxiv.org/pdf/1704.04861.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_128.tgz) | 10.3 Mb | 62.0% | 83.8% | 5.5 毫秒 | 2.6 毫秒 | 5.1 毫秒
Mobilenet_V1_0.75_160 | [论文](https://arxiv.org/pdf/1704.04861.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_160.tgz) | 10.3 Mb | 65.2% | 85.9% | 8.2 毫秒 | 3.1 毫秒 | 6.3 毫秒
Mobilenet_V1_0.75_192 | [论文](https://arxiv.org/pdf/1704.04861.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_192.tgz) | 10.3 Mb | 67.1% | 87.2% | 11.0 毫秒 | 4.5 毫秒 | 7.2 毫秒
Mobilenet_V1_0.75_224 | [论文](https://arxiv.org/pdf/1704.04861.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_224.tgz) | 10.3 Mb | 68.3% | 88.1% | 14.6 毫秒 | 4.9 毫秒 | 9.9 毫秒
Mobilenet_V1_1.0_128 | [论文](https://arxiv.org/pdf/1704.04861.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_128.tgz) | 16.9 Mb | 65.2% | 85.7% | 9.0 毫秒 | 4.4 毫秒 | 6.3 毫秒
Mobilenet_V1_1.0_160 | [论文](https://arxiv.org/pdf/1704.04861.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_160.tgz) | 16.9 Mb | 68.0% | 87.7% | 13.4 毫秒 | 5.0 毫秒 | 8.4 毫秒
Mobilenet_V1_1.0_192 | [论文](https://arxiv.org/pdf/1704.04861.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_192.tgz) | 16.9 Mb | 69.9% | 89.1% | 18.1 毫秒 | 6.3 毫秒 | 10.6 毫秒
Mobilenet_V1_1.0_224 | [论文](https://arxiv.org/pdf/1704.04861.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz) | 16.9 Mb | 71.0% | 89.9% | 24.0 毫秒 | 6.5 毫秒 | 13.8 毫秒
Mobilenet_V2_1.0_224 | [论文](https://arxiv.org/pdf/1801.04381.pdf)，[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz) | 14.0 Mb | 71.8% | 90.6% | 17.5 毫秒 | 6.2 毫秒 | 11.23 毫秒

### AutoML 移动端模型

下列图像分类模型使用 <a href="https://cloud.google.com/automl/">Cloud AutoML</a> 创建。性能值在运行 Android 10 的 Pixel 3 上测定。

您可以在 [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite&q=MnasNet) 中找到这些模型，并获取更多模型信息。

模型名称 | 论文和模型 | 模型大小 | Top-1 准确率 | Top-5 准确率 | CPU（4 线程） | GPU | NNAPI
--- | :-: | --: | --: | --: | --: | --: | --:
MnasNet_0.50_224 | [论文](https://arxiv.org/abs/1807.11626)，[tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_0.5_224_09_07_2018.tgz) | 8.5 Mb | 68.03% | 87.79% | 9.5 毫秒 | 5.9 毫秒 | 16.6 毫秒
MnasNet_0.75_224 | [论文](https://arxiv.org/abs/1807.11626)，[tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_0.75_224_09_07_2018.tgz) | 12 Mb | 71.72% | 90.17% | 13.7 毫秒 | 7.1 毫秒 | 16.7 毫秒
MnasNet_1.0_96 | [论文](https://arxiv.org/abs/1807.11626)，[tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_96_09_07_2018.tgz) | 17 Mb | 62.33% | 83.98% | 5.6 毫秒 | 5.4 毫秒 | 12.1 毫秒
MnasNet_1.0_128 | [论文](https://arxiv.org/abs/1807.11626)，[tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_128_09_07_2018.tgz) | 17 Mb | 67.32% | 87.70% | 7.5 毫秒 | 5.8 毫秒 | 12.9 毫秒
MnasNet_1.0_160 | [论文](https://arxiv.org/abs/1807.11626)，[tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_160_09_07_2018.tgz) | 17 Mb | 70.63% | 89.58% | 11.1 毫秒 | 6.7 毫秒 | 14.2 毫秒
MnasNet_1.0_192 | [论文](https://arxiv.org/abs/1807.11626)，[tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_192_09_07_2018.tgz) | 17 Mb | 72.56% | 90.76% | 14.5 毫秒 | 7.7 毫秒 | 16.6 毫秒
MnasNet_1.0_224 | [论文](https://arxiv.org/abs/1807.11626)，[tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_224_09_07_2018.tgz) | 17 Mb | 74.08% | 91.75% | 19.4 毫秒 | 8.7 毫秒 | 19 毫秒
MnasNet_1.3_224 | [论文](https://arxiv.org/abs/1807.11626)，[tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.3_224_09_07_2018.tgz) | 24 Mb | 75.24% | 92.55% | 27.9 毫秒 | 10.6 毫秒 | 22.0 毫秒

注：性能数值来自在 Pixel-3 (Android 10) 上进行的基准测试。准确率数值使用 [TFLite 图像分类评估工具](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification)计算得出。

## 物体检测

有关物体检测的更多信息，请参阅<a href="../models/object_detection/overview.md">物体检测</a>。探索 TensorFlow Lite Task 库，以获取有关如何在短短几行代码中[集成物体检测模型](../inference_with_metadata/task_library/object_detector)的说明。

请从 TensorFlow Hub 中获取[物体检测模型](https://tfhub.dev/s?deployment-format=lite&module-type=image-object-detection)。

## 姿势预测

有关姿态估计的更多信息，请参阅<a href="../models/pose_estimation/overview.md">姿势预测</a>。

请从 TensorFlow Hub 中获取[姿势预测模型](https://tfhub.dev/s?deployment-format=lite&module-type=image-pose-detection)。

## 图像分割

有关图像分割的更多信息，请参阅<a href="../models/segmentation/overview.md">分割</a>。探索 TensorFlow Lite Task 库，以获取有关如何在短短几行代码中[集成图像分割模型](../inference_with_metadata/task_library/image_segmenter)的说明。

请从 TensorFlow Hub 中获取[图像分割模型](https://tfhub.dev/s?deployment-format=lite&module-type=image-segmentation)。

## 问答

有关使用 MobileBERT 进行问答的更多信息，请参阅<a href="../models/bert_qa/overview.md">问答</a>。探索 TensorFlow Lite Task 库，以获取有关如何在短短几行代码中[集成问答模型](../inference_with_metadata/task_library/bert_question_answerer)的说明。

请从 TensorFlow Hub 中获取 [Mobile BERT 模型](https://tfhub.dev/tensorflow/mobilebert/1)。

## 智能回复

有关智能回复的更多信息，请参阅<a href="../models/smart_reply/overview.md">智能回复</a>。

请从 TensorFlow Hub 中获取[智能回复模型](https://tfhub.dev/tensorflow/smartreply/1)。
