# TensorFlow Lite 委托

## 简介

**委托**能够利用设备端的加速器（如 GPU 和[数字信号处理器 (DSP)](https://en.wikipedia.org/wiki/Digital_signal_processor) 来启用 TensorFlow Lite 模型的硬件加速。

默认情况下，TensorFlow Lite 会使用针对 [ARM Neon](https://developer.arm.com/documentation/dht0002/a/Introducing-NEON/NEON-architecture-overview/NEON-instructions) 指令集进行了优化的 CPU 内核。但是，CPU 是一种多用途处理器，不一定会针对机器学习模型中常见的繁重计算（例如，卷积层和密集层中的矩阵数学）进行优化。

另一方面，大多数现代手机包含的芯片在处理这些繁重运算方面表现更好。将它们用于神经网络运算，可以在延迟和功率效率方面获得巨大好处。例如，GPU 可以在延迟方面提供高达 [5 倍的加速](https://blog.tensorflow.org/2020/08/faster-mobile-gpu-inference-with-opencl.html)，而 [Qualcomm® Hexagon DSP](https://developer.qualcomm.com/software/hexagon-dsp-sdk/dsp-processor) 在我们的实验中显示可以降低高达 75% 的功耗。

这些加速器均具有支持自定义计算的相关 API，例如用于移动 GPU 的 [OpenCL](https://www.khronos.org/opencl/) 或 [OpenGL ES](https://www.khronos.org/opengles/)，以及用于 DSP 的 [Qualcomm® Hexagon SDK](https://developer.qualcomm.com/software/hexagon-dsp-sdk)。通常情况下，您必须编写大量自定义代码才能通过这些接口运行神经网络。当考虑到每种加速器都各有利弊，并且无法执行神经网络中的所有运算时，事情就会变得更加复杂。TensorFlow Lite 的 Delegate API 通过作为 TFLite 运行时和这些较低级别 API 之间的桥梁，解决了这个问题。

![runtime with delegates](images/delegate_runtime.png)

## 选择委托

TensorFlow Lite 支持多种委托，每种委托都针对特定的平台和特定类型的模型进行了优化。通常情况下，会有多种委托适用于您的用例，这取决于两个主要标准：*平台*（Android 还是 iOS？），以及您要加速的*模型类型*（浮点还是量化？）。

### 按平台分类的委托

#### 跨平台（Android 和 iOS）

- **GPU 委托** - GPU 委托在 Android 和 iOS 上均可使用。它经过了优化，可以在有 GPU 的情况下运行基于 32 位和 16 位浮点的模型。它还支持 8 位量化模型，并可提供与其浮点版本相当的 GPU 性能。有关 GPU 委托的详细信息，请参阅[适用于 GPU 的 TensorFlow Lite](gpu_advanced.md)。有关在 Android 和 iOS 上使用 GPU 委托的分步教程，请参阅 [TensorFlow Lite GPU 委托教程](gpu.md)。

#### Android

- **适用于较新 Android 设备的 NNAPI 委托** - NNAPI 委托可用于在具有 GPU、DSP 和/或 NPU 的设备上加速模型。它可在 Android 8.1 (API 27+) 或更高版本中使用。有关 NNAPI 委托的概述、分步说明和最佳做法，请参阅 [TensorFlow Lite NNAPI 委托](nnapi.md)。
- **适用于较旧 Android 设备的 Hexagon 委托** - Hexagon 委托可用于在具有 Qualcomm Hexagon DSP 的 Android 设备上加速模型。它可以在运行较旧版本 Android（不支持 NNAPI）的设备上使用。请参阅 [TensorFlow Lite Hexagon 委托](hexagon_delegate.md)，了解详细信息。

#### iOS

- **适用于较新 iPhone 和 iPad 的 Core ML 委托** - 对于提供了 Neural Engine 的较新的 iPhone 和 iPad，您可以使用 Core ML 委托来加快 32 位或 16 位 浮点模型的推断。Neural Engine 适用于具有 A12 SoC 或更高版本的 Apple 移动设备。有关 Core ML 委托的概述和分步说明，请参阅 [TensorFlow Lite Core ML 委托](coreml_delegate.md)。

### 按模型类型分类的委托

每种加速器的设计都考虑了一定的数据位宽。如果为仅支持 8 位量化运算的委托（例如 [Hexagon 委托](hexagon_delegate.md)）提供浮点模型，它将拒绝其所有运算，并且模型将完全在 CPU 上运行。为了避免此类意外，下面提供了基于模型类型的委托支持情况一览表：

**模型类型** | **GPU** | **NNAPI** | **Hexagon** | **CoreML**
--- | --- | --- | --- | ---
浮点（32 位） | 是 | 是 | 否 | 是
[训练后 float16 量化](post_training_float16_quant.ipynb) | 是 | 否 | 否 | 是
[训练后动态范围量化](post_training_quant.ipynb) | 是 | 是 | 否 | 否
[训练后整数量化](post_training_integer_quant.ipynb) | 是 | 是 | 是 | 否
[量化感知训练](http://www.tensorflow.org/model_optimization/guide/quantization/training) | 是 | 是 | 是 | 否

### 验证性能

本部分的信息可作为一个粗略指南，用于筛选可以改进您的应用的委托。但是，需要注意的是，每种委托都有一组支持的预定义运算，且执行情况可能会因模型和设备而异；例如，[NNAPI 委托](nnapi.md)可能会选择在 Pixel 手机上使用 Google 的 Edge-TPU，而在其他设备上使用 DSP。因此，通常建议您进行一些基准测试，以衡量委托对您的需求有多大用处。这还有助于判断与将委托附加到 TensorFlow Lite 运行时相关的二进制文件大小的增加是否合理。

TensorFlow Lite 拥有丰富的性能和准确率评估工具，可以让开发者有信心在其应用中使用委托。下一部分将讨论这些工具。

## 评估工具

### 延迟和内存占用

TensorFlow Lite 的[基准测试工具](https://www.tensorflow.org/lite/performance/measurement)可以使用合适的参数来评估模型性能，包括平均推断延迟、初始化开销、内存占用等。此工具支持多个标志，以确定模型的最佳委托配置。例如，可以使用 `--use_gpu` 指定 `--gpu_backend=gl`，以衡量 OpenGL 的 GPU 执行情况。[详细文档](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar)中定义了受支持的委托参数的完整列表。

下面是一个通过 `adb` 使用 GPU 运行量化模型的示例：

```
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_v1_224_quant.tflite \
  --use_gpu=true
```

您可以在[此处](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model.apk)下载该工具的 Android 64 位 ARM 架构预构建版本（[详细信息](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/android)）。

### 准确率和正确性

委托通常会以不同于 CPU 的精度执行计算。因此，在利用委托进行硬件加速时，会有（通常较小的）精度折衷。请注意，情况并不*总是*这样；例如，由于 GPU 会使用浮点精度来运行量化模型，精度可能会略有提升（例如，ILSVRC 图像分类 Top-5 提升 &lt;1%）。

TensorFlow Lite 有两种类型的工具来衡量委托对于给定模型的行为准确性：*基于任务的*工具和*与任务无关的*工具。本节描述的所有工具都支持前一部分基准测试工具使用的[高级委托参数](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar)。请注意，下面的小节关注的是*委托评估*（委托是否与 CPU 性能相同）而非模型评估（模型本身是否适合任务）。

#### 基于任务的评估

TensorFlow Lite 具有用于评估两个基于图像的任务的正确性的工具：

- [ILSVRC 2012](http://image-net.org/challenges/LSVRC/2012/)（图像分类），具有 [Top-K 准确率](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision_at_K)

- 具有[全类平均精度 (mAP)](https://cocodataset.org/#detection-2020) 的 [COCO 物体检测（含边界框）](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision)

这些工具（Android，64 位 ARM 架构）的预构建二进制文件以及文档可在以下位置找到：

- [ImageNet 图像分类](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_imagenet_image_classification)（[详细信息](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification)）
- [COCO 物体检测](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_coco_object_detection)（[详细信息](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/coco_object_detection)）

下面的示例演示了在 Pixel 4 上利用 Google 的 Edge-TPU 使用 NNAPI 进行的[图像分类评估](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification)。

```
adb shell /data/local/tmp/run_eval \
  --model_file=/data/local/tmp/mobilenet_quant_v1_224.tflite \
  --ground_truth_images_path=/data/local/tmp/ilsvrc_images \
  --ground_truth_labels=/data/local/tmp/ilsvrc_validation_labels.txt \
  --model_output_labels=/data/local/tmp/model_output_labels.txt \
  --output_file_path=/data/local/tmp/accuracy_output.txt \
  --num_images=0 # Run on all images. \
  --use_nnapi=true \
  --nnapi_accelerator_name=google-edgetpu
```

预期的输出是一个从 1 到 10 的 Top-K 指标列表：

```
Top-1 Accuracy: 0.733333
Top-2 Accuracy: 0.826667
Top-3 Accuracy: 0.856667
Top-4 Accuracy: 0.87
Top-5 Accuracy: 0.89
Top-6 Accuracy: 0.903333
Top-7 Accuracy: 0.906667
Top-8 Accuracy: 0.913333
Top-9 Accuracy: 0.92
Top-10 Accuracy: 0.923333
```

#### 与任务无关的评估

对于没有现成设备端评估工具的任务，或者如果您在尝试使用自定义模型，TensorFlow Lite 提供了 [Inference Diff](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/inference_diff) 工具。（Android，64 位 ARM 二进制架构二进制文件见[此处](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_inference_diff)）

Inference Diff 会比较以下两种设置的 TensorFlow Lite 执行情况（在延迟和输出值偏差方面）：

- 单线程 CPU 推断
- 用户定义的推断 - 由[这些参数](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar)定义

为此，该工具会生成随机高斯数据，并将其传递给两个 TFLite 解释器：一个运行单线程 CPU 内核，另一个通过用户参数进行参数化。

它会以每个元素为基础，测量两者的延迟，以及每个解释器的输出张量之间的绝对差。

对于具有单个输出张量的模型，输出可能如下所示：

```
Num evaluation runs: 50
Reference run latency: avg=84364.2(us), std_dev=12525(us)
Test run latency: avg=7281.64(us), std_dev=2089(us)
OutputDiff[0]: avg_error=1.96277e-05, std_dev=6.95767e-06
```

这意味着，对于索引 `0` 处的输出张量，CPU 输出的元素与委托输出的元素平均相差 `1.96e-05`。

请注意，解释这些数字需要对模型和每个输出张量的含义有更深入的了解。如果它是确定某种得分或嵌入向量的简单回归，那么差异应该很小（否则为委托错误）。然而，像 SSD 模型中的“检测类”这样的输出有点难以解释。例如，使用此工具可能会显示出差异，但这并不意味着委托真的有什么问题：请考虑两个（假）类：“TV (ID: 10)”，“Monitor (ID:20)”。如果某个委托稍微偏离了黄金真理，并且显示的是 Monitor 而非 TV，那么这个张量的输出差异可能会高达 20-10 = 10。
