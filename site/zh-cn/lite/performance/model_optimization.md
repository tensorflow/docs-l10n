# 模型优化

Tensorflow Lite 和 [Tensorflow Model Optimization Toolkit](https://tensorflow.google.cn/model_optimization) (Tensorflow模型优化工具包)提供了最小优化推理复杂性的工具。

TensorFlow Lite 和 [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization) 提供的工具可将优化推断的复杂性降至最低。

深度神经网络的量化使用了一些技术，这些技术可以降低权重的精确表示，并且可选的降低存储和计算的激活值。量化的好处有:

## 模型量化

TensorFlow Lite 对量化提供了多种级别的对量化支持。

### 延时和准确性结果

以下是一些模型经过 post-training quantization 和 quantization-aware training 后的延迟和准确性结果。所有延迟数都是在使用单个大内核的 Pixel 2 设备上测量的。随着工具包的改进，这些数字也会随之提高:

- **较小的存储大小**：小模型在用户设备上占用的存储空间更少。例如，一个使用小模型的 Android 应用在用户的移动设备上会占用更少的存储空间。
- **较小的下载大小**：小模型下载到用户设备所需的时间和带宽较少。
- **更少的内存用量**：小模型在运行时使用的内存更少，从而释放内存供应用的其他部分使用，并可以转化为更好的性能和稳定性。

首先，检查 [hosted models](../guide/hosted_models.md) 中的模型是否适合您的应用程序。如果没有，我们建议用户从 [post-training quantization tool](post_training_quantization.md) 开始，因为它广泛适用的，且无需训练数据。

### 降低延迟

对于精度和延迟目标没有达到，或者需要硬件加速器支持情况， [quantization-aware training](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize){:.external} 是更好的选择。参见 Tensorflow 模型优化工具包[Tensorflow Model Optimization Toolkit](https://tensorflow.google.cn/model_optimization) 中的的其他优化技术。

注意: Quantization-aware training 支持卷积神经网络体系结构的子集。

### 加速器兼容性

某些硬件加速器，如 [Edge TPU](https://cloud.google.com/edge-tpu/)，可以使用已正确优化的模型以极快的速度运行推断。

通常，这些类型的设备要求以特定方式对模型进行量化。请参阅每个硬件加速器的文档以详细了解其要求。

## 工具选择

优化可能会导致模型准确率发生变化，这在应用开发过程中必须予以考虑。

准确率的变化取决于被优化的单个模型，而且很难提前预测。一般来说，针对大小或延迟进行优化的模型会损失少量准确率。根据您应用的不同，这可能会或可能不会影响您的用户体验。在极少数情况下，某些模型可能会因为优化过程而获得准确性的小幅提升。

## 优化的类型

TensorFlow Lite 目前支持通过量化、剪枝和聚类进行优化。

这些都是 [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization) 的一部分，该工具包提供了与 TensorFlow Lite 兼容的模型优化技术的资源。

### 量化

[量化](https://www.tensorflow.org/model_optimization/guide/quantization/post_training)的工作原理是降低用于表示模型参数的数字（默认情况为 32 位浮点数）的精度。这样可以获得较小的模型大小和较快的计算速度。

TensorFlow Lite 提供以下量化类型:

技术 | 数据要求 | 大小缩减 | 准确率 | 支持的硬件
--- | --- | --- | --- | ---
训练后 Float16 量化 | 无数据 | 高达 50% | 轻微的准确率损失 | CPU、GPU
训练后动态范围量化 | 无数据 | 高达 75% | 极小的准确率损失 | CPU、GPU (Android)
[训练后量化](post_training_integer_quant.ipynb) | 无标签的代表性样本 | 高达 75% | 极小的准确率损失 | CPU、GPU (Android)、Edge TPU、Hexagon DSP
量化感知训练 | 带标签的训练数据 | 高达 75% | 极小的准确率损失 | CPU、GPU (Android)、Edge TPU、Hexagon DSP

以下决策树可帮助您仅根据预期的模型大小和准确率来选择要用于模型的量化方案。

![量化决策树](images/quantization_decision_tree.png)

下面是几个模型的训练后量化和量化感知训练的延迟和准确率结果。所有延迟数字均在使用单一大核心 CPU 的 Pixel 2 设备上测得。随着工具包的改进，此处的数字也会随之提高：

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Top-1 Accuracy (Original) </th>
      <th>Top-1 Accuracy (Post Training Quantized) </th>
      <th>Top-1 Accuracy (Quantization Aware Training) </th>
      <th>Latency (Original) (ms) </th>
      <th>Latency (Post Training Quantized) (ms) </th>
      <th>Latency (Quantization Aware Training) (ms) </th>
      <th> Size (Original) (MB)</th>
      <th> Size (Optimized) (MB)</th>
    </tr> <tr><td>Mobilenet-v1-1-224</td><td>0.709</td><td>0.657</td><td>0.70</td>
      <td>124</td><td>112</td><td>64</td><td>16.9</td><td>4.3</td></tr>
    <tr><td>Mobilenet-v2-1-224</td><td>0.719</td><td>0.637</td><td>0.709</td>
      <td>89</td><td>98</td><td>54</td><td>14</td><td>3.6</td></tr>
   <tr><td>Inception_v3</td><td>0.78</td><td>0.772</td><td>0.775</td>
      <td>1130</td><td>845</td><td>543</td><td>95.7</td><td>23.9</td></tr>
   <tr><td>Resnet_v2_101</td><td>0.770</td><td>0.768</td><td>N/A</td>
      <td>3973</td><td>2868</td><td>N/A</td><td>178.3</td><td>44.9</td></tr>
 </table>
  <figcaption>
    <b>Table 1</b> Benefits of model quantization for select CNN models
  </figcaption>
</figure>

### 使用 int16 激活和 int8 权重的全整数量化

[使用 int16 激活的量化](https://www.tensorflow.org/model_optimization/guide/quantization/post_training)是一个具有 int16 激活和 int8 权重的全整数量化方案。与激活和权重均为 int8 的全整数量化方案相比，这种模式可以提高量化模型的准确率，并保持相似的模型大小。建议在激活对量化敏感时使用。

<i>注</i>：目前在 TFLite 中只有未优化的参考内核实现可用于此量化方案，因此默认情况下，与 int8 内核相比，性能较慢。目前可以通过专门的硬件或自定义软件来获得这种模式的全部优势。

以下是一些受益于此模式的模型的准确率结果。

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Accuracy metric type </th>
      <th>Accuracy (float32 activations) </th>
      <th>Accuracy (int8 activations) </th>
      <th>Accuracy (int16 activations) </th>
    </tr> <tr><td>Wav2letter</td><td>WER</td><td>6.7%</td><td>7.7%</td>
      <td>7.2%</td></tr>
    <tr><td>DeepSpeech 0.5.1 (unrolled)</td><td>CER</td><td>6.13%</td><td>43.67%</td>
      <td>6.52%</td></tr>
    <tr><td>YoloV3</td><td>mAP(IOU=0.5)</td><td>0.577</td><td>0.563</td>
      <td>0.574</td></tr>
    <tr><td>MobileNetV1</td><td>Top-1 Accuracy</td><td>0.7062</td><td>0.694</td>
      <td>0.6936</td></tr>
    <tr><td>MobileNetV2</td><td>Top-1 Accuracy</td><td>0.718</td><td>0.7126</td>
      <td>0.7137</td></tr>
    <tr><td>MobileBert</td><td>F1(Exact match)</td><td>88.81(81.23)</td><td>2.08(0)</td>
      <td>88.73(81.15)</td></tr>
 </table>
  <figcaption>
    <b>Table 2</b> Benefits of model quantization with int16 activations
  </figcaption>
</figure>

### 剪枝

[剪枝](https://www.tensorflow.org/model_optimization/guide/pruning)的工作原理是移除模型中对其预测影响很小的参数。剪枝后的模型在磁盘上的大小相同，并且具有相同的运行时延迟，但可以更高效地压缩。这使剪枝成为缩减模型下载大小的实用技术。

未来，TensorFlow Lite 将降低剪枝后模型的延迟。

### 聚类

[聚类](https://www.tensorflow.org/model_optimization/guide/clustering)的工作原理是将模型中每一层的权重归入预定数量的聚类中，然后共享属于每个单独聚类的权重的质心值。这就减少了模型中唯一权重值的数量，从而降低了其复杂性。

这样一来，就可以更高效地压缩聚类后的模型，从而提供类似于剪枝的部署优势。

## 开发工作流

首先，检查[托管模型](../guide/hosted_models.md)中的模型能否用于您的应用。如果不能，我们建议用户从[训练后量化工具](post_training_quantization.md)开始，因为它适用范围广，且无需训练数据。

对于无法达到准确率和延迟目标，或硬件加速器支持很重要的情况，[量化感知训练](https://www.tensorflow.org/model_optimization/guide/quantization/training){:.external}是更好的选择。请参阅 [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization) 下的其他优化技术。

如果要进一步缩减模型大小，可以在量化模型之前尝试[剪枝](#pruning)和/或[聚类](#clustering)。
