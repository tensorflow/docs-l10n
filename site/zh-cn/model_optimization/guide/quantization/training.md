# 量化感知训练

<sub>由 TensorFlow Model Optimization 维护</sub>

有两种形式的量化：训练后量化和量化感知训练。请从[训练后量化](post_training.md)开始，因为它更易于使用，尽管量化感知训练在模型准确率方面的表现通常更好。

本页面概述了量化感知训练，旨在帮助您确定它与您的用例的契合程度。

- 要查看端到端示例，请参阅[量化感知训练示例](training_example.md)。
- 要快速找到您的用例所需的 API，请参阅[量化感知训练综合指南](training_comprehensive_guide.md)。

## 概述

量化感知训练可以模拟推断时间量化，同时创建一个模型，下游工具将使用该模型生成实际量化模型。量化模型使用较低的精度（例如 8 位而不是 32 位浮点数），这样可在部署期间带来诸多好处。

### 使用量化部署

量化通过压缩模型和减少延迟带来了诸多改进。使用 API​​ 默认值时，模型大小可缩减至原来的四分之一，我们通常会在测试的后端中看到 CPU 延迟缩短为原来的三分之二到四分之一。最终，可以在兼容的机器学习加速器（例如 [EdgeTPU](https://coral.ai/docs/edgetpu/benchmarks/) 和 NNAPI）上看到延迟方面的改善。

这种技术用于语音、视觉、文本和翻译用例的生产中。代码目前支持[其中一部分模型](#general-support-matrix)。

### 试验量化和相关硬件

用户可以配置量化参数（例如位数），并在一定程度上配置底层算法。请注意，对 API 默认值进行这些更改后，目前没有支持的路径可以部署到后端。例如，TFLite 转换和内核实现仅支持 8 位量化。

特定于此配置的 API 是实验性的，不具备向后兼容性。

### API 兼容性

用户可以使用以下 API 应用量化：

- 模型构建：仅包含序贯模型和函数式模型的 `tf.keras`。
- TensorFlow 版本：TF 2.x Nightly 版本。
    - 不支持包含 TF 2.X 软件包的 `tf.compat.v1`。
- TensorFlow 执行模式：Eager Execution

根据我们的路线图，将在以下方面增加支持：

<!-- TODO(tfmot): file Github issues. -->

- 模型构建：阐明子类化模型是如何被限制为不支持的
- 分布式训练：`tf.distribute`

### 一般支持矩阵

在以下方面提供支持：

- 模型覆盖：使用[allowlisted layers](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/quantization/keras/default_8bit/default_8bit_quantize_registry.py)的模型，遵循 Conv2D 和 DepthwiseConv2D 层时的 BatchNormalization，以及少数情况下的 `Concat`。
    <!-- TODO(tfmot): add more details and ensure they are all correct. -->
- 硬件加速：我们的 API 默认值兼容 EdgeTPU、NNAPI 和 TFLite 后端等设备上的加速。请参阅路线图中的注意事项。
- 使用量化部署：目前仅支持卷积层的按轴量化，不支持按张量量化。

根据我们的路线图，将在以下方面增加支持：

<!-- TODO(tfmot): file Github issue. Update as more functionality is added prior
to launch. -->

- 模型覆盖：扩展到包括 RNN/LSTM 和一般的 Concat 支持。
- 硬件加速：确保 TFLite 转换器可以产生全整数模型。有关详细信息，请参阅[此问题](https://github.com/tensorflow/tensorflow/issues/38285)。
- 试验量化用例：
    - 试验跨越 Keras 层或需要训练步骤的量化算法。
    - 使 API 稳定。

## 结果

### 使用工具进行图像分类

<figure>
  <table>
    <tr>
      <th>模型</th>
      <th>非量化的 Top-1 准确率</th>
      <th>8 位量化准确率</th>
    </tr>
    <tr>
      <td>MobilenetV1 224</td>
      <td>71.03%</td>
      <td>71.06%</td>
    </tr>
    <tr>
      <td>Resnet v1 50</td>
      <td>76.3%</td>
      <td>76.1%</td>
    </tr>
    <tr>
      <td>MobilenetV2 224</td>
      <td>70.77%</td>
      <td>70.01%</td>
    </tr>
 </table>
</figure>

模型基于 Imagenet 进行测试，并在 TensorFlow 和 TFLite 中进行评估。

### 技术的图像分类

<figure>
  <table>
    <tr>
      <th>模型</th>
      <th>非量化的 Top-1 准确率</th>
      <th>8 位量化准确率</th>
    </tr>
<tr>
      <td>Nasnet-Mobile</td>
      <td>74%</td>
      <td>73%</td>
    </tr>
    <tr>
      <td>Resnet-v2 50</td>
      <td>75.6%</td>
      <td>75%</td>
    </tr>
 </table>
</figure>

模型基于 Imagenet 进行测试，并在 TensorFlow 和 TFLite 中进行评估。

## 示例

除了[量化感知训练示例](training_example.md)外，另请参阅以下示例：

- 基于使用量化的 MNIST 手写数字分类任务的 CNN 模型：[code](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/quantization/keras/quantize_functional_test.py)

有关类似内容的背景信息，请参阅[论文](https://arxiv.org/abs/1712.05877) *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*。这篇论文介绍了此工具使用的一些概念。实现并不完全相同，而且此工具中还使用了其他概念（例如按轴量化）。
