**更新时间：2021 年 6 月**

TensorFlow 的 Model Optimization Toolkit (MOT) 已广泛用于将 TensorFlow 模型转换/优化为尺寸更小、性能更优且准确率可接受的 TensorFlow Lite 模型，以便在移动和物联网设备上运行。我们现在正致力于将 MOT 技术和工具扩展到 TensorFlow Lite 之外，以支持 TensorFlow SavedModel。

下面提供了我们的路线图的简要概览。请注意，该路线图随时可能发生变化，以下顺序也不代表任何类型的优先级。我们强烈鼓励您对我们的路线图发表评论，并在[讨论组](https://groups.google.com/a/tensorflow.org/g/tflite)中向我们提供反馈。

## 量化

#### TensorFlow Lite

- 旨在从量化中排除某些层的选择性训练后量化。
- 用于逐层检查量化误差损失的量化调试程序。
- 将量化感知训练应用于更大的模型覆盖范围，例如 TensorFlow Model Garden。
- 训练后动态范围量化的质量和性能改进。

#### TensorFlow

- 训练后量化（bf16 * int8 动态范围）。
- 量化感知训练（bf16 * int8 仅权重，使用假量化）。
- 旨在从量化中排除某些层的选择性训练后量化。
- 用于逐层检查量化误差损失的量化调试程序。

## 稀疏性

#### TensorFlow Lite

- 为更多模型提供稀疏模型执行支持。
- 稀疏性的目标感知创作。
- 使用高性能 x86 内核扩展稀疏运算集。

#### TensorFlow

- TensorFlow 中的稀疏性支持。

## 级联压缩技术

- 量化 + 张量压缩 + 稀疏性：演示全部 3 种技术协同工作。

## 压缩

- 张量压缩 API 可帮助压缩算法开发者实现他们自己的模型压缩算法（例如权重聚类），包括提供测试/基准测试的标准方法。
