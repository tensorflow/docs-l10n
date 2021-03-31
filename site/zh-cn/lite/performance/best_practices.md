# 性能的最佳实践

移动和嵌入式设备的计算资源有限，因此保持应用的资源效率非常重要。我们整理了一份最佳实践和策略的清单，您可以用它们来提高 TensorFlow Lite 模型的性能。

## 为任务选择最佳的模型

根据任务的不同，你会需要在模型复杂度和大小之间做取舍。如果你的任务需要高准确率，那么你可能需要一个大而复杂的模型。对于精确度不高的任务，就最好使用小一点的模型，因为小的模型不仅占用更少的磁盘和内存，也一般更快更高效。比如，下图展示了常见的图像分类模型中准确率和延迟对模型大小的影响。

![模型大小和准确度的关系图](../images/performance/model_size_vs_accuracy.png "模型大小和准确度")

![准确度和延迟时间的关系图](../images/performance/accuracy_vs_latency.png "准确度和延迟时间")

[MobileNets](https://arxiv.org/abs/1704.04861) 是针对移动设备优化的模型的一个示例，该模型针对移动视觉应用进行了优化。[托管模型](../guide/hosted_models.md)列出了其他几种专门针对移动设备和嵌入式设备进行了优化的模型。

你可以用你自己的数据通过迁移学习再训练这些模型。查看我们的迁移学习教程：[图像分类](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0) 和 [物体检测](https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193)。

## 测试你的模型

在选择了一个适合你的任务的模型之后，测试该模型和设立基准很好的行为。TensorFlow Lite [测试工具](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark) 有内置的测试器，可展示每一个运算符的测试数据。这能帮助理解性能瓶颈和哪些运算符主导了运算时间。

您还可以使用 [TensorFlow Lite 跟踪](measurement.md#trace_tensorflow_lite_internals_in_android)，在您的 Android 应用中使用标准的 Android 系统跟踪来对模型进行性能分析，还可以使用基于 GUI 的性能分析工具按时间直观呈现算子的调用。

## 测试和优化图（graph）中的运算符

如果某个特定的算子频繁出现在模型中，并且基于性能分析，您发现该算子消耗的时间最多，则可以考虑优化该算子。这种情况应该会很少见，因为 TensorFlow Lite 为大多数算子提供了优化版本。但是，如果您知道执行算子的约束条件，则可以编写自定义算子的更快版本。请查看我们的[自定义算子文档](../custom_operators.md)。

## 优化你的模型

如果你的模型使用浮点权重或者激励函数，那么模型大小或许可以通过量化减少75%，该方法有效地将浮点权重从32字节转化为8字节。量化分为：[训练后量化](post_training_quantization.md) 和 [量化训练](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/quantize/README.md){:.external}。前者不需要再训练模型，但是在极少情况下会有精度损失。当精度损失超过了可接受范围，则应该使用量化训练。

有关详细信息，请查看我们的[模型优化文档](model_optimization.md)。

## 调整线程数

TensorFlow Lite 支持用于许多算子的多线程内核。您可以增加线程数并加快算子的执行速度。但是，增加线程数会使模型使用更多资源和功率。

对有些应用来说，延迟或许比能源效率更重要。你可以通过设定 [解释器](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h#L346) 的数量来增加线程数。然而，根据同时运行的其他操作不同，多线程运行会增加性能的可变性。比如，隔离测试可能显示多线程的速度是单线程的两倍，但如果同时有另一个应用在运行的话，性能测试结果可能比单线程更差。

## 清除冗余副本

如果您的应用没有仔细设计，则在向模型输入和从模型读取输出时，可能会出现冗余副本。请确保消除冗余副本。如果您使用的是更高级别的 API（如Java），请务必仔细检查文档中的性能注意事项。例如，如果将 `ByteBuffers` 用作[输入](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/java/org/tensorflow/lite/Interpreter.java#L175)，Java API 的速度就会快很多。

## 用平台特定工具测试你的应用

平台特定工具，如 [Android profiler](https://developer.android.com/studio/profile/android-profiler) 和 [Instruments](https://help.apple.com/instruments/mac/current/)，提供了丰富的可被用于调试应用的测试信息。有时性能问题可能不出自于模型，而是与模型交互的应用代码。确保熟悉平台特定测试工具和对该平台最好的测试方法。

## 评估你的模型是否受益于使用设备上可用的硬件加速器

TensorFlow Lite 添加了使用速度更快的硬件（如 GPU、DSP 和神经加速器等）来加速模型的新方式。通常，这些加速器会通过接管解释器部分执行的[委托](delegates.md)子模块公开。TensorFlow Lite 可以通过以下方式使用委托：

- 使用 Android 的[神经网络 API](https://developer.android.com/ndk/guides/neuralnetworks/)。<br>您可以利用这些硬件加速器后端来提高模型的速度和效率。<br>要启用神经网络 API，请查看 [NNAPI 委托](nnapi.md)指南。
- 我们已经发布了一个仅限二进制的 GPU 代理，Android 和 iOS 分别使用 OpenGL 和 Metal。要试用它们，查看 [GPU 代理教程](gpu.md) 和 [文档](gpu_advanced.md)。
- 在 Android 上可以使用 Hexagon 委托。如果在设备上可用，它会利用 Qualcomm Hexagon DSP。请参阅 [Hexagon 委托教程](hexagon_delegate.md)了解更多信息。
- 如果您可以访问非标准硬件，则可以创建自己的委托。请参阅 [TensorFlow Lite 委托](delegates.md)了解更多信息。

请注意，有的加速器在某些模型效果更好。为每个代理设立基准以测试出最优的选择是很重要的。比如，如果你有一个非常小的模型，那可能没必要将模型委托给 NN API 或 GPU。相反，对于具有高算术强度的大模型来说，加速器就是一个很好的选择。

## 需要更多帮助？

TensorFlow 团队非常乐意帮助你诊断和定位具体的性能问题。请在 [GitHub](https://github.com/tensorflow/tensorflow/issues) 提出问题并描述细节。
