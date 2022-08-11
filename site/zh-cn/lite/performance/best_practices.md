# 性能最佳做法

移动设备和嵌入式设备的计算资源有限，因此保持应用的资源效率非常重要。我们整理了一份最佳做法和策略的清单，可用于改善 TensorFlow Lite 模型的性能。

## 为任务选择最佳模型

您需要根据任务在模型复杂性和大小之间进行权衡。如果您的任务需要高准确率，那么您可能需要一个大而复杂的模型。对于准确率要求较低的任务，则最好使用较小的模型，因为它们不仅占用的磁盘空间和内存更少，而且通常速度更快且更节能。例如，下图显示了一些常见图像分类模型的准确率和延迟权衡。

![模型大小和准确度的关系图](../images/performance/model_size_vs_accuracy.png "模型大小和准确度")

![准确度和延迟时间的关系图](../images/performance/accuracy_vs_latency.png "准确度和延迟时间")

[MobileNets](https://arxiv.org/abs/1704.04861) 是针对移动设备优化的模型的一个示例，该模型针对移动视觉应用进行了优化。[TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite) 列出了其他几种专门针对移动设备和嵌入式设备进行了优化的模型。

您可以使用迁移学习在您自己的数据集上重新训练这些模型。请查看使用 TensorFlow Lite [Model Maker](../models/modify/model_maker/) 的迁移学习教程。

## 对您的模型进行性能分析

在选择了适合您的任务的候选模型后，最好对模型进行性能分析和基准测试。TensorFlow Lite [基准测试工具](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)有内置的性能分析器，可展示每个算子的性能分析数据。这能帮助理解性能瓶颈，以及哪些算子占据了大部分计算时间。

您还可以使用 [TensorFlow Lite 跟踪](measurement#trace_tensorflow_lite_internals_in_android)，在您的 Android 应用中使用标准的 Android 系统跟踪来对模型进行性能分析，还可以使用基于 GUI 的性能分析工具按时间直观呈现算子的调用。

## 对计算图中的算子进行性能分析和优化

如果某个特定的算子频繁出现在模型中，并且基于性能分析，您发现该算子消耗的时间最多，则可以考虑优化该算子。这种情况应该会很少见，因为 TensorFlow Lite 为大多数算子提供了优化版本。但是，如果您知道执行算子的约束条件，则可以编写自定义算子的更快版本。请查看我们的[自定义算子指南](../guide/ops_custom)。

## 优化模型

模型优化旨在创建更小的模型，这些模型通常更快、更节能，从而可以将其部署在移动设备上。TensorFlow Lite 支持量化等多种优化技术。

有关详细信息，请查看[模型优化文档](model_optimization)。

## 调整线程数

TensorFlow Lite 支持用于许多算子的多线程内核。您可以增加线程数并加快算子的执行速度。但是，增加线程数会使模型使用更多资源和能源。

对某些应用来说，延迟或许比能效更重要。您可以通过设置解释器[线程数](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h#L346)来增加线程数。然而，多线程执行的代价是增加了性能可变性，具体取决于并发执行的其他内容。对于移动应用来说，情况尤其如此。例如，单独的测试可能会显示速度比单线程快 2 倍，但是，如果另一个应用同时执行，可能会导致其性能比单线程更差。

## 消除冗余副本

如果您的应用没有仔细设计，则在向模型输入和从模型读取输出时，可能会出现冗余副本。请确保消除冗余副本。如果您使用的是更高级别的 API（如 Java），请务必仔细检查文档中的性能注意事项。例如，如果将 `ByteBuffers` 用作[输入](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/java/org/tensorflow/lite/Interpreter.java#L175)，Java API 的速度就会快很多。

## 用平台特定工具对您的应用进行性能分析

平台特定工具（如 [Android Profiler](https://developer.android.com/studio/profile/android-profiler) 和 [Instruments](https://help.apple.com/instruments/mac/current/)）提供了丰富的可被用于调试应用的性能分析信息。有时错误可能不在模型中，而在与模型交互的部分应用代码中。请务必熟悉平台特定的性能分析工具和适用于该平台的最佳做法。

## 评估您的模型是否能从使用设备上可用的硬件加速器中受益

TensorFlow Lite 添加了使用速度更快的硬件（如 GPU、DSP 和神经加速器等）来加速模型的新方式。通常，这些加速器会通过接管解释器部分执行的[委托](delegates)子模块公开。TensorFlow Lite 可以通过以下方式使用委托：

- 使用 Android 的 [Neural Networks API](https://developer.android.com/ndk/guides/neuralnetworks/)。您可以利用这些硬件加速器后端来提高模型的速度和效率。要启用 Neural Networks API，请查看 [NNAPI 委托](https://www.tensorflow.org/lite/android/delegates/nnapi)指南。
- 在 Android 和 iOS 上均可以使用 GPU 委托，分别使用 OpenGL/OpenCL 和 Metal。要尝试使用它们，请查看 [GPU 委托教程](gpu)和[文档](gpu_advanced)。
- 在 Android 上可以使用 Hexagon 委托。如果在设备上可用，它会利用 Qualcomm Hexagon DSP。请参阅 [Hexagon 委托教程](https://www.tensorflow.org/lite/android/delegates/hexagon)了解更多信息。
- 如果您可以访问非标准硬件，则可以创建自己的委托。请参阅 [TensorFlow Lite 委托](delegates)了解更多信息。

请注意，有些加速器更适合不同类型的模型。有些委托只支持浮点模型或以特定方式优化的模型。请务必对每个委托进行[基准测试](measurement)，以查看它是否适合您的应用。例如，如果您有一个非常小的模型，将该模型委托给 NN API 或 GPU 可能不值得。相反，对于具有高运算强度的大型模型来说，加速器是很好的选择。

## 需要更多帮助？

TensorFlow 团队非常乐意帮助诊断和解决您可能面对的具体性能问题。请在 [GitHub](https://github.com/tensorflow/tensorflow/issues) 提交问题并提供问题的详细信息。
