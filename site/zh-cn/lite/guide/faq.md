# 常见问题解答

如果您在此处未找到您所关注问题的解答，请查看相关主题的详细文档，或提交 [GitHub 问题](https://github.com/tensorflow/tensorflow/issues)。

## 模型转换

#### 从 TensorFlow 向 TensorFlow Lite 转换支持哪些格式？

TensorFlow Lite 转换器支持以下格式：

- SavedModel：[TFLiteConverter.from_saved_model](../convert/python_api.md#exporting_a_savedmodel_)
- 由 [freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py) 生成的冻结 GraphDef：[TFLiteConverter.from_frozen_graph](../convert/python_api.md#exporting_a_graphdef_from_file_)
- tf.keras HDF5 模型：[TFLiteConverter.from_keras_model_file](../convert/python_api.md#exporting_a_tfkeras_file_)
- tf.Session：[TFLiteConverter.from_session](../convert/python_api.md#exporting_a_graphdef_from_tfsession_)

建议将 [Python 转换器](../convert/python_api.md)集成在您的模型流水线中，以便尽早检测兼容性问题。

#### 为什么我的模型无法转换？

由于 TensorFlow Lite 的运算数量低于 TensorFlow，某些推理模型可能无法转换。对于未实现的运算，请查看有关[缺少算子](faq.md#why-are-some-operations-not-implemented-in-tensorflow-lite)的问题。不支持的算子包括嵌入向量和 LSTM/RNN。对于 LSTM/RNN 模型，您还可以尝试使用实验性 API [OpHint](https://www.tensorflow.org/api_docs/python/tf/lite/OpHint) 进行转换。目前尚不支持对具有控制流运算（开关、合并等）的模型进行转换，但我们正努力在 Tensorflow Lite 中添加对控制流的支持，请参见 [GitHub 问题](https://github.com/tensorflow/tensorflow/issues/28485)。

对于与缺少运算或控制流运算无关的转换问题，请搜索我们的 [GitHub 问题](https://github.com/tensorflow/tensorflow/issues?q=label%3Acomp%3Alite+)或提交[新问题](https://github.com/tensorflow/tensorflow/issues)。

#### 如何确定 GraphDef 协议缓冲区的输入/输出？

从 `.pb` 文件检查计算图的最简单方式是使用 [Netron](https://github.com/lutzroeder/netron)，这是一种面向机器学习模型的开源查看器。

如果 Netron 无法打开计算图，您可以尝试 [summarize_graph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md#inspecting-graphs) 工具。

如果 summarize_graph 工具产生错误，您可以使用 [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) 可视化 GraphDef 并在计算图中查找输入和输出。要可视化 `.pb` 文件，请使用 [`import_pb_to_tensorboard.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/import_pb_to_tensorboard.py) 脚本，如下所示：

```shell
python import_pb_to_tensorboard.py --model_dir <model path> --log_dir <log dir path>
```

#### 如何检查 `.tflite` 文件？

[Netron](https://github.com/lutzroeder/netron) 是可视化 TensorFlow Lite 模型的最简单方式。

如果 Netron 无法打开您的 TensorFlow Lite 模型，您可以尝试我们的仓库中提供的 [visualize.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py) 脚本。

如果您使用的是 TF 2.5 或更高版本

```shell
python -m tensorflow.lite.tools.visualize model.tflite visualized_model.html
```

否则，您可以用 Bazel 运行此脚本

- [克隆 TensorFlow 仓库](https://www.tensorflow.org/install/source)
- 使用 Bazel 运行 `visualize.py` 脚本：

```shell
bazel run //tensorflow/lite/tools:visualize model.tflite visualized_model.html
```

## 模型和运算

#### 为何有些运算未在 TensorFlow Lite 中实现？

为使 TensorFlow Lite 更加轻量化，转换器中仅使用了一些特定运算。[兼容性指南](ops_compatibility.md)中提供了 TensorFlow Lite 目前支持的运算的列表。

如果您未发现所列的特定运算（或等效运算），则可能是该运算未被优先处理。团队会跟踪 GitHub [问题 #21526](https://github.com/tensorflow/tensorflow/issues/21526) 中的新运算请求。如果您的请求尚未被提及，请发表评论。

同时，您可以尝试实现[自定义算子](ops_custom.md)或使用仅包含支持算子的其他模型。如果二进制文件大小不受限制，请尝试将 TensorFlow Lite 与[精选 TensorFlow 算子](ops_select.md)结合使用。

#### 如何测试 TensorFlow Lite 模型行为与原始 TensorFlow 模型相同？

测试 TensorFlow Lite 模型行为的最佳方式是使用我们的 API 和测试数据，比较相同输入的 TensorFlow 输出。请查看我们的 [Python 解释器示例](../convert/python_api.md)，它可以生成要馈入解释器的随机数据。

## 优化

#### 如何缩减转换后的 TensorFlow Lite 模型的大小？

可以在转换至 TensorFlow Lite 的过程中使用[训练后量化](../performance/post_training_quantization.md)缩减模型大小。训练后量化可将权重从浮点量化至 8 位精度，并在运行时对其进行去量化以执行浮点计算。但请注意，这可能会影响准确率。

如果需要重新训练模型，请考虑采用[量化感知训练](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize)。但请注意，量化感知训练仅适用于卷积神经网络架构的子集。

如需深入了解不同的优化方法，请参见[模型优化](../performance/model_optimization.md)。

#### 如何针对我的机器学习任务优化 TensorFlow Lite 性能？

优化 TensorFlow Lite 性能的高级过程如下所示：

- *确保您为任务选择合适的模型*。对于图像分类，请查看我们的[托管模型列表](hosted_models.md)。
- *调整线程数*。许多 TensorFlow Lite 算子都支持多线程内核。您可以在 [C++ API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h#L345) 中使用 `SetNumThreads()` 执行调整。但是，增加线程会导致性能发生变化，具体取决于环境。
- *使用硬件加速器*。TensorFlow Lite 支持使用委托针对特定硬件进行模型加速。例如，要使用 Android 的 Neural Networks API，请在解释器上调用 [`UseNNAPI`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h#L343)。或者，请参阅我们的 [GPU 委托教程](../performance/gpu.md)。
- *（高级）分析模型*。Tensorflow Lite [基准测试工具](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)具有内置分析器，可显示每个算子的统计信息。如果您了解如何针对您的特定平台优化算子性能，那么您可以实现[自定义算子](ops_custom.md)。

有关如何优化性能的更为深入的讨论，请参见[最佳做法](../performance/best_practices.md)。
