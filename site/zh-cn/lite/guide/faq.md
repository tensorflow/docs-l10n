# 常见问题解答

如果您在此处未找到您所关注问题的解答，请查看相关主题的详细文档，或提交 [GitHub 议题](https://github.com/tensorflow/tensorflow/issues)。

## 模型转换

#### 从 TensorFlow 到 TensorFlow Lite 的转换支持哪些格式？

[这里](../models/convert/index#python_api)列出的受支持的格式

#### 为何有些运算未在 TensorFlow Lite 中实现？

为了使 TFLite 保持轻量，TFLite 中只支持某些 TF 算子（列在[允许列表](op_select_allowlist)中）。

#### 为什么我的模型无法转换？

由于 TensorFlow Lite 运算的数量少于 TensorFlow，因此某些模型可能无法转换。[此处](../models/convert/index#conversion-errors)列出了一些常见错误。

对于与缺少运算或控制流运算无关的转换问题，请搜索我们的 [GitHub 议题](https://github.com/tensorflow/tensorflow/issues?q=label%3Acomp%3Alite+)或提交[新议题](https://github.com/tensorflow/tensorflow/issues)。

#### 如何测试 TensorFlow Lite 模型与原始 TensorFlow 模型的行为相同？

测试的最佳方式是比较具有相同输入（测试数据或随机输入）的 TensorFlow 和 TensorFlow Lite 模型的输出，如[此处](inference#load-and-run-a-model-in-python)所示。

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

如果您使用的是 TF 2.5 或更高版本，请运行以下代码：

```shell
python -m tensorflow.lite.tools.visualize model.tflite visualized_model.html
```

否则，您可以用 Bazel 运行此脚本

- [克隆 TensorFlow 库](https://www.tensorflow.org/install/source)
- 使用 Bazel 运行 `visualize.py` 脚本：

```shell
bazel run //tensorflow/lite/tools:visualize model.tflite visualized_model.html
```

## 优化

#### 如何缩减转换后的 TensorFlow Lite 模型的大小？

转换至 TensorFlow Lite 的过程中可以使用[训练后量化](../performance/post_training_quantization)以缩小模型尺寸。训练后量化可将权重从浮点量化至 8 位精度，并在运行时对其进行去量化以执行浮点计算。但请注意，这可能会影响准确性。

如果需要重新训练模型，请考虑采用[量化感知训练](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize)。但请注意，量化感知训练仅适用于卷积神经网络架构的子集。

如需深入了解不同的优化方法，请参见[模型优化](../performance/model_optimization)。

#### 如何针对我的机器学习任务优化 TensorFlow Lite 性能？

优化 TensorFlow Lite 性能的高级过程如下所示：

- *确保您为任务选用了合适的模型*。对于图像分类，请查看 [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite&module-type=image-classification)。
- *调整线程数*。许多 TensorFlow Lite 算子都支持多线程内核。您可以在 [C++ API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h#L345) 中使用 `SetNumThreads()` 执行调整。但是，增加线程数会导致性能发生变化，具体取决于环境。
- *使用硬件加速器*。TensorFlow Lite 支持使用委托针对特定硬件进行模型加速。请参阅我们的[委托](../performance/delegates)指南，了解支持哪些加速器以及如何在设备端模型中使用它们。
- *（高级）对模型进行性能分析*。TensorFlow Lite [基准测试工具](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)具有内置性能分析器，可显示每个算子的统计信息。如果您了解如何针对您的特定平台优化算子性能，那么您可以实现[自定义算子](ops_custom)。

有关如何优化性能的更为深入的讨论，请参见[最佳做法](../performance/best_practices)。
