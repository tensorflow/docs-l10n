# 模型转换

TensorFlow.js 附带各种预训练模型，这些模型可以在浏览器中使用，您可以在我们的[模型仓库](https://github.com/tensorflow/tfjs-models)中找到它们。但是，您可能已经在其他地方找到或创建了一个 TensorFlow 模型，并希望在网络应用中使用该模型。TensorFlow.js 为此目的提供了一个[模型转换器](https://github.com/tensorflow/tfjs-converter)。TensorFlow.js 转换器有两个组件：

1. 一个命令行实用工具，用于转换 Keras 和 TensorFlow 模型以在 TensorFlow.js 中使用。
2. 一个 API ，用于在浏览器中使用 TensorFlow.js 加载和执行模型。

## 转换您的模型

TensorFlow.js 转换器可以转换以下几种格式的模型：

**SavedModel**：保存 TensorFlow 模型的默认格式。有关 SavedModel 格式的详细信息，请参阅[此处](https://tensorflow.google.cn/guide/saved_model)。

**Keras 模型**：Keras 模型通常保存为 HDF5 文件。有关保存 Keras 模型的更多信息，请访问[此处](https://keras.io/getting-started/faq/#savingloading-whole-models-architecture-weights-optimizer-state)。

**TensorFlow Hub 模块**：这些是打包后用于在 TensorFlow Hub 上分发的模型，TensorFlow Hub 是一个共享和发现模型的平台。模型库位于[此处](tfhub.dev)。

根据您尝试转换的模型的类型，您需要将不同的参数传递给转换器。例如，假设您将一个名为 `model.h5` 的 Keras 模型保存到 `tmp/` 目录中。要使用 TensorFlow.js 转换器转换模型，您可以运行以下命令：

```
$ tensorflowjs_converter --input_format=keras /tmp/model.h5 /tmp/tfjs_model
```

这会转换 `/tmp/model.h5` 下的模型并将 `model.json` 文件及二进制权重文件输出到 `tmp/tfjs_model/` 目录中。

有关不同模型格式对应的命令行参数的更多详细信息，请参阅 TensorFlow.js 转换器[自述文件](https://github.com/tensorflow/tfjs-converter)。

在转换过程中，我们会遍历模型计算图并检查 TensorFlow.js 是否支持每个运算。如果支持，我们会将计算图转换成浏览器可以使用的格式。我们尝试通过将权重分成 4MB 的文件（这样它们可以被浏览器缓存）来优化模型以便在网络上应用。我们也尝试使用开放源代码 [Grappler](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/grappler) 项目简化模型计算图。计算图简化包括折叠相邻运算，从而消除常见子计算图等。这些变更对模型的输出没有影响。要进行进一步优化，用户可以传入参数以指示转换器将模型量化到特定的字节大小。量化是一种缩减模型大小的技术，它使用更少的位来表示权重。用户必须谨慎操作，以确保量化后模型的准确率保持在可接受范围内。

如果在转换过程中遇到不支持的运算，该过程将失败，我们将为用户打印该运算的名称。请在我们的 [GitHub](https://github.com/tensorflow/tfjs/issues) 下提交议题告诉我们相关信息，我们会尝试根据用户需求实现新运算。

### 最佳做法

虽然我们会在转换过程中尽力优化您的模型，但通常确保您的模型高效运行的最佳方式是在构建时考虑资源受限的环境。这意味着避免过于复杂的架构和尽可能减少参数（权重）的数量。

## 运行您的模型

成功转换模型之后，您将得到一组权重文件和一个模型拓扑文件。TensorFlow.js 提供了模型加载 API，您可以使用这些 API 提取模型资源并在浏览器中运行推断。

以下是适用于转换后的 TensorFlow SavedModel 或 TensorFlow Hub 模块的 API：

```js
const model = await tf.loadGraphModel(‘path/to/model.json’);
```

 以下是适用于转换后的 Keras 模型的 API：

```js
const model = await tf.loadLayersModel(‘path/to/model.json’);
```

`tf.loadGraphModel` API 返回 `tf.FrozenModel`，这意味着参数已被固定并且您无法使用新数据微调模型。`tf.loadLayersModel` API 返回可训练的 tf.Model。有关如何训练 tf.Model 的信息，请参阅[训练模型指南](train_models.md)。

转换后，建议您运行几次推断并对模型的速度进行基准测试。为此，我们提供了一个独立的基准测试页面：https://tensorflow.github.io/tfjs/e2e/benchmarks/local-benchmark/index.html。您可能注意到我们丢弃了初始预热运行中的测量值，这是因为（通常情况下），由于创建纹理和编译着色器的开销，您的模型的首次推断将比后续推断慢几倍。
