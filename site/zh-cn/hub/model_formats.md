<!--* freshness: { owner: 'maringeo' reviewed: '2021-03-15' review_interval: '3 months'} *-->

# 模型格式

[thub.dev](https://tfhub.dev) 托管了以下模型格式：SavedModel、TF1 Hub 格式、TF.js 和 TFLite。本页概述了每种模型格式。

## TensorFlow 格式

[thub.dev](https://tfhub.dev) 托管了 SavedModel 格式和 TF1 Hub 格式的 TensorFlow 模型。我们建议尽可能使用标准化的 SavedModel 格式模型，避免使用已弃用的 TF1 Hub 格式。

### SavedModel

SavedModel 是共享 TensorFlow 模型时的推荐格式。您可以参阅 [TensorFlow SavedModel](https://www.tensorflow.org/guide/saved_model) 指南，详细了解 SavedModel 格式。

您可以通过在 [thub.dev 浏览页面](https://tfhub.dev/s?subtype=module,placeholder)中使用 TF2 版本筛选器，或通过访问[此链接](https://tfhub.dev/s?subtype=module,placeholder&tf-version=tf2)在 tfhub.dev 上浏览 SavedModel。

您在使用 tfhub.dev 中的 SavedModel 时无需依赖于 `tensorflow_hub` 库，因为此格式为核心 TensorFlow 的一部分。

详细了解 TF Hub 上的 SavedModel：

- [使用 TF2 SavedModel](tf2_saved_model.md)
- [导出 TF2 SavedModel](exporting_tf2_saved_model.md)
- [TF2 SavedModel 的 TF1/TF2 兼容性](model_compatibility.md)

### TF1 Hub 格式

TF1 Hub 格式是 TF Hub 库所使用的自定义序列化格式。在语法层面上，TF1 Hub 格式与 TensorFlow 2 的 SavedModel 格式类似（文件名和协议消息相同），但在针对模块重用、构成和重新训练的语义上有所不同（例如，资源初始值设定项的存储方式不同，元图的标记惯例不同）。最简单的区分方式是查看磁盘上是否存在 `tfhub_module.pb` 文件。

您可以通过在 [thub.dev 浏览页面](https://tfhub.dev/s?subtype=module,placeholder)中使用 TF1 版本筛选器，或通过访问[此链接](https://tfhub.dev/s?subtype=module,placeholder&tf-version=tf1)在 tfhub.dev 上浏览 TF1 Hub 格式的模型。

详细了解 TF Hub 上 TF1 Hub 格式的模型：

- [使用 TF1 Hub 格式的模型](tf1_hub_module.md)
- [以 TF1 Hub 格式导出模型](exporting_hub_format.md)
- [TF1 Hub 格式的 TF1/TF2 兼容性](model_compatibility.md)

## TFLite 格式

TFLite 格式用于设备端推断。您可以参阅 [TFLite 文档](https://www.tensorflow.org/lite)了解更多信息。

您可以通过在 [thub.dev 浏览页面](https://tfhub.dev/s?subtype=module,placeholder)中使用 TF Lite 模型格式筛选器，或通过访问[此链接](https://tfhub.dev/lite)在 tfhub.dev 上浏览 TF Lite 模型。

## TFJS 格式

TF.js format 格式用于浏览器内机器学习。您可以参阅 [TF.js 文档](https://www.tensorflow.org/js)了解更多信息。

您可以通过在 [thub.dev 浏览页面](https://tfhub.dev/s?subtype=module,placeholder)中使用 TF.js 模型格式筛选器，或通过访问[此链接](https://tfhub.dev/js)在 tfhub.dev 上浏览 TF.js 模型。
