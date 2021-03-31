<!--* freshness: { owner: 'maringeo' reviewed: '2021-03-15' review_interval: '3 months'} *-->

# 模型格式

[tfhub.dev](https://tfhub.dev) hosts the following model formats: SavedModel, TF1 Hub format, TF.js and TFLite. This page provides an overview of each model format.

## TensorFlow 格式

[tfhub.dev](https://tfhub.dev) hosts TensorFlow models in the SavedModel format and TF1 Hub format. We recommend using models in the standardized SavedModel format instead of the deprecated TF1 Hub format when possible.

### SavedModel

SavedModel 是共享 TensorFlow 模型时的推荐格式。您可以参阅 [TensorFlow SavedModel](https://www.tensorflow.org/guide/saved_model) 指南，详细了解 SavedModel 格式。

You can browse SavedModels on tfhub.dev by using the TF2 version filter on the [tfhub.dev browse page](https://tfhub.dev/s?subtype=module,placeholder) or by following [this link](https://tfhub.dev/s?subtype=module,placeholder&tf-version=tf2).

您在使用 tfhub.dev 中的 SavedModel 时无需依赖于 `tensorflow_hub` 库，因为此格式为核心 TensorFlow 的一部分。

详细了解 TF Hub 上的 SavedModel：

- [使用 TF2 SavedModel](tf2_saved_model.md)
- [导出 TF2 SavedModel](exporting_tf2_saved_model.md)
- [TF2 SavedModel 的 TF1/TF2 兼容性](model_compatibility.md)

### TF1 Hub 格式

TF1 Hub 格式是 TF Hub 库所使用的自定义序列化格式。在语法层面上，TF1 Hub 格式与 TensorFlow 2 的 SavedModel 格式类似（文件名和协议消息相同），但在针对模块重用、构成和重新训练的语义上有所不同（例如，资源初始值设定项的存储方式不同，元图的标记惯例不同）。最简单的区分方式是查看磁盘上是否存在 `tfhub_module.pb` 文件。

You can browse models in the TF1 Hub format on tfhub.dev by using the TF1 version filter on the [tfhub.dev browse page](https://tfhub.dev/s?subtype=module,placeholder) or by following [this link](https://tfhub.dev/s?subtype=module,placeholder&tf-version=tf1).

详细了解 TF Hub 上 TF1 Hub 格式的模型：

- [使用 TF1 Hub 格式的模型](tf1_hub_module.md)
- [以 TF1 Hub 格式导出模型](exporting_hub_format.md)
- [TF1 Hub 格式的 TF1/TF2 兼容性](model_compatibility.md)

## TFLite 格式

TFLite 格式用于设备端推断。您可以参阅 [TFLite 文档](https://www.tensorflow.org/lite)了解更多信息。

You can browse TF Lite models on tfhub.dev by using the TF Lite model format filter on the [tfhub.dev browse page](https://tfhub.dev/s?subtype=module,placeholder) or by following [this link](https://tfhub.dev/lite).

## TFJS 格式

TF.js format 格式用于浏览器内机器学习。您可以参阅 [TF.js 文档](https://www.tensorflow.org/js)了解更多信息。

You can browse TF.js models on tfhub.dev by using the TF.js model format filter on the [tfhub.dev browse page](https://tfhub.dev/s?subtype=module,placeholder) or by following [this link](https://tfhub.dev/js).
