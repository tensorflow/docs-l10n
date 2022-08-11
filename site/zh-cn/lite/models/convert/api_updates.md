# API 更新 <a name="api_updates"></a>

本页面提供了在 TensorFlow 2.x 中对 `tf.lite.TFLiteConverter` [Python API](index.md) 进行的更新的信息。

注：如果您对任何更改有疑问，请提交 [GitHub 议题](https://github.com/tensorflow/tensorflow/issues/new?template=60-tflite-converter-issue.md)。

- TensorFlow 2.3

    - Support integer (previously, only float) input/output type for integer quantized models using the new `inference_input_type` and `inference_output_type` attributes. Refer to this [example usage](../../performance/post_training_quantization.md#integer_only).
    - 支持使用动态维度转换和调整模型大小。
    - 添加了具有 16 位激活和 8 位权重的新实验性量化模式。

- TensorFlow 2.2

    - 默认情况下，利用[基于 MLIR 的转换](https://mlir.llvm.org/)（Google 最前沿的机器学习编译技术）。它可以转换新模型类，包括 Mask R-CNN、MobileBERT 等，同时也支持使用函数式控制流的模型。

- TensorFlow 2.0 与 TensorFlow 1.x

    - 将 `target_ops` 特性重命名为 `target_spec.supported_ops`
    - 移除了以下特性：
        - *quantization*: `inference_type`, `quantized_input_stats`, `post_training_quantize`, `default_ranges_stats`, `reorder_across_fake_quant`, `change_concat_input_ranges`, `get_input_arrays()`. Instead, [quantize aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training) is supported through the `tf.keras` API and [post training quantization](../../performance/post_training_quantization.md) uses fewer attributes.
        - *可视化*：`output_format`、`dump_graphviz_dir`、`dump_graphviz_video`。现在，可视化 TensorFlow Lite 模型的推荐方式是使用 [visualize.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py)。
        - *冻结计算图*：`drop_control_dependency`，因为 TensorFlow 2.x 不支持冻结计算图。
    - 移除了其他转换器 API，如 `tf.lite.toco_convert` 和 `tf.lite.TocoConverter`
    - 移除了其他相关 API，如 `tf.lite.OpHint` 和 `tf.lite.constants`（为了减少重复，`tf.lite.constants.*` 类型已映射到 `tf.*` TensorFlow 数据类型）
