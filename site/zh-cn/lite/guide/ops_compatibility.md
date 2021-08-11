# TensorFlow Lite 和 TensorFlow 算子兼容性

TensorFlow Lite supports a number of TensorFlow operations used in common inference models. As they are processed by the TensorFlow Lite Optimizing Converter, those operations may be elided or fused, before the supported operations are mapped to their TensorFlow Lite counterparts.

由于 TensorFlow Lite 内置算子库仅支持有限数量的 TensorFlow 算子，所以并非所有模型都可以转换。即使对于支持的运算，出于性能原因，有时也需要非常特定的使用模式。我们希望在未来的 TensorFlow Lite 版本中扩展支持的运算。

The best way to understand how to build a TensorFlow model that can be used with TensorFlow Lite is to carefully consider how operations are converted and optimized, along with the limitations imposed by this process.

## 支持的类型

大多数 TensorFlow Lite 运算都针对的是浮点（`float32`）和量化（`uint8`、`int8`）推断，但许多算子尚不适用于其他类型（如 `tf.float16` 和字符串）。

除了使用不同版本的运算外，浮点模型和量化模型之间的另一个区别是它们的转换方式。量化转换需要张量的动态范围信息。这需要在模型训练期间进行“伪量化”，通过校准数据集获取范围信息，或进行“实时”范围估计。请参阅[量化](../performance/model_optimization.md)。

## 支持的运算和限制

TensorFlow Lite supports a subset of TensorFlow operations with some limitations. For full list of operations and limitations see [TF Lite Ops page](https://www.tensorflow.org/mlir/tfl_ops).

## 直接转换、常量折叠以及融合

TensorFlow Lite 可以处理许多 TensorFlow 运算，即使它们没有直接等效项。这种情况适用于：可直接从计算图中移除的运算（`tf.identity`）、可用张量替换的运算（`tf.placeholder`）和可融合为更复杂运算的运算（`tf.nn.bias_add`）。有时甚至可以通过这些过程移除某些支持的运算。

下面是通常会从计算图中移除的 TensorFlow 运算的非详尽列表：

- `tf.add`
- `tf.check_numerics`
- `tf.constant`
- `tf.div`
- `tf.divide`
- `tf.fake_quant_with_min_max_args`
- `tf.fake_quant_with_min_max_vars`
- `tf.identity`
- `tf.maximum`
- `tf.minimum`
- `tf.multiply`
- `tf.no_op`
- `tf.placeholder`
- `tf.placeholder_with_default`
- `tf.realdiv`
- `tf.reduce_max`
- `tf.reduce_min`
- `tf.reduce_sum`
- `tf.rsqrt`
- `tf.shape`
- `tf.sqrt`
- `tf.square`
- `tf.subtract`
- `tf.tile`
- `tf.nn.batch_norm_with_global_normalization`
- `tf.nn.bias_add`
- `tf.nn.fused_batch_norm`
- `tf.nn.relu`
- `tf.nn.relu6`

注：上述许多运算没有 TensorFlow Lite 等效项，而且如果它们无法被消除或融合，相应的模型将不可转换。

## 实验性运算

以下 TensorFlow Lite 运算虽然存在，但尚未准备好用于自定义模型：

- `CALL`
- `CONCAT_EMBEDDINGS`
- `CUSTOM`
- `EMBEDDING_LOOKUP_SPARSE`
- `HASHTABLE_LOOKUP`
- `LSH_PROJECTION`
- `SKIP_GRAM`
- `SVDF`
