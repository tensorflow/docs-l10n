# TensorFlow Lite 和 TensorFlow 算子兼容性

您在模型中使用的机器学习 (ML) 算子可能会影响将 TensorFlow 模型转换为 TensorFlow Lite 格式的过程。TensorFlow Lite 转换器支持常见推断模型中使用的有限数量的 TensorFlow 运算，这意味着并非每个模型都可以直接转换。转换器工具允许您包含其他算子，但以这种方式转换模型还需要您修改用于执行模型的 TensorFlow Lite 运行时环境，这可能会限制您使用标准运行时部署选项（例如 [Google Play 服务](../android/play_services)）的能力。

TensorFlow Lite 转换器旨在分析模型结构并应用优化，以使其与直接支持的算子兼容。例如，根据模型中的机器学习算子，转换器可能会[省略或融合](../models/convert/operation_fusion)这些算子，以将它们映射到其 TensorFlow Lite 对应项。

即使对于受支持的运算，出于性能原因，有时也需要特定的使用模式。要理解如何构建可以和 TensorFlow Lite 一起使用的 TensorFlow 模型，最好的方式是仔细考虑运算的转换和优化方式，以及此过程带来的限制。

## 支持的算子

TensorFlow Lite 内置算子是一部分属于 TensorFlow 核心库的算子。您的 TensorFlow 模型还可能包含以复合算子或您定义的新算子形式的自定义算子。下图显示了这些算子之间的关系。

![TensorFlow 算子](../images/convert/tf_operators_relationships.png)

在这一系列机器学习模型算子中，转换过程支持 3 种类型的模型：

1. 仅具有 TensorFlow Lite 内置算子的模型。（**推荐**）
2. 具有内置算子和精选 TensorFlow 核心算子的模型。
3. 具有内置算子、TensorFlow 核心算子和/或自定义算子的模型。

如果您的模型仅包含 TensorFlow Lite 原生支持的运算，则不需要任何额外的标志即可转换它。这是推荐的路径，因为这种类型的模型将平滑转换，并且使用默认的 TensorFlow Lite 运行时优化和运行更简单。此外，您还可以为模型提供更多部署选项，例如 [Google Play 服务](../android/play_services)。您可以开始阅读 [TensorFlow Lite 转换器指南](../models/convert/convert_models)。有关内置算子的列表，请参阅 [TensorFlow Lite 运算](https://www.tensorflow.org/mlir/tfl_ops)页面。

如果您需要从核心库中包含精选 TensorFlow 运算，则必须在转换时指定并确保您的运行时包含这些运算。有关详细步骤，请参阅[精选 TensorFlow 算子](ops_select.md)主题。

尽可能避免在转换后的模型中包含自定义算子的最后一个选项。[自定义算子](https://www.tensorflow.org/guide/create_op)要么是通过组合多个基元 TensorFlow 核心算子创建的算子，要么是定义一个全新算子。转换自定义算子时，它们会通过在内置 TensorFlow Lite 库之外产生依赖关系来增加整个模型的大小。与服务器环境相比，如果不是专门为移动或设备部署创建的自定义运算，当部署到资源受限的设备时，可能会导致性能更差。最后，就像包含精选 TensorFlow 核心算子一样，自定义算子会要求您[修改模型运行时环境](ops_custom#create_and_register_the_operator)，这会限制您利用标准运行时服务，例如 [Google Play 服务](../android/play_services)。

## 支持的类型

大多数 TensorFlow Lite 运算都针对的是浮点（`float32`）和量化（`uint8`、`int8`）推断，但许多算子尚不适用于其他类型（如 `tf.float16` 和字符串）。

除了使用不同版本的运算外，浮点模型和量化模型之间的另一个区别是它们的转换方式。量化转换需要张量的动态范围信息。这需要在模型训练期间进行“伪量化”，通过校准数据集获取范围信息，或进行“实时”范围估计。请参阅[量化](../performance/model_optimization.md)，了解更多详细信息。

## 直接转换、常量折叠以及融合

TensorFlow Lite 可以处理许多 TensorFlow 运算，即使它们没有直接等效项。这种情况适用于：可直接从计算图中移除的运算（`tf.identity`）、可用张量替换的运算（`tf.placeholder`）和可融合为更复杂运算的运算（`tf.nn.bias_add`）。有时甚至可以通过这些过程移除某些支持的运算。

下面是通常会从计算图中移除的 TensorFlow 运算的非详尽列表：

- `tf.add`
- `tf.debugging.check_numerics`
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
