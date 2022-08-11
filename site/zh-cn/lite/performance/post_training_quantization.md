# 训练后量化

训练后量化是一种转换技术，它可以在改善 CPU 和硬件加速器延迟的同时缩减模型大小，且几乎不会降低模型准确率。使用 [TensorFlow Lite 转换器](../models/convert/)将已训练的浮点 TensorFlow 模型转换为 TensorFlow Lite 格式后，可以对该模型进行量化。

注：此页面上的过程需要 TensorFlow 1.15 或更高版本。

### 优化方法

有几种训练后量化选项可供选择。下面是各种选项及其优势的汇总表：

技术 | 优势 | 硬件
--- | --- | ---
动态范围 | 大小缩减至原来的四分之一，速度加快 2-3 倍 | CPU
: 量化         :                           :                  : |  |
全整数 | 大小缩减至原来的四分之一，速度加快 3+ 倍 | CPU、Edge TPU、
: 量化         :                           : 微控制器 : |  |
Float16 量化 | 大小缩减至原来的二分之一，GPU | CPU、GPU
:                      : 加速              :                  : |  |

以下决策树可以帮助确定最适合您用例的训练后量化方法：

![训练后量化选项](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/lite/performance/images/optimization.jpg?raw=true)

### 动态范围量化

训练后量化最简单的形式是仅将权重从浮点静态量化为整数（具有 8 位精度）：

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

推断时，权重从 8 位精度转换为浮点，并使用浮点内核进行计算。此转换会完成一次并缓存，以减少延迟。

为了进一步改善延迟，“动态范围”算子会根据激活的范围将其动态量化为 8 位，并使用 8 位权重和激活执行计算。此优化提供的延迟接近全定点推断。但是，输出仍使用浮点进行存储，因此使用动态范围算子的加速小于全定点计算。

### 全整数量化

您可以通过确保所有模型数学均为整数量化，进一步改善延迟，减少峰值内存用量，以及兼容仅支持整数的硬件设备或加速器。

对于全整数量化，需要校准或估算模型中所有浮点张量的范围，即 (min, max)。与权重和偏差等常量张量不同，模型输入、激活（中间层的输出）和模型输出等变量张量不能校准，除非我们运行几个推断周期。因此，转换器需要一个有代表性的数据集来校准它们。这个数据集可以是训练数据或验证数据的一个小子集（大约 100-500 个样本）。请参阅下面的 `representative_dataset()` 函数。

从 TensorFlow 2.7 版本开始，您可以通过[签名](../guide/signatures.ipynb)指定代表性数据集，示例如下：

<pre>def representative_dataset():
  for data in dataset:
    yield {
      "image": data.image,
      "bias": data.bias,
    }
</pre>

如果给定的 TensorFlow 模型中有多个签名，则可以通过指定签名密钥来指定多个数据集：

<pre>def representative_dataset():
  # Feed data set for the "encode" signature.
  for data in encode_signature_dataset:
    yield (
      "encode", {
        "image": data.image,
        "bias": data.bias,
      }
    )

  # Feed data set for the "decode" signature.
  for data in decode_signature_dataset:
    yield (
      "decode", {
        "image": data.image,
        "hint": data.hint,
      },
    )
</pre>

您可以通过提供输入张量列表来生成代表性数据集：

<pre>
def representative_dataset():
  for data in tf.data.Dataset.from_tensor_slices((images)).batch(1).take(100):
    yield [tf.dtypes.cast(data, tf.float32)]
</pre>

从 TensorFlow 2.7 版本开始，我们推荐使用基于签名的方法，而不是基于输入张量列表的方法，因为输入张量排序可以很容易地翻转。

出于测试目的，您可以使用如下所示的虚拟数据集：

<pre>
def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 244, 244, 3)
      yield [data.astype(np.float32)]
 </pre>

#### 带有浮点回退的整数（使用默认浮点输入/输出）

为了对模型进行全整数量化，但在模型没有整数实现时使用浮点算子（以确保转换顺利进行），请按照以下步骤进行操作：

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

注：为了与原始的全浮点模型具有相同的接口，此 `tflite_quant_model` 不兼容仅支持整数的设备（如 8 位微控制器）和加速器（如 Coral Edge TPU），因为输入和输出仍为浮点。

#### 仅整数

*对于[适用于微控制器的 TensorFlow Lite](https://www.tensorflow.org/lite/microcontrollers) 和 [Coral Edge TPU](https://coral.ai/)，创建全整数模型是常见的用例。*

注：从 TensorFlow 2.3.0 开始，我们支持 `inference_input_type` 和 `inference_output_type` 特性。

此外，为了确保兼容仅支持整数的设备（如 8 位微控制器）和加速器（如 Coral Edge TPU），您可以使用以下步骤对包括输入和输出在内的所有算子强制执行全整数量化：

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
&lt;b&gt;converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]&lt;/b&gt;
&lt;b&gt;converter.inference_input_type = tf.int8&lt;/b&gt;  # or tf.uint8
&lt;b&gt;converter.inference_output_type = tf.int8&lt;/b&gt;  # or tf.uint8
tflite_quant_model = converter.convert()
</pre>

注：如果遇到当前无法量化的运算，转换器会引发错误。

### float16 量化

您可以通过将权重量化为 float16（16 位浮点数的 IEEE 标准）来缩减浮点模型的大小。要启用权重的 float16 量化，请使用以下步骤：

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

float16 量化的优点如下：

- 将模型的大小缩减一半（因为所有权重都变成其原始大小的一半）。
- 实现最小的准确率损失。
- 支持可直接对 float16 数据进行运算的部分委托（例如 GPU 委托），从而使执行速度比 float32 计算更快。

float16 量化的缺点如下：

- 它不像对定点数学进行量化那样减少那么多延迟。
- 默认情况下，float16 量化模型在 CPU 上运行时会将权重值“反量化”为 float32。（请注意，GPU 委托不会执行此反量化，因为它可以对 float16 数据进行运算。）

### 仅整数：具有 8 位权重的 16 位激活（实验性）

这是一个实验性量化方案。它与“仅整数”方案类似，但会根据激活的范围将其量化为 16 位，权重会被量化为 8 位整数，偏差会被量化为 64 位整数。这被进一步称为 16x8 量化。

这种量化的主要优点是可以显著提高准确率，但只会稍微增加模型大小。

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.representative_dataset = representative_dataset
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

如果模型中的部分算子不支持 16x8 量化，模型仍然可以量化，但不受支持的算子会保留为浮点。要允许此操作，应将以下选项添加到 target_spec 中。

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.representative_dataset = representative_dataset
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
&lt;b&gt;tf.lite.OpsSet.TFLITE_BUILTINS&lt;/b&gt;]
tflite_quant_model = converter.convert()
</pre>

通过此量化方案提高了准确率的用例示例包括：* 超分辨率、* 音频信号处理（如降噪和波束成形）、* 图像降噪、* 单个图像 HDR 重建。

这种量化的缺点是：

- 由于缺少优化的内核实现，目前的推断速度明显比 8 位全整数慢。
- 目前它不兼容现有的硬件加速 TFLite 委托。

注：这是一项实验性功能。

可以在[此处](post_training_integer_quant_16x8.ipynb)找到该量化模型的教程。

### 模型准确率

由于权重是在训练后量化的，因此可能会造成准确率损失，对于较小的网络尤其如此。[TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite&q=quantized){:.external} 为特定网络提供了预训练的完全量化模型。请务必检查量化模型的准确率，以验证准确率的任何下降都在可接受的范围内。有一些工具可以评估 [TensorFlow Lite 模型准确率](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks){:.external}。

另外，如果准确率下降过多，请考虑使用[量化感知训练](https://www.tensorflow.org/model_optimization/guide/quantization/training)。但是，这样做需要在模型训练期间进行修改以添加伪量化节点，而此页面上的训练后量化技术使用的是现有的预训练模型。

### 量化张量的表示

8 位量化近似于使用以下公式得到的浮点值。

$$real_value = (int8_value - zero_point) \times scale$$

该表示包含两个主要部分：

- 由 int8 补码值表示的逐轴（即逐通道）或逐张量权重，范围为 [-127, 127]，零点等于 0。

- 由 int8 补码值表示的按张量激活/输入，范围为 [-128, 127]，零点范围为 [-128, 127]。

有关量化方案的详细信息，请参阅我们的[量化规范](./quantization_spec)。对于想要插入 TensorFlow Lite 委托接口的硬件供应商，我们鼓励您实现此规范中描述的量化方案。
