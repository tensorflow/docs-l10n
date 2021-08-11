# 训练后量化

Post-training quantization is a conversion technique that can reduce model size while also improving CPU and hardware accelerator latency, with little degradation in model accuracy. You can quantize an already-trained float TensorFlow model when you convert it to TensorFlow Lite format using the [TensorFlow Lite Converter](../convert/).

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

![post-training optimization options](images/optimization.jpg)

### 动态范围量化

训练后量化最简单的形式是仅将权重从浮点静态量化为整数（具有 8 位精度）：

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

At inference, weights are converted from 8-bits of precision to floating point and computed using floating-point kernels. This conversion is done once and cached to reduce latency.

To further improve latency, "dynamic-range" operators dynamically quantize activations based on their range to 8-bits and perform computations with 8-bit weights and activations. This optimization provides latencies close to fully fixed-point inference. However, the outputs are still stored using floating point so that the speedup with dynamic-range ops is less than a full fixed-point computation.

### 全整数量化

您可以通过确保所有模型数学均为整数量化，进一步改善延迟，减少峰值内存用量，以及兼容仅支持整数的硬件设备或加速器。

对于全整数量化，需要校准或估算模型中所有浮点张量的范围，即 (min, max)。与权重和偏差等常量张量不同，模型输入、激活（中间层的输出）和模型输出等变量张量不能校准，除非我们运行几个推断周期。因此，转换器需要一个有代表性的数据集来校准它们。这个数据集可以是训练数据或验证数据的一个小子集（大约 100-500 个样本）。请参阅下面的 `representative_dataset()` 函数。

<pre>def representative_dataset():
  for data in tf.data.Dataset.from_tensor_slices((images)).batch(1).take(100):
    yield [tf.dtypes.cast(data, tf.float32)]
</pre>

出于测试目的，您可以使用如下所示的虚拟数据集：

<pre>def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 244, 244, 3)
      yield [data.astype(np.float32)]
 </pre>

#### 带有浮点回退的整数（使用默认浮点输入/输出）

为了对模型进行全整数量化，但在模型没有整数实现时使用浮点算子（以确保转换顺利进行），请按照以下步骤进行操作：

<pre>import tensorflow as tf
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

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
&lt;b&gt;converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]&lt;/b&gt;
&lt;b&gt;converter.inference_input_type = tf.int8&lt;/b&gt;  # or tf.uint8
&lt;b&gt;converter.inference_output_type = tf.int8&lt;/b&gt;  # or tf.uint8
tflite_quant_model = converter.convert()
</pre>

Note: The converter will throw an error if it encounters an operation it cannot currently quantize.

### float16 量化

You can reduce the size of a floating point model by quantizing the weights to float16, the IEEE standard for 16-bit floating point numbers. To enable float16 quantization of weights, use the following steps:

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

float16 量化的优点如下：

- It reduces model size by up to half (since all weights become half of their original size).
- 实现最小的准确率损失。
- 支持可直接对 float16 数据进行运算的部分委托（例如 GPU 委托），从而使执行速度比 float32 计算更快。

float16 量化的缺点如下：

- 它不像对定点数学进行量化那样减少那么多延迟。
- By default, a float16 quantized model will "dequantize" the weights values to float32 when run on the CPU. (Note that the GPU delegate will not perform this dequantization, since it can operate on float16 data.)

### 仅整数：具有 8 位权重的 16 位激活（实验性）

这是一个实验性量化方案。它与“仅整数”方案类似，但会根据激活的范围将其量化为 16 位，权重会被量化为 8 位整数，偏差会被量化为 64 位整数。这被进一步称为 16x8 量化。

这种量化的主要优点是可以显著提高准确率，但只会稍微增加模型大小。

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.representative_dataset = representative_dataset
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

如果模型中的部分算子不支持 16x8 量化，模型仍然可以量化，但不受支持的算子会保留为浮点。要允许此操作，应将以下选项添加到 target_spec 中。

<pre>import tensorflow as tf
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

由于权重是在训练后量化的，因此可能会造成准确率损失，对于较小的网络更是如此。[TensorFlow Lite 模型存储库](../models/)为特定网络提供了预训练的完全量化模型。请务必检查量化模型的准确率，以验证准确率的任何下降都在可接受的范围内。有一些工具可以评估 [TensorFlow Lite 模型准确率](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks){:.external}。

Alternatively, if the accuracy drop is too high, consider using [quantization aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training) . However, doing so requires modifications during model training to add fake quantization nodes, whereas the post-training quantization techniques on this page use an existing pre-trained model.

### 量化张量的表示

8-bit quantization approximates floating point values using the following formula.

$$real_value = (int8_value - zero_point) \times scale$$

该表示包含两个主要部分：

- 由 int8 补码值表示的逐轴（即逐通道）或逐张量权重，范围为 [-127, 127]，零点等于 0。

- 由 int8 补码值表示的逐张量激活/输入，范围为 [-128, 127]，零点范围为 [-128, 127]。

有关量化方案的详细信息，请参阅我们的[量化规范](./quantization_spec.md)。对于想要插入 TensorFlow Lite 委托接口的硬件供应商，我们鼓励您实现此规范中描述的量化方案。
