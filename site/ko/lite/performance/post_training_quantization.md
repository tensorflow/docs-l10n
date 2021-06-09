# 훈련 후 양자화

Post-training quantization is a conversion technique that can reduce model size while also improving CPU and hardware accelerator latency, with little degradation in model accuracy. You can quantize an already-trained float TensorFlow model when you convert it to TensorFlow Lite format using the [TensorFlow Lite Converter](../convert/).

Note: The procedures on this page require TensorFlow 1.15 or higher.

### Optimization Methods

There are several post-training quantization options to choose from. Here is a summary table of the choices and the benefits they provide:

기술 | Benefits | 하드웨어
--- | --- | ---
동적 범위 | 4x smaller, 2x-3x speedup | CPU
: 양자화 : : : |  |
전체 정수 | 4x smaller, 3x+ speedup | CPU, Edge TPU,
: 양자화 : : 마이크로 컨트롤러 : |  |
Float16 양자화 | 2x smaller, GPU | CPU, GPU
: : 가속 : : |  |

The following decision tree can help determine which post-training quantization method is best for your use case:

![훈련 후 최적화 옵션](images/optimization.jpg)

### 동적 범위 양자화

The simplest form of post-training quantization statically quantizes only the weights from floating point to integer, which has 8-bits of precision:

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

At inference, weights are converted from 8-bits of precision to floating point and computed using floating-point kernels. This conversion is done once and cached to reduce latency.

To further improve latency, "dynamic-range" operators dynamically quantize activations based on their range to 8-bits and perform computations with 8-bit weights and activations. This optimization provides latencies close to fully fixed-point inference. However, the outputs are still stored using floating point so that the speedup with dynamic-range ops is less than a full fixed-point computation.

### Full integer quantization

You can get further latency improvements, reductions in peak memory usage, and compatibility with integer only hardware devices or accelerators by making sure all model math is integer quantized.

For full integer quantization, you need to calibrate or estimate the range, i.e, (min, max) of all floating-point tensors in the model. Unlike constant tensors such as weights and biases, variable tensors such as model input, activations (outputs of intermediate layers) and model output cannot be calibrated unless we run a few inference cycles. As a result, the converter requires a representative dataset to calibrate them. This dataset can be a small subset (around ~100-500 samples) of the training or validation data. Refer to the `representative_dataset()` function below.

<pre>def representative_dataset():
  for data in tf.data.Dataset.from_tensor_slices((images)).batch(1).take(100):
    yield [tf.dtypes.cast(data, tf.float32)]
</pre>

For testing purposes, you can use a dummy dataset as follows:

<pre>def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 244, 244, 3)
      yield [data.astype(np.float32)]
 </pre>

#### Integer with float fallback (using default float input/output)

In order to fully integer quantize a model, but use float operators when they don't have an integer implementation (to ensure conversion occurs smoothly), use the following steps:

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

Note: This `tflite_quant_model` won't be compatible with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU) because the input and output still remain float in order to have the same interface as the original float only model.

#### Integer only

*Creating integer only models is a common use case for [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) and [Coral Edge TPUs](https://coral.ai/).*

Note: Starting TensorFlow 2.3.0, we support the `inference_input_type` and `inference_output_type` attributes.

Additionally, to ensure compatibility with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU), you can enforce full integer quantization for all ops including the input and output, by using the following steps:

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

### Float16 양자화

You can reduce the size of a floating point model by quantizing the weights to float16, the IEEE standard for 16-bit floating point numbers. To enable float16 quantization of weights, use the following steps:

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

float16 양자화의 장점은 다음과 같습니다.

- It reduces model size by up to half (since all weights become half of their original size).
- It causes minimal loss in accuracy.
- It supports some delegates (e.g. the GPU delegate) which can operate directly on float16 data, resulting in faster execution than float32 computations.

float16 양자화의 단점은 다음과 같습니다.

- 고정 소수점 수학에 대한 양자화만큼 지연 시간을 줄이지 않습니다.
- By default, a float16 quantized model will "dequantize" the weights values to float32 when run on the CPU. (Note that the GPU delegate will not perform this dequantization, since it can operate on float16 data.)

### Integer only: 16-bit activations with 8-bit weights (experimental)

This is an experimental quantization scheme. It is similar to the "integer only" scheme, but activations are quantized based on their range to 16-bits, weights are quantized in 8-bit integer and bias is quantized into 64-bit integer. This is referred to as 16x8 quantization further.

The main advantage of this quantization is that it can improve accuracy significantly, but only slightly increase model size.

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.representative_dataset = representative_dataset
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

If 16x8 quantization is not supported for some operators in the model, then the model still can be quantized, but unsupported operators kept in float. The following option should be added to the target_spec to allow this.

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.representative_dataset = representative_dataset
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
&lt;b&gt;tf.lite.OpsSet.TFLITE_BUILTINS&lt;/b&gt;]
tflite_quant_model = converter.convert()
</pre>

Examples of the use cases where accuracy improvements provided by this quantization scheme include: * super-resolution, * audio signal processing such as noise cancelling and beamforming, * image de-noising, * HDR reconstruction from a single image.

이 양자화의 단점은 다음과 같습니다.

- Currently inference is noticeably slower than 8-bit full integer due to the lack of optimized kernel implementation.
- Currently it is incompatible with the existing hardware accelerated TFLite delegates.

Note: This is an experimental feature.

A tutorial for this quantization mode can be found [here](post_training_integer_quant_16x8.ipynb).

### Model accuracy

Since weights are quantized post training, there could be an accuracy loss, particularly for smaller networks. Pre-trained fully quantized models are provided for specific networks in the [TensorFlow Lite model repository](../models/). It is important to check the accuracy of the quantized model to verify that any degradation in accuracy is within acceptable limits. There are tools to evaluate [TensorFlow Lite model accuracy](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks){:.external}.

Alternatively, if the accuracy drop is too high, consider using [quantization aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training) . However, doing so requires modifications during model training to add fake quantization nodes, whereas the post-training quantization techniques on this page use an existing pre-trained model.

### Representation for quantized tensors

8-bit quantization approximates floating point values using the following formula.

$$real_value = (int8_value - zero_point) \times scale$$

표현에는 두 가지 주요 부분이 있습니다.

- Per-axis (aka per-channel) or per-tensor weights represented by int8 two’s complement values in the range [-127, 127] with zero-point equal to 0.

- Per-tensor activations/inputs represented by int8 two’s complement values in the range [-128, 127], with a zero-point in range [-128, 127].

For a detailed view of our quantization scheme, please see our [quantization spec](./quantization_spec.md). Hardware vendors who want to plug into TensorFlow Lite's delegate interface are encouraged to implement the quantization scheme described there.
