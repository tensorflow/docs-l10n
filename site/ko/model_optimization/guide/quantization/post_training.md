# 훈련 후 양자화

Post-training quantization includes general techniques to reduce CPU and hardware accelerator latency, processing, power, and model size with little degradation in model accuracy. These techniques can be performed on an already-trained float TensorFlow model and applied during TensorFlow Lite conversion. These techniques are enabled as options in the [TensorFlow Lite converter](https://www.tensorflow.org/lite/convert/).

To jump right into end-to-end examples, see the following tutorials:

- [훈련 후 동적 범위 양자화](https://www.tensorflow.org/lite/performance/post_training_quant)
- [Post-training full integer quantization](https://www.tensorflow.org/lite/performance/post_training_integer_quant)
- [훈련 후 float16 양자화](https://www.tensorflow.org/lite/performance/post_training_float16_quant)

## 가중치 양자화

Weights can be converted to types with reduced precision, such as 16 bit floats or 8 bit integers. We generally recommend 16-bit floats for GPU acceleration and 8-bit integer for CPU execution.

For example, here is how to specify 8 bit integer weight quantization:

```
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
```

At inference, the most critically intensive parts are computed with 8 bits instead of floating point. There is some inference-time performance overhead, relative to quantizing both weights and activations below.

For more information, see the TensorFlow Lite [post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization) guide.

## Full integer quantization of weights and activations

Improve latency, processing, and power usage, and get access to integer-only hardware accelerators by making sure both weights and activations are quantized. This requires a small representative data set.

```
import tensorflow as tf

def representative_dataset_gen():
  for _ in range(num_calibration_steps):
    # Get sample input data as a numpy array in a method of your choosing.
    yield [input]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
```

결과 모델은 편의를 위해 여전히 float 입력 및 출력을 사용합니다.

For more information, see the TensorFlow Lite [post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization_of_weights_and_activations) guide.
