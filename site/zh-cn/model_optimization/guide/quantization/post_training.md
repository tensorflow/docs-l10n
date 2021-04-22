# 训练后量化

训练后量化包括减少 CPU 和硬件加速器延迟、处理时间、功耗和模型大小而几乎不降低模型准确率的通用技术。这些技术可以在已经训练好的浮点 TensorFlow 模型上执行，并在 TensorFlow Lite 转换期间应用。这些技术在 [TensorFlow Lite 转换器](https://www.tensorflow.org/lite/convert/)中以选项方式启用。

要查看端到端示例，请参阅以下教程：

- [训练后动态范围量化](https://www.tensorflow.org/lite/performance/post_training_quant)
- [训练后全整数量化](https://www.tensorflow.org/lite/performance/post_training_integer_quant)
- [训练后 float16 量化](https://www.tensorflow.org/lite/performance/post_training_float16_quant)

## 量化权重

权重可能会转换为精度降低的类型，例如 16 位浮点数或 8 位整数。我们通常建议将 16 位浮点数用于 GPU 加速，而将 8 位整数用于 CPU 执行。

例如，下面给出了指定 8 位整数权重量化的方法：

```
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
```

推理时，最关键的密集部分使用 8 位而不是浮点数进行计算。与下面对权重和激活进行量化相比，存在一些推理时间性能开销。

有关详细信息，请参阅 TensorFlow Lite [训练后量化](https://www.tensorflow.org/lite/performance/post_training_quantization)指南。

## 权重和激活的全整数量化

通过确保量化权重和激活，可以改善延迟、处理时间和功耗，并访问仅支持整数的硬件加速器。这需要一个较小的代表性数据集。

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

为方便起见，生成的模型仍采用浮点输入和输出。

有关详细信息，请参阅 TensorFlow Lite [训练后量化](https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization_of_weights_and_activations)指南。
