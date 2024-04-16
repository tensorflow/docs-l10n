# Cuantización posentrenamiento

La cuantización posentrenamiento incluye técnicas generales para reducir la latencia, el procesamiento, la potencia y el tamaño del modelo del acelerador de hardware y la CPU con poca degradación en la precisión del modelo. Estas técnicas se pueden realizar en un modelo flotante de TensorFlow ya entrenado y aplicarse durante la conversión a TensorFlow Lite. Estas técnicas están habilitadas como opciones en el [convertidor TensorFlow Lite](https://www.tensorflow.org/lite/convert/).

Para pasar directamente a ejemplos de principio a fin, consulte los siguientes tutoriales:

- [Cuantización del rango dinámico postentrenamiento](https://www.tensorflow.org/lite/performance/post_training_quant)
- [Cuantización de enteros completos posentrenamiento](https://www.tensorflow.org/lite/performance/post_training_integer_quant)
- [Cuantización de float16 posentrenamiento](https://www.tensorflow.org/lite/performance/post_training_float16_quant)

## Cuantizar pesos

Los pesos se pueden convertir a tipos con precisión reducida, como flotantes de 16 bits o enteros de 8 bits. Generalmente recomendamos flotantes de 16 bits para la aceleración de la GPU y enteros de 8 bits para la ejecución de la CPU.

Por ejemplo, aquí se explica cómo especificar la cuantización de peso de enteros de 8 bits:

```
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
```

Por inferencia, las partes más críticamente intensivas se calculan con 8 bits en lugar de punto flotante. Hay cierta sobrecarga de rendimiento en el tiempo de inferencia, en relación con la cuantización tanto de los pesos como de las activaciones a continuación.

Para obtener más información, consulta la guía de [cuantización posentrenamiento](https://www.tensorflow.org/lite/performance/post_training_quantization) de TensorFlow Lite.

## Cuantización de enteros completos de pesos y activaciones

Mejore la latencia, el procesamiento y el uso de energía, y obtenga acceso a aceleradores de hardware de solo números enteros asegurándose de que tanto los pesos como las activaciones se cuanticen. Esto requiere un pequeño conjunto de datos representativos.

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

El modelo resultante seguirá recibiendo entradas y salidas flotantes por conveniencia.

Para obtener más información, consulta la guía de [cuantización posentrenamiento](https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization_of_weights_and_activations) de TensorFlow Lite.
