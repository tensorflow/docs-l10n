# Cuantización posentrenamiento

La cuantización postentrenamiento es una técnica de conversión que puede reducir el tamaño del modelo a la vez que mejora la latencia de la CPU y del acelerador de hardware, con poca degradación de la precisión del modelo. Puede cuantizar un modelo TensorFlow flotante ya entrenado cuando lo convierta al formato TensorFlow Lite utilizando el [Conversor TensorFlow Lite](../models/convert/).

Nota: Los procedimientos de esta página requieren TensorFlow 1.15 o superior.

### Métodos de optimización

Hay varias opciones de cuantización postentrenamiento entre las que seleccionar. Aquí tiene un cuadro resumen de las opciones y de los beneficios que proporcionan:

Técnica | Beneficios | Hardware
--- | --- | ---
Rango dinámico | 4 veces más pequeño, 2-3 veces más rápido | CPU
: cuantización         :                           :                  : |  |
Completa de enteros | 4 veces más pequeño, más de 3 veces más rápido | CPU, Edge TPU,
: cuantización         :                           : Microcontroladores : |  |
Cuantización Float16 | 2 veces más pequeño, GPU | CPU, GPU
:                      : aceleración              :                  : |  |

El siguiente árbol de decisión puede ayudarle a determinar qué método de cuantización postentrenamiento es el mejor para su caso de uso:

![opciones de optimización postentrenamiento](images/optimization.jpg)

### Cuantización del rango dinámico

La cuantización de rango dinámico es un punto de partida recomendado porque ofrece un uso reducido de memoria y un cálculo más rápido sin tener que aportar un conjunto de datos representativo para la calibración. Este tipo de cuantización, cuantiza estáticamente sólo las ponderaciones de punto flotante a entero en el momento de la conversión, lo que da 8 bits de precisión:

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

Para reducir aún más la latencia durante la inferencia, los operarios de "rango dinámico" cuantifican dinámicamente las activaciones en función de su rango a 8 bits y realizan los cálculos con ponderaciones y activaciones de 8 bits. Esta optimización proporciona latencias cercanas a las inferencias totalmente en punto fijo. Sin embargo, las salidas se siguen almacenando usando punto flotante, por lo que el aumento de velocidad de los ops de rango dinámico es menor que el de un cálculo completo en punto fijo.

### Cuantización de entero completo

Puede lograr más mejoras de latencia, reducciones en el uso máximo de memoria y compatibilidad con dispositivos de hardware o aceleradores de sólo enteros si se asegura de que todas las matemáticas del modelo están cuantizadas en enteros.

Para una cuantización entera completa, es necesario calibrar o estimar el rango, es decir, (mín, máx) de todos los tensores de punto flotante del modelo. Contrariamente a los tensores fijos, como las ponderaciones y los sesgos, los tensores variables, como la entrada del modelo, las activaciones (salidas de las capas intermedias) y la salida del modelo, no pueden calibrarse a menos que ejecutemos algunos ciclos de inferencia. Como resultado, el convertidor necesita un conjunto de datos representativo para calibrarlos. Este conjunto de datos puede ser un subconjunto pequeño (alrededor de ~100-500 muestras) de los datos de entrenamiento o validación. Si desea más información, consulte la función `representative_dataset()` que aparece a continuación.

A partir de la versión 2.7 de TensorFlow, puede especificar el conjunto de datos representativo mediante una [firma](../guide/signatures.ipynb) como en el siguiente ejemplo:

<pre>
def representative_dataset():
  for data in dataset:
    yield {
      "image": data.image,
      "bias": data.bias,
    }
</pre>

Si hay más de una firma en el modelo TensorFlow dado, puede especificar el conjunto de datos múltiple especificando las claves de la firma:

<pre>
def representative_dataset():
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

Puede generar el conjunto de datos representativo proporcionando una lista de tensores de entrada:

<pre>
def representative_dataset():
  for data in tf.data.Dataset.from_tensor_slices((images)).batch(1).take(100):
    yield [tf.dtypes.cast(data, tf.float32)]
</pre>

Desde la versión 2.7 de TensorFlow, recomendamos usar el enfoque basado en firmas en lugar del enfoque basado en listas de tensores de entrada, ya que el orden de los tensores de entrada puede invertirse fácilmente.

Para realizar pruebas, puede usar un conjunto de datos ficticio como el siguiente:

<pre>
def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 244, 244, 3)
      yield [data.astype(np.float32)]
 </pre>

#### Entero con flotante como reserva (usando la entrada/salida predeterminada de flotante)

Para cuantificar completamente en enteros un modelo, pero usar operadores flotantes cuando no tienen una implementación en enteros (para asegurar que la conversión se produce sin problemas), utilice los siguientes pasos:

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

Nota: Este `tflite_quant_model` no será compatible con dispositivos de sólo enteros (como microcontroladores de 8 bits) y aceleradores (como la TPU Coral Edge) porque la entrada y la salida siguen siendo flotantes para tener la misma interfaz que el modelo original de sólo flotantes.

#### Sólo enteros

*Crear modelos de sólo enteros es un caso de uso común para [TensorFlow Lite para microcontroladores](https://www.tensorflow.org/lite/microcontrollers) y [Coral Edge TPUs](https://coral.ai/).*

Nota: A partir de TensorFlow 2.3.0, admitimos los atributos `inference_input_type` y `inference_output_type`.

Además, para asegurar la compatibilidad con dispositivos de sólo enteros (como los microcontroladores de 8 bits) y aceleradores (como la TPU Coral Edge), puede forzar la cuantización completa de enteros para todas las ops, incluyendo la entrada y la salida, usando los siguientes pasos:

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
&lt;b&gt;converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]&lt;/b&gt;
&lt;b&gt;converter.inference_input_type = tf.int8&lt;/b&gt;  # or tf.uint8
&lt;b&gt;converter.inference_output_type = tf.int8&lt;/b&gt;  # or tf.uint8
tflite_quant_model = converter.convert()
</pre>

### Cuantización Float16

Puede reducir el tamaño de un modelo de punto flotante cuantizando las ponderaciones a float16, el estándar IEEE para números de punto flotante de 16 bits. Para habilitar la cuantización en float16 de las ponderaciones, use los siguientes pasos:

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

Las ventajas de la cuantización float16 son las siguientes:

- Reduce el tamaño del modelo hasta la mitad (ya que todas las ponderaciones pasan a ser la mitad de su tamaño original).
- La pérdida de precisión es mínima.
- Admite algunos delegados (por ejemplo, el delegado de la GPU) que pueden operar directamente sobre datos float16, lo que resulta en una ejecución más rápida que los cálculos float32.

Las desventajas de la cuantización float16 son las siguientes:

- No reduce la latencia tanto como una cuantización a matemáticas de punto fijo.
- De forma predeterminada, un modelo cuantizado float16 "decuantizará" los valores de las ponderaciones a float32 cuando se ejecute en la CPU (tenga en cuenta que el delegado de la GPU no realizará esta decuantización, ya que puede operar con datos float16).

### Sólo enteros: activaciones de 16 bits con ponderaciones de 8 bits (experimental)

Se trata de un esquema de cuantización experimental. Es similar al esquema de "sólo enteros", pero las activaciones se cuantifican en función de su rango a 16 bits, las ponderaciones se cuantifican en enteros de 8 bits y la ponderación se cuantifica en enteros de 64 bits. En adelante a esto se llama cuantización 16x8.

La principal ventaja de esta cuantización es que puede mejorar la precisión de forma significativa, pero sólo aumenta ligeramente el tamaño del modelo.

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.representative_dataset = representative_dataset
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

Si no se admite la cuantización 16x8 para algunos operarios del modelo, entonces el modelo aún puede cuantizarse, pero los operarios no admitidos se mantienen en flotante. La siguiente opción debe añadirse al target_spec para permitir esto.

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.representative_dataset = representative_dataset
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
&lt;b&gt;tf.lite.OpsSet.TFLITE_BUILTINS&lt;/b&gt;]
tflite_quant_model = converter.convert()
</pre>

Entre los ejemplos de casos de uso en los que se mejora la precisión gracias a este esquema de cuantización se incluyen:

- superresolución,
- procesamiento de señales de audio, como la cancelación de ruido y la formación de haces,
- eliminación de ruido de la imagen,
- Reconstrucción HDR a partir de una sola imagen.

La desventaja de esta cuantización es:

- Actualmente la inferencia es notablemente más lenta que la de enteros completos de 8 bits debido a la falta de una implementación optimizada del kernel.
- Actualmente es incompatible con los delegados TFLite acelerados por hardware existentes.

Nota: Se trata de una función experimental.

[Aquí](post_training_integer_quant_16x8.ipynb) puede encontrar un tutorial sobre este modo de cuantización.

### Precisión del modelo

Dado que las ponderaciones se cuantizan tras el entrenamiento, podría producirse una pérdida de precisión, sobre todo en las redes más pequeñas. Se ofrecen modelos cuantizados preentrenados para redes específicas en [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite&q=quantized){:.external}. Es importante revisar la precisión del modelo cuantizado para verificar que cualquier degradación en la precisión se encuentra dentro de límites aceptables. Existen herramientas para evaluar la precisión del modelo [TensorFlow Lite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks){:.external}.

Alternativamente, si la caída de la precisión es demasiado alta, considere usar [entrenamiento consciente de la cuantización](https://www.tensorflow.org/model_optimization/guide/quantization/training) . Sin embargo, hacerlo requiere modificaciones durante el entrenamiento del modelo para añadir nodos de cuantización falsos, mientras que las técnicas de cuantización postentrenamiento de esta página usan un modelo preentrenado existente.

### Representación para tensores cuantizados

La cuantización de 8 bits aproxima los valores de punto flotante usando la siguiente fórmula.

$$real_value = (int8_value - zero_point) \veces scale$$

La representación consta de dos partes principales:

- Ponderaciones por eje (también llamadas "por canal") o por tensor representadas por valores de complemento a dos int8 en el intervalo [-127, 127] con punto cero igual a 0.

- Activaciones/entradas por sensor representadas por valores int8 de complemento a dos en el rango [-128, 127], con un punto cero en el rango [-128, 127].

Para una visión detallada de nuestro esquema de cuantización, consulte nuestra [especificación de cuantización](./quantization_spec). Se recomienda a los proveedores de hardware que deseen conectarse a la interfaz de delegado de TensorFlow Lite que implementen el esquema de cuantización allí descrito.
