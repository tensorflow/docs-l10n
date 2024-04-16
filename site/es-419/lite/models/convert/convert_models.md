# Convertir modelos TensorFlow

Esta página describe cómo convertir un modelo TensorFlow en un modelo TensorFlow Lite (un formato optimizado [FlatBuffer](https://google.github.io/flatbuffers/) identificado por la extensión de archivo `.tflite`) usando el conversor de TensorFlow Lite.

Nota: Esta guía asume que usted ha [instalado TensorFlow 2.x](https://www.tensorflow.org/install/pip#tensorflow-2-packages-are-available) y entrenado modelos en TensorFlow 2.x. Si su modelo está entrenado en TensorFlow 1.x, considere [migrar a TensorFlow 2.x](https://www.tensorflow.org/guide/migrate/tflite). Para identificar la versión de TensorFlow instalada, ejecute `print(tf.__version__)`.

## Flujo de trabajo de conversión

El diagrama siguiente ilustra el flujo de trabajo de alto nivel para convertir su modelo:

![Flujo de trabajo del convertidor de TFLite](../../images/convert/convert.png)

**Figura 1.** Flujo de trabajo del convertidor.

Puede convertir su modelo usando una de las siguientes opciones:

1. [API de Python](#python_api) (***recomendada***): Esto le permite integrar la conversión en su flujo de desarrollo, aplicar optimizaciones, añadir metadatos y muchas otras tareas que simplifican el proceso de conversión.
2. [Línea de comandos](#cmdline): Sólo admite la conversión básica de modelos.

Nota: En caso de que encuentre algún problema durante la conversión del modelo, cree un [GitHub issue](https://github.com/tensorflow/tensorflow/issues/new?template=60-tflite-converter-issue.md).

## API de Python <a name="python_api"></a>

*Código ayudante: Para obtener más información sobre la API del conversor TensorFlow Lite, ejecute `print(help(tf.lite.TFLiteConverter))`.*

Convierta un modelo TensorFlow usando [`tf.lite.TFLiteConverter`](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter). Un modelo TensorFlow se almacena usando el formato SavedModel y se genera bien usando las APIs de alto nivel `tf.keras.*` (un modelo Keras) o las APIs de bajo nivel `tf.*` (a partir de las cuales genera funciones concretas). Resultan las tres opciones siguientes (encontrará ejemplos en las próximas secciones):

- `tf.lite.TFLiteConverter.from_saved_model()` (**recomendado**): Convierte un [SavedModel](https://www.tensorflow.org/guide/saved_model).
- `tf.lite.TFLiteConverter.from_keras_model()`: Convierte un modelo [Keras](https://www.tensorflow.org/guide/keras/overview).
- `tf.lite.TFLiteConverter.from_concrete_functions()`: Convierte [funciones concretas](https://www.tensorflow.org/guide/intro_to_graphs).

### Convert a SavedModel (recomendado) <a name="saved_model"></a>

El siguiente ejemplo muestra cómo convertir un [SavedModel](https://www.tensorflow.org/guide/saved_model) en un modelo TensorFlow Lite.

```python
import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

### Convertir un modelo Keras <a name="keras"></a>

El siguiente ejemplo muestra cómo convertir un modelo [Keras](https://www.tensorflow.org/guide/keras/overview) en un modelo TensorFlow Lite.

```python
import tensorflow as tf

# Create a model using high-level tf.keras.* APIs
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1]),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
model.compile(optimizer='sgd', loss='mean_squared_error') # compile the model
model.fit(x=[-1, 0, 1], y=[-3, -1, 1], epochs=5) # train the model
# (to generate a SavedModel) tf.saved_model.save(model, "saved_model_keras_dir")

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

### Convertir funciones concretas <a name="concrete_function"></a>

El siguiente ejemplo muestra cómo convertir [funciones concretas](https://www.tensorflow.org/guide/intro_to_graphs) en un modelo TensorFlow Lite.

```python
import tensorflow as tf

# Create a model using low-level tf.* APIs
class Squared(tf.Module):
  @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
  def __call__(self, x):
    return tf.square(x)
model = Squared()
# (ro run your model) result = Squared(5.0) # This prints "25.0"
# (to generate a SavedModel) tf.saved_model.save(model, "saved_model_tf_dir")
concrete_func = model.__call__.get_concrete_function()

# Convert the model.

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func],
                                                            model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

### Otras características

- Aplique [optimizaciones](../../performance/model_optimization.md). Una optimización comúnmente usada es la [cuantización posterior al entrenamiento](../../performance/post_training_quantization.md), que puede reducir aún más la latencia y el tamaño de su modelo con una pérdida mínima de precisión.

- Añada [metadatos](metadata.md), lo que facilita la creación de código contenedor específico de la plataforma al implementar modelos en dispositivos.

### Errores de conversión

A continuación se indican los errores de conversión más comunes y sus soluciones:

- Error: `Some ops are not supported by the native TFLite runtime, you can enable TF kernels fallback using TF Select. See instructions: https://www.tensorflow.org/lite/guide/ops_select. TF Select ops: ..., .., ...`

    Solución: El error se produce porque su modelo tiene ops TF sin la correspondiente implementación en TFLite. Puede resolverlo [usando la op TF en el modelo TFLite](../../guide/ops_select.md) (recomendado). Si desea generar un modelo sólo con ops TFLite, puede añadir una solicitud para la op TFLite que falta en [Github issue #21526](https://github.com/tensorflow/tensorflow/issues/21526) (deje un comentario si su solicitud no se ha mencionado ya) o [crear usted mismo la op TFLite](../../guide/ops_custom#create_and_register_the_operator).

- Error: `.. is neither a custom op nor a flex op`

    Solución: Si esta op TF:

    - Está admitida en TF: El error se produce porque falta la op TF en la [allowlist](../../guide/op_select_allowlist.md) (una lista exhaustiva de las ops TF admitidas por TFLite). Puede resolverlo de la siguiente manera:

        1. [Añadir las ops que faltan a la allowlist](../../guide/op_select_allowlist.md#add_tensorflow_core_operators_to_the_allowed_list)[.](../../guide/op_select_allowlist.md#add_tensorflow_core_operators_to_the_allowed_list)
        2. [Convertir el modelo TF en un modelo TFLite y ejecutar la inferencia](../../guide/ops_select.md).

    - No está admitida en TF: El error se produce porque TFLite desconoce el operario de TF personalizado definido por usted. Puede resolverlo de la siguiente manera:

        1. [Crear la op TF](https://www.tensorflow.org/guide/create_op).
        2. [Convertir el modelo TF en un modelo TFLite](../../guide/op_select_allowlist.md#users_defined_operators).
        3. [Crear la op TFLite](../../guide/ops_custom.md#create_and_register_the_operator) y ejecutar la inferencia vinculándola al runtime TFLite.

## Herramienta de línea de comandos <a name="cmdline"></a>

**Nota:** Se recomienda especialmente usar en su lugar, si es posible, la [API de Python](#python_api) indicada anteriormente.

Si ha [instalado TensorFlow 2.x desde pip](https://www.tensorflow.org/install/pip), use el comando `tflite_convert`. Para ver todos los Indicadores disponibles, use el siguiente comando:

```sh
$ tflite_convert --help

`--output_file`. Type: string. Full path of the output file.
`--saved_model_dir`. Type: string. Full path to the SavedModel directory.
`--keras_model_file`. Type: string. Full path to the Keras H5 model file.
`--enable_v1_converter`. Type: bool. (default False) Enables the converter and flags used in TF 1.x instead of TF 2.x.

You are required to provide the `--output_file` flag and either the `--saved_model_dir` or `--keras_model_file` flag.
```

Si usted tiene la [fuente TensorFlow 2.x](https://www.tensorflow.org/install/source) descargada y desea ejecutar el convertidor desde esa fuente sin compilar e instalar el paquete, puede reemplazar '`tflite_convert`' por '`bazel run tensorflow/lite/python:tflite_convert --`' en el comando.

### Convertir un SavedModel <a name="cmdline_saved_model"></a>

```sh
tflite_convert \
  --saved_model_dir=/tmp/mobilenet_saved_model \
  --output_file=/tmp/mobilenet.tflite
```

### Conversión de un modelo Keras H5 <a name="cmdline_keras_model"></a>

```sh
tflite_convert \
  --keras_model_file=/tmp/mobilenet_keras_model.h5 \
  --output_file=/tmp/mobilenet.tflite
```

## Siguientes pasos

Usar el [intérprete TensorFlow Lite](../../guide/inference.md) para ejecutar la inferencia en un dispositivo cliente (por ejemplo, móvil, integrado).
