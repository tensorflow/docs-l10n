# SavedModels de TF Hub en TensorFlow 2

El [formato SavedModel de TensorFlow 2](https://www.tensorflow.org/guide/saved_model) es la forma recomendada de compartir modelos y partes de modelo preentrenados en TensorFlow Hub. Reemplaza el antiguo [formato TF1 Hub](tf1_hub_module.md) y viene con un conjunto de API nuevo.

En esta página se explica cómo reutilizar TF2 SavedModels en un programa de TensorFlow 2 con la API `hub.load()` de bajo nivel y su contenedor `hub.KerasLayer`. (Por lo general, `hub.KerasLayer` se combina con otros `tf.keras.layers` para crear un modelo Keras o el `model_fn` de un TF2 Estimador). Estas API también pueden cargar los modelos heredados en formato TF1 Hub, dentro de ciertos límites, consulte la [guía de compatibilidad](model_compatibility.md).

Los usuarios de TensorFlow 1 pueden actualizar a TF 1.15 y luego usar las mismas API. Las versiones anteriores de TF1 no funcionan.

## Usar SavedModel de TF Hub

### Usar un SavedModel en Keras

[Keras](https://www.tensorflow.org/guide/keras/) es la API de alto nivel de TensorFlow para crear modelos de aprendizaje profundo mediante la composición de objetos Keras Layer. La biblioteca `tensorflow_hub` proporciona la clase `hub.KerasLayer` que se inicializa con la URL (o ruta del sistema de archivos) de un SavedModel y luego proporciona el cálculo del SavedModel, incluidos sus pesos preentrenados.

A continuación se muestra un ejemplo del uso de una incrustación de texto preentrenada:

```python
import tensorflow as tf
import tensorflow_hub as hub

hub_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
embed = hub.KerasLayer(hub_url)
embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
print(embeddings.shape, embeddings.dtype)
```

A partir de esto, se puede construir un clasificador de texto de la forma habitual en Keras:

```python
model = tf.keras.Sequential([
    embed,
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
```

La [colab de clasificación de texto](https://colab.research.google.com/github/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/tf2_text_classification.ipynb) es un ejemplo completo de cómo entrenar y evaluar dicho clasificador.

Los pesos del modelo en un `hub.KerasLayer` están configurados como no entrenables de forma predeterminada. Consulte la sección sobre ajuste a continuación para saber cómo cambiarlo. Los pesos se comparten entre todas las aplicaciones del mismo objeto de capa, como es habitual en Keras.

### Usar un SavedModel en un Estimador

Los usuarios de la API [Estimator](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator) de TensorFlow para entrenamiento distribuido pueden usar SavedModels de TF Hub al escribir su `model_fn` en términos de `hub.KerasLayer` entre otras `tf.keras.layers`.

### Detrás de escena: descarga y almacenamiento en caché de SavedModel

Al usar un SavedModel de TensorFlow Hub (u otros servidores HTTPS que implementen su protocolo [de alojamiento](hosting.md)) se descarga y se descomprime en el sistema de archivos local si aún no existe. La variable de entorno `TFHUB_CACHE_DIR` se puede configurar para anular la ubicación temporal predeterminada para almacenar en caché los SavedModels descargados y sin comprimir. Para obtener más información, consulte [Almacenamiento en caché](caching.md).

### Usar un SavedModel en TensorFlow de bajo nivel

#### Identificadores de modelo

SavedModels se puede cargar desde un `handle` específico, donde el `handle` es una ruta del sistema de archivos, una URL válida del modelo de TFhub.dev (por ejemplo, "https://tfhub.dev/..."). Las URL de los modelos Kaggle reflejan los identificadores de TFhub.dev de acuerdo con nuestros Términos y la licencia asociada con los activos del modelo, por ejemplo, "https://www.kaggle.com/...". Los identificadores de los modelos Kaggle son equivalentes a su identificador de TFhub.dev correspondiente.

La función `hub.load(handle)` descarga y descomprime un SavedModel (a menos que `handle` ya sea una ruta del sistema de archivos) y luego devuelve el resultado al cargarlo con la función incorporada de TensorFlow `tf.saved_model.load()`. Por lo tanto, `hub.load()` puede controlar cualquier SavedModel válido (a diferencia de su predecesor `hub.Module` para TF1).

#### Tema avanzado: qué pasa con el SavedModel después de cargarlo

Según el contenido de SavedModel, el resultado de `obj = hub.load(...)` se puede invocar de varias maneras (como se explica con mucho más detalle en la [Guía de SavedModel](https://www.tensorflow.org/guide/saved_model) de TensorFlow:

- Las signaturas de servicio de SavedModel (si las hay) se representan como un diccionario de funciones concretas y se pueden llamar así `tensors_out = obj.signatures["serving_default"](**tensors_in)`, con diccionarios de tensores codificados por los respectivos nombres de entradas y salidas y sujetos a las restricciones de forma y tipo de la signatura.

- Los métodos decorados con [`@tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) del objeto guardado (si hay) se restauran como objetos tf.function que pueden llamarse mediante las combinaciones de argumentos de Tensor y argumentos que no son de Tensor para los que se [trazó](https://www.tensorflow.org/tutorials/customization/performance#tracing) la función tf.function antes de guardarse. En particular, si hay un método `obj.__call__` con los trazados adecuados, se puede llamar al propio `obj` como una función de Python. Un ejemplo simple podría verse como `output_tensor = obj(input_tensor, training=False)`.

Esto deja mucha libertad en las interfaces que SavedModels puede implementar. La [interfaz de SavedModel reutilizables](reusable_saved_models.md) para `obj` establece convenciones, por eso el código del cliente, incluso los adaptadores como `hub.KerasLayer`, sabe cómo usar SavedModel.

Es posible que algunos SavedModels no sigan esa convención, especialmente los modelos completos que no están destinados a ser reutilizados en modelos más grandes y solo proporcionan signaturas de servicio.

Las variables entrenables en un SavedModel se recargan como entrenables y `tf.GradientTape` las verá de forma predeterminada. Consulte la sección sobre ajustes a continuación para conocer algunas advertencias y tengalas en cuenta para empezar. Incluso si desea realizar un ajuste, es posible que desee ver si `obj.trainable_variables` recomienda volver a entrenar solo un subconjunto de las variables originalmente entrenables.

## Crear SavedModel para TF Hub

### Descripción general

SavedModel es el formato de serialización estándar de TensorFlow para modelos entrenados o partes de modelos. Almacena los pesos entrenados del modelo junto con las operaciones exactas de TensorFlow para realizar su cálculo. Se puede usar independientemente del código que lo creó. En particular, se puede reutilizar en diferentes API de creación de modelos de alto nivel como Keras, porque las operaciones de TensorFlow son su lenguaje básico común.

### Guardar de Keras

A partir de TensorFlow 2, `tf.keras.Model.save()` y `tf.keras.models.save_model()` tienen de forma predeterminada el formato SavedModel (no HDF5). Los SavedModels resultantes se pueden usar con `hub.load()`, `hub.KerasLayer` y adaptadores similares para otras API de alto nivel a medida que estén disponibles.

Para compartir un modelo Keras completo, simplemente guárdelo con `include_optimizer=False`.

Para compartir una parte de un modelo Keras, convierta la parte en un modelo en sí misma y luego guárdela. Puede diseñar el código así desde el principio...

```python
piece_to_share = tf.keras.Model(...)
full_model = tf.keras.Sequential([piece_to_share, ...])
full_model.fit(...)
piece_to_share.save(...)
```

...o recortar la parte para compartirla después (si se alinea con las capas de su modelo completo):

```python
full_model = tf.keras.Model(...)
sharing_input = full_model.get_layer(...).get_output_at(0)
sharing_output = full_model.get_layer(...).get_output_at(0)
piece_to_share = tf.keras.Model(sharing_input, sharing_output)
piece_to_share.save(..., include_optimizer=False)
```

[TensorFlow Models](https://github.com/tensorflow/models) en GitHub usa el primer enfoque para BERT (consulte [nlp/tools/export_tfhub_lib.py](https://github.com/tensorflow/models/blob/master/official/nlp/tools/export_tfhub_lib.py), tenga en cuenta la división entre `core_model` para exportar y el `pretrainer` para restaurar el punto de verificación) y el último enfoque para ResNet (consulte [Legacy/image_classification/tfhub_export. py](https://github.com/tensorflow/models/blob/master/official/legacy/image_classification/resnet/tfhub_export.py)).

### Guardar desde TensorFlow de bajo nivel

Esto requiere tener buen conocimiento de la [Guía de SavedModel](https://www.tensorflow.org/guide/saved_model) de TensorFlow.

Si quiere proporcionar algo más que una signatura de entrega, debe implementar la [interfaz de SavedModel reutilizable](reusable_saved_models.md). Conceptualmente, se ve así

```python
class MyMulModel(tf.train.Checkpoint):
  def __init__(self, v_init):
    super().__init__()
    self.v = tf.Variable(v_init)
    self.variables = [self.v]
    self.trainable_variables = [self.v]
    self.regularization_losses = [
        tf.function(input_signature=[])(lambda: 0.001 * self.v**2),
    ]

  @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
  def __call__(self, inputs):
    return tf.multiply(inputs, self.v)

tf.saved_model.save(MyMulModel(2.0), "/tmp/my_mul")

layer = hub.KerasLayer("/tmp/my_mul")
print(layer([10., 20.]))  # [20., 40.]
layer.trainable = True
print(layer.trainable_weights)  # [2.]
print(layer.losses)  # 0.004
```

## Ajuste

Entrenar las variables ya entrenadas de un SavedModel importado junto con las del modelo que lo rodea se denomina *ajustar* el SavedModel. Esto puede dar como resultado una mejor calidad, pero a menudo hace que el entrenamiento sea más exigente (puede llevar más tiempo, depender más del optimizador y sus hiperparámetros, aumenta el riesgo de sobreajuste y requiere un aumento del conjunto de datos, especialmente para las CNN). Aconsejamos a los consumidores de SavedModel que consideren realizar ajustes solo después de haber establecido un buen régimen de entrenamiento y solo si el editor de SavedModel lo recomienda.

El ajuste cambia los parámetros del modelo "continuo" que se entrenan. No cambia las transformaciones codificadas, como la tokenización de la entrada de texto y la asignación de tokens a sus entradas correspondientes en una matriz de incrustación.

### Para consumidores de SavedModel

Crear un `hub.KerasLayer` como

```python
layer = hub.KerasLayer(..., trainable=True)
```

permite el ajuste del SavedModel que carga la capa. Agrega los pesos entrenables y los regularizadores de peso declarados en SavedModel al modelo Keras y ejecuta el cálculo de SavedModel en modo de entrenamiento (piense en el abandono, etc.).

La [colab de clasificación de imágenes](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/tf2_image_retraining.ipynb) contiene un ejemplo de principio a fin con ajuste opcional.

#### Reexportar el resultado del ajuste

Es posible que los usuarios avanzados quieran guardar los resultados del ajuste en un SavedModel que pueda usarse en lugar del que se cargó originalmente. Esto se puede hacer con código como el siguiente

```python
loaded_obj = hub.load("https://tfhub.dev/...")
hub_layer = hub.KerasLayer(loaded_obj, trainable=True, ...)

model = keras.Sequential([..., hub_layer, ...])
model.compile(...)
model.fit(...)

export_module_dir = os.path.join(os.getcwd(), "finetuned_model_export")
tf.saved_model.save(loaded_obj, export_module_dir)
```

### Para creadores de SavedModel

Al crear un SavedModel para compartir en TensorFlow Hub, piense con anticipación si sus consumidores deberían ajustarlo y cómo, y proporcione ayuda en la documentación.

Si se guarda desde un modelo de Keras debería hacer que todos los mecanismos de ajuste funcionen (guardar pérdidas de regularización de peso, declarar variables entrenables, trazar `__call__` tanto para `training=True` como `training=False`, etc.)

Elija una interfaz de modelo que funcione bien con el flujo de gradiente, por ejemplo, logits de salida en lugar de probabilidades softmax o predicciones top-k.

Si el modelo usa abandono, normalización por lotes o técnicas de entrenamiento similares que involucran hiperparámetros, configúrelos en valores que tengan sentido en muchos problemas objetivo y tamaños de lote previstos. (En el momento de escribir este artículo, guardar desde Keras no hace que sea más fácil para los consumidores ajustarlos).

Los regularizadores de peso en capas individuales se guardan (con sus coeficientes de fuerza de regularización), pero la regularización de peso desde dentro del optimizador (como `tf.keras.optimizers.Ftrl.l1_regularization_strength=...)`) se pierde. Informe a los consumidores sobre su SavedModel como corresponde.
