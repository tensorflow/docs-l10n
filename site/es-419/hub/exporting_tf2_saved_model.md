# Exportar un SavedModel

En esta página se describen los detalles de cómo exportar (guardar) un modelo desde un programa de TensorFlow en [formato SavedModel de TensorFlow 2](https://www.tensorflow.org/guide/saved_model). Este formato es la forma recomendada de compartir modelos y partes de modelo preentrenados en TensorFlow Hub. Reemplaza el antiguo [formato TF1 Hub](tf1_hub_module.md) y viene con un conjunto nuevo de APIs. Puede encontrar más información sobre cómo exportar los modelos en formato TF1 Hub en [Exportación en formato TF1 Hub](exporting_hub_format.md). Puede encontrar detalles sobre cómo comprimir SavedModel para compartirlo en TensorFlow Hub [aquí](writing_documentation.md#model-specific_asset_content).

Algunos conjuntos de herramientas de creación de modelos ya proporcionan herramientas para hacer esto (por ejemplo, consulte [TensorFlow Model Garden](#tensorflow-model-garden) a continuación).

## Descripción general

SavedModel es el formato de serialización estándar de TensorFlow para modelos entrenados o partes de modelos. Almacena los pesos entrenados del modelo junto con las operaciones exactas de TensorFlow para realizar su cálculo. Se puede usar independientemente del código que lo creó. En particular, se puede reutilizar en diferentes API de creación de modelos de alto nivel como Keras, porque las operaciones de TensorFlow son su lenguaje básico común.

## Guardar desde Keras

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

## Guardar desde TensorFlow de bajo nivel

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

## Consejos para creadores de SavedModel

Al crear un SavedModel para compartir en TensorFlow Hub, piense con anticipación si sus consumidores deberían ajustarlo y cómo, y proporcione ayuda en la documentación.

Si se guarda desde un modelo de Keras debería hacer que todos los mecanismos de ajuste funcionen (guardar pérdidas de regularización de peso, declarar variables entrenables, trazar `__call__` tanto para `training=True` como `training=False`, etc.)

Elija una interfaz de modelo que funcione bien con el flujo de gradiente, por ejemplo, logits de salida en lugar de probabilidades softmax o predicciones top-k.

Si el modelo usa abandono, normalización por lotes o técnicas de entrenamiento similares que involucran hiperparámetros, configúrelos en valores que tengan sentido en muchos problemas objetivo y tamaños de lote previstos. (En el momento de escribir este artículo, guardar desde Keras no hace que sea más fácil para los consumidores ajustarlos).

Los regularizadores de peso en capas individuales se guardan (con sus coeficientes de fuerza de regularización), pero la regularización de peso desde dentro del optimizador (como `tf.keras.optimizers.Ftrl.l1_regularization_strength=...)`) se pierde. Informe a los consumidores sobre su SavedModel como corresponde.

<a name="tensorflow-model-garden"></a>

## TensorFlow Model Garden

El repositorio de [TensorFlow Model Garden](https://github.com/tensorflow/models/tree/master/research/official) contiene muchos ejemplos de creación de SavedModels TF2 reutilizables para cargar en [tfhub.dev](https://tfhub.dev/).

## Solicitudes de la comunidad

El equipo de TensorFlow Hub genera solo una pequeña fracción de los activos que están disponibles en tfhub.dev. Dependemos principalmente de investigadores de Google y DeepMind, instituciones de investigación académicas y corporativas y entusiastas del aprendizaje automático para producir modelos. Como resultado, no podemos garantizar que podamos cumplir con las solicitudes de la comunidad para activos específicos y no podemos proporcionar estimaciones de tiempo para la disponibilidad de activos nuevos.

A continuación, el [hito de solicitudes de modelo de la comunidad](https://github.com/tensorflow/hub/milestone/1) contiene las solicitudes de la comunidad de activos específicos. Si a usted o a alguien que conoce le interesa producir el activo y compartirlo en tfhub.dev, le agradecemos su contribución.
