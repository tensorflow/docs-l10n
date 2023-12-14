# API de SavedModel comunes para tareas de imágenes

En esta página se describe cómo [TF2 SavedModels](../tf2_saved_model.md) para tareas relacionadas con imágenes debe implementar la [API de SavedModel reutilizable](../reusable_saved_models.md). (Esto reemplaza las [signaturas comunes para imágenes](../common_signatures/images.md) para el [formato TF1 Hub](../tf1_hub_module) ahora obsoleto).

<a name="feature-vector"></a>

## Vector de características de imágenes

### Resumen de uso

Un **vector de características de imágenes** es un tensor unidimensional denso que representa una imagen completa, generalmente para que lo use un clasificador de retroalimentación simple en el modelo de consumidor. (En términos de las CNN clásicas, este es el valor del cuello de botella después de que la extensión espacial se haya agrupado o aplanado, pero antes de que se realice la clasificación; para eso, consulte [clasificación de imágenes](#classification) a continuación).

Un SavedModel reutilizable para la extracción de características de imágenes tiene un método `__call__` en el objeto raíz que asigna un lote de imágenes a un lote de vectores de características. Se puede usar de la siguiente manera:

```python
obj = hub.load("path/to/model")  # That's tf.saved_model.load() after download.
images = ...  # A batch of images with shape [batch_size, height, width, 3].
features = obj(images)   # A batch with shape [batch_size, num_features].
```

En Keras, el equivalente es

```python
features = hub.KerasLayer("path/to/model")(images)
```

La entrada sigue la convención general para [la entrada de imágenes](#input). La documentación del modelo especifica el rango permitido para `height` y `width` de la entrada.

La salida es un tensor único de dtype `float32` y de forma `[batch_size, num_features]`. El `batch_size` es el mismo que en la entrada. `num_features` es una constante específica del módulo independiente del tamaño de entrada.

### Detalles de la API

La [API SavedModel reutilizable](../reusable_saved_models.md) también proporciona una lista de `obj.variables` (por ejemplo, para la inicialización cuando no se carga en eager).

Un modelo que admite ajustes proporciona una lista de `obj.trainable_variables`. Es posible que se deba aprobar `training=True` para ejecutarlo en modo de entrenamiento (por ejemplo, para abandono). Algunos modelos permiten argumentos opcionales para anular hiperparámetros (por ejemplo, tasa de abandono; se describirá en la documentación del modelo). El modelo también puede proporcionar una lista de `obj.regularization_losses`. Para obtener más información, consulte [API SavedModel reutilizable](../reusable_saved_models.md).

En Keras, `hub.KerasLayer` se encarga de esto: inicialícelo con `trainable=True` para permitir el ajuste y (en el caso insual de que se apliquen anulaciones de hparam) con `arguments=dict(some_hparam=some_value, ...))`.

### Notas

La aplicación del abandono en las características de salida (o no) debe dejarse en manos del consumidor del modelo. El SavedModel en sí no debería realizar el abandono en las salidas reales (incluso si se usa el abandono internamente en otros lugares).

### Ejemplos

Los SavedModel reutilizables para vectores de características de imágenes se usan en

- el tutorial de Colab [Reentrenamiento de un clasificador de imágenes](https://colab.research.google.com/github/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/tf2_image_retraining.ipynb),

<a name="classification"></a>

## Clasificación de imágenes

### Resumen de uso

**La clasificación de imágenes** asigna los píxeles de una imagen a puntuaciones lineales (logits) para pertenecer a las clases de una taxonomía *que selecciona el editor del módulo*. Esto le permite a los consumidores del modelo que saquen conclusiones de la clasificación particular que aprende el módulo del editor. (Para la clasificación de imágenes con un nuevo conjunto de clases, es común reutilizar un modelo de [vector de características de imágenes](#feature-vector) con un clasificador nuevo).

Un SavedModel reutilizable para la clasificación de imágenes tiene un método `__call__` en el objeto raíz que asigna un lote de imágenes a un lote de logits. Se puede usar de la siguiente manera:

```python
obj = hub.load("path/to/model")  # That's tf.saved_model.load() after download.
images = ...  # A batch of images with shape [batch_size, height, width, 3].
logits = obj(images)   # A batch with shape [batch_size, num_classes].
```

En Keras, el equivalente es

```python
logits = hub.KerasLayer("path/to/model")(images)
```

La entrada sigue la convención general para [la entrada de imágenes](#input). La documentación del modelo especifica el rango permitido para `height` y `width` de la entrada.

Los `logits` de salida son un tensor único de dtype `float32` y de forma `[batch_size, num_classes]`. El `batch_size` es el mismo que en la entrada. `num_classes` es el número de clases en la clasificación, que es una constante específica del modelo.

El valor `logits[i, c]` es una puntuación que predice la pertenencia del ejemplo `i` a la clase con índice `c`.

Dependerá de la clasificación de base, que esta puntuación se use o no con softmax (para clases mutuamente excluyentes), sigmoid (para clases ortogonales) o alguna otra opción. Se debería describir en la documentación del módulo. Además, debería haber una referencia a una definición de los índices de las clases.

### Detalles de la API

La [API SavedModel reutilizable](../reusable_saved_models.md) también proporciona una lista de `obj.variables` (por ejemplo, para la inicialización cuando no se carga en eager).

Un modelo que admite ajustes proporciona una lista de `obj.trainable_variables`. Es posible que se deba aprobar `training=True` para ejecutarlo en modo de entrenamiento (por ejemplo, para abandono). Algunos modelos permiten argumentos opcionales para anular hiperparámetros (por ejemplo, tasa de abandono; se describirá en la documentación del modelo). El modelo también puede proporcionar una lista de `obj.regularization_losses`. Para obtener más información, consulte [API SavedModel reutilizable](../reusable_saved_models.md).

En Keras, `hub.KerasLayer` se encarga de esto: inicialícelo con `trainable=True` para permitir el ajuste y (en el caso insual de que se apliquen anulaciones de hparam) con `arguments=dict(some_hparam=some_value, ...))`.

<a name="input"></a>

## Entrada de imagen

Esto es común a todos los tipos de modelos de imágenes.

Un modelo que toma un lote de imágenes como entrada las acepta como un tensor de 4 dimensiones denso de dtype `float32` y de forma `[batch_size, height, width, 3]` cuyos elementos son valores de color RGB de pixeles normalizados para el rango [0, 1]. Es lo que se obtiene a partir de `tf.image.decode_*()` seguido por `tf.image.convert_image_dtype(..., tf.float32)`.

El modelo acepta cualquier `batch_size`. La documentación del modelo especifica el rango permitido de `height` y `width`. La última dimensión está fijada a 3 canales RGB.

Se recomienda que los modelos usen el diseño de tensores `channels_last` (o `NHWC`) en todo momento y que dejen que el optimizador de gráficos de TensorFlow lo reescriba en `channels_first` (o `NCHW`) si es necesario.
