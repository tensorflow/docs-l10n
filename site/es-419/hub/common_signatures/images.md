# Firmas comunes para imágenes

En esta página se describen las firmas comunes que se deberían implementar por módulos en el [formato TF1 Hub](../tf1_hub_module.md) para tareas relacionadas con imágenes. (Para el [formato TF2 SavedModel](../tf2_saved_model.md), consulte la [API SavedModel](../common_saved_model_apis/images.md) análoga).

Algunos módulos se pueden usar para más de una tarea (p. ej., los módulos para clasificación de imágenes tienden a hacer algunas extracciones de características en el proceso). Por lo tanto, cada módulo proporciona (1) firmas con nombre para todas las tareas anticipadas por el publicador y (2) una firma predeterminada `output = m(images)` para la tarea principal designada.

<a name="feature-vector"></a>

## Vector de características de imagen

### Resumen de uso

Un **vector de características de imagen** es un tensor D denso que representa una imagen completa, por lo común para la clasificación de un modelo consumidor. (A diferencia de lo que sucede con las activaciones intermediarias de las CNN, en este caso no se ofrece un desglosamiento espacial. Por el contrario, también, ahora de lo que ofrece la [clasificación de imágenes](#classification), aquí se descarta la clasificación aprendida por quien publicó el modelo).

Un módulo para extracción de características de una imagen tiene una firma predeterminada que mapea un lote de imágenes con un lote de vectores de características. Se puede usar del siguiente modo:

```python
  module_spec = hub.load_module_spec("path/to/module")
  height, width = hub.get_expected_image_size(module_spec)
  images = ...  # A batch of images with shape [batch_size, height, width, 3].
  module = hub.Module(module_spec)
  features = module(images)   # A batch with shape [batch_size, num_features].
```

También define la firma de nombre correspondiente.

### Especificaciones de la firma

La firma con nombre para extraer los vectores de características de una imagen se invoca de la siguiente manera:

```python
  outputs = module(dict(images=images), signature="image_feature_vector",
                   as_dict=True)
  features = outputs["default"]
```

La entrada sigue la convención general para [entrada de imágenes](#input).

El diccionario de salidas contiene una salida `"default"` de `float32` tipo d y forma `[batch_size, num_features]`. El `batch_size` es igual al de la entrada, pero no se conoce en el momento de la construcción del grafo. `num_features` es una constante específica del módulo y conocida, que además, es independiente del tamaño de la entrada.

Estos vectores de características fueron previstos para ser usados para realizar la clasificación con un clasificador simple prealimentado (como las características agrupadas de la capa convolucional más alta en una CNN típica para clasificación de imágenes).

La aplicación de la omisión aleatoria (<em>dropout</em>) a las características de salida debería hacerla el consumidor del módulo. El módulo mismo no debería realizar dicha omisión en las salidas reales (incluso, aunque lo use internamente en otros lugares).

El diccionario de salidas puede proporcionar otras salidas más. Por ejemplo, las activaciones de capas ocultas dentro del módulo. Sus claves y valores son dependientes del módulo. Se recomienda prefijar las claves de arquitectura dependiente con un nombre de arquitectura (para evitar confundir la capa intermedia `"InceptionV3/Mixed_5c"` con la capa convolucional más alta `"InceptionV2/Mixed_5c"`).

<a name="classification"></a>

## Clasificación de imágenes

### Resumen de uso

La **clasificación de imágenes** mapea los pixeles de una imagen con puntajes lineales (funciones logit) para miembros de las clases de una taxonomía *seleccionada por el publicador del módulo*. De este modo, los consumidores pueden sacar conclusiones a partir de la clasificación particular aprendida por el módulo publicador y no solamente sus funciones base (compare con el [vector de características de una imagen](#feature-vector)).

Un módulo para extracción de características de una imagen tiene una firma predeterminada que mapea un lote de imágenes con un lote de funciones logit. Se puede usar del siguiente modo:

```python
  module_spec = hub.load_module_spec("path/to/module")
  height, width = hub.get_expected_image_size(module_spec)
  images = ...  # A batch of images with shape [batch_size, height, width, 3].
  module = hub.Module(module_spec)
  logits = module(images)   # A batch with shape [batch_size, num_classes].
```

También define la firma de nombre correspondiente.

### Especificaciones de la firma

La firma con nombre para extraer los vectores de características de una imagen se invoca de la siguiente manera:

```python
  outputs = module(dict(images=images), signature="image_classification",
                   as_dict=True)
  logits = outputs["default"]
```

La entrada sigue la convención general para [entrada de imágenes](#input).

El diccionario de salidas contiene una salida `"default"` de `float32` tipo d y forma `[batch_size, num_classes]`. El `batch_size` es igual al de la entrada, pero no se conoce en el momento de la construcción del grafo. `num_classes` es la cantidad de clases de clasificación, que es una constante conocida independiente del tamaño de la entrada.

Al evaluar `outputs["default"][i, c]` se produce un puntaje que predice la membresía del ejemplo `i` en la clase con el índice `c`.

Dependerá de la clasificación de base, que estos puntajes se usen o no con softmax (para clases mutuamente excluyentes), sigmoid (para clases ortogonales) o alguna otra opción. En la documentación del módulo debería estar descrito. Además, debería haber una referencia a una definición de los índices de las clases.

El diccionario de salidas puede proporcionar otras salidas más. Por ejemplo, las activaciones de capas ocultas dentro del módulo. Sus claves y valores son dependientes del módulo. Se recomienda prefijar las claves de arquitectura dependiente con un nombre de arquitectura (para evitar confundir la capa intermedia `"InceptionV3/Mixed_5c"` con la capa convolucional más alta `"InceptionV2/Mixed_5c"`).

<a name="input"></a>

## Entrada de imagen

Es común para todos los tipos de módulos y firmas de imágenes.

Una firma que toma un lote de imágenes como entrada las acepta como un tensor 4-D denso de tipo d `float32` y forma `[batch_size, height, width, 3]` cuyos elementos son valores de color RGB de pixeles normalizados para el rango [0, 1]. Es lo que se obtiene a partir de `tf.image.decode_*()` seguido por `tf.image.convert_image_dtype(..., tf.float32)`.

Un módulo con exactamente una entrada (o una principal) de imágenes usa el nombre `"images"` para esta entrada.

El módulo acepta cualquier `batch_size` y, en consecuencia, define la primera dimensión de TensorInfo.tensor_shape como "desconocida". La última dimensión está determinada por el número `3` de canales RGB. Las dimensiones de `height` y `width` se determinan por el tamaño esperado de las imágenes de entrada. (Probablemente, más adelante será posible quitar la restricción para módulos completamente convolucionales).

Los consumidores del módulo no deberían inspeccionar directamente el tamaño, sino obtener la información sobre el tamaño llamando a hub.get_expected_image_size() en el módulo o en sus especificaciones. También se espera que las imágenes de entrada cambien su tamaño en consecuencia (normalmente antes o durante la agrupación en lotes).

Para hacerlo más sencillo, los módulos de TF-Hub usan el diseño `channels_last` (o `NHWC`) de tensores y, en caso de ser necesaria, dejan que el optimizador de grafos de TensorFlow se ocupe de la reescritura para `channels_first` (o `NCHW`). Esto se ha estado haciendo de este modo de forma predeterminada desde la versión 1.7 de TensorFlow.
