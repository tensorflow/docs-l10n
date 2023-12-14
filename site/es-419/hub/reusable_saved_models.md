# SavedModel reutilizables

## Introducción

TensorFlow Hub aloja SavedModel para TensorFlow 2, entre otros activos. Se pueden volver a cargar en un programa de Python con `obj = hub.load(url)` [[obtenga más información](tf2_saved_model)]. El `obj` que se devuelve es el resultado de `tf.saved_model.load()` (consulte [la guía de SavedModel](https://www.tensorflow.org/guide/saved_model) de TensorFlow). Este objeto puede tener atributos arbitrarios que son tf.functions, tf.Variables (que se inicializan a partir de sus valores preentrenados), otros recursos y, de forma recursiva, más objetos similares.

En esta página se describe una interfaz que implementará el `obj` cargado para que se *reutilice* en un programa de Python de TensorFlow. Los SavedModel que se ajustan a esta interfaz se denominan *SavedModel reutilizables*.

Reutilizar significa construir un modelo más grande alrededor del `obj`, incluso la capacidad de ajustarlo. Ajustar significa un entrenamiento adicional de los pesos en el `obj` cargado como parte del modelo circundante. La función de pérdida y el optimizador determina el modelo circundante; `obj` solo define la asignación de las activaciones de la entrada a la salida (el "paso siguiente"), posiblemente incluye técnicas como abandono o normalización por lotes.

**El equipo de TensorFlow Hub recomienda implementar la interfaz de SavedModel reutilizable** en todos los SavedModels que deben reutilizarse en el sentido que se menciona anteriormente. Muchas utilidades de la biblioteca `tensorflow_hub`, en particular `hub.KerasLayer`, requieren SavedModels para implementarlas.

### Relación con SignatureDefs

Esta interfaz en términos de tf.functions y otras características de TF2 es distinto de las signaturas de SavedModel, que estaban disponibles desde TF1 y continúan usándose en TF2 para inferencias (tales como implementar SavedModels en TF Serving o TF Lite). Las signaturas para inferencias no son lo suficientemente expresivas para admitir ajustes, y [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) proporciona una [API de Python](https://www.tensorflow.org/tutorials/customization/performance) más natural y expresiva para el modelo reutilizado.

### Relación con las bibliotecas de construcción de modelos

Un SavedModel reutilizable usa solo primitivos de TensorFlow 2, independientemente de cualquier biblioteca de creación de modelos en particular, como Keras o Sonnet. Esto facilita la reutilización en las bibliotecas de creación de modelos, sin dependencias del código de creación de modelos original.

Se necesitará cierto nivel de adaptación para cargar SavedModels reutilizables o para guardarlos en cualquier biblioteca de creación de modelos. Para Keras, [hub.KerasLayer](https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer) proporciona la carga, y el guardado integrado de Keras en el formato SavedModel se ha rediseñado para TF2 con el objetivo de proporcionar un superconjunto de esta interfaz (consulte el [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190509-keras-saved-model.md) de mayo de 2019).

### Relación con las "API de SavedModel comunes" específicamente de tareas

La definición de interfaz que se presenta en esta página permite cualquier número y tipo de entradas y salidas. Las [API SavedModel comunes para TF Hub](common_saved_model_apis/index.md) refinan esta interfaz general con convenciones de uso para tareas específicas para hacer que los modelos sean fácilmente intercambiables.

## Definición de interfaz

### Atributos

Un SavedModel reutilizable es un SavedModel de TensorFlow 2, tanto que `obj = tf.saved_model.load(...)` devuelve un objeto con los siguientes atributos

- `__call__`. Requerido. Una tf.function que implementa el cálculo del modelo (el "paso siguiente") sujeto a la siguiente especificación.

- `variables`: una lista de objetos tf.Variable, que enumera todas las variables que usan mediante cualquier invocación posible de `__call__`, incluidas las que se pueden entrenar y las que no se pueden entrenar.

    Esta lista se puede omitir si está vacía.

    Nota: Convenientemente, este nombre coincide con el atributo que sintetiza `tf.saved_model.load(...)` al cargar un TF1 SavedModel para representar su colección `GLOBAL_VARIABLES`.

- `trainable_variables`: una lista de objetos tf.Variable por lo que `v.trainable` es verdadero para todos los elementos. Estas variables deben ser un subconjunto de `variables`. Estas son las variables que se entrenarán al ajustar el objeto. Quien cree el SavedModel puede optar por omitir algunas variables que eran entrenables originalmente para indicar que no deben modificarse durante el ajuste.

    Esta lista se puede omitir si está vacía, en particular, si SavedModel no admite ajustes.

- `regularization_losses`: una lista de tf.functions, cada una de las cuales toma cero entradas y devuelve un único tensor flotante escalar. Para realizar ajustes, se recomienda que el usuario de SavedModel los incluya como términos de regularización adicionales en la pérdida (en el caso más simple, sin mayor escala). Normalmente, se usan para representar regularizadores de peso. (Por falta de entradas, estas tf.functions no pueden expresar regularizadores de actividad).

    Esta lista se puede omitir si está vacía, en particular, si SavedModel no admite ajustes o no desea prescribir la regularización del peso.

### La función `__call__`

Un `obj` de SavedModel restaurado tiene un atributo `obj.__call__` que es una función tf.function restaurada y permite llamar a `obj` de la siguiente manera.

Sinopsis (pseudocódigo):

```python
outputs = obj(inputs, trainable=..., **kwargs)
```

#### Argumentos

Los argumentos son los siguientes.

- Hay un argumento posicional requerido con un lote de activaciones de entrada del SavedModel. Su tipo es

    - un solo tensor para una sola entrada,
    - una lista de tensores para una secuencia ordenada de entradas sin nombre,
    - un dict de tensores codificados en un conjunto particular de nombres de entrada.

    (Las revisiones futuras de esta interfaz pueden permitir anidaciones más generales). Quien cree SavedModel elige uno y las formas y tipos de tensor. Cuando sea útil, algunas dimensiones de la forma no deben estar definidas (en particular, el tamaño del lote).

- Puede haber un `training` de argumentos de palabras clave opcional que acepte un booleano de Python, `True` o `False`. El valor predeterminado es `False`. Si el modelo admite el ajuste y si su cálculo difiere entre los dos (por ejemplo, como en el abandono y la normalización por lotes), esa distinción se implementa con este argumento. De lo contrario, este argumento puede estar ausente.

    No es necesario que `__call__` acepte un argumento `training` con valor de tensor. Depende del llamador que se use `tf.cond()` si es necesario para enlazarlos.

- Quien cree el SavedModel puede optar por aceptar más `kwargs` opcionales de nombres particulares.

    - Para los argumentos con valores de tensor, quien cree el SavedModel define sus tipos y formas permitidos. `tf.function` acepta un valor predeterminado de Python en un argumento que se rastrea con una entrada tf.TensorSpec. Se pueden usar estos argumentos para permitir la personalización de hiperparámetros numéricos involucrados en `__call__` (por ejemplo, la tasa de abandono).

    - Para los argumentos con valor de Python, quien cree el SavedModel define sus valores permitidos. Estos argumentos se pueden usar como indicadores para realizar elecciones discretas en la función trazada (pero tenga en cuenta la expansión combinatoria de trazados).

La función `__call__` restaurada debe proporcionar seguimientos para todas las combinaciones permitidas de argumentos. Cambiar `training` entre `True` y `False` no debe cambiar la permisibilidad de los argumentos.

#### Resultado

Los `outputs` de la llamada de `obj` pueden ser

- un solo tensor para una sola salida,
- una lista de tensores para una secuencia ordenada de salidas sin nombre,
- un dict de tensores codificados en un conjunto particular de nombres de salida.

(Las revisiones futuras de esta interfaz pueden permitir anidamientos más generales). El tipo de devolución puede variar según los kwargs con valor de Python. Esto permite que los indicadores produzcan salidas adicionales. Quien cree el SavedModel define los tipos y formas de salida y su dependencia de las entradas.

### Invocables con nombre

Un SavedModel reutilizable puede proporcionar varias partes del modelo como mencionamos anteriormente, al colocarlas en subobjetos con nombre, por ejemplo, `obj.foo`, `obj.bar`, etc. Cada subobjeto proporciona un método `__call__` y atributos de soporte sobre las variables, etc., específicos de esa parte del modelo. Para el ejemplo anterior, habría `obj.foo.__call__`, `obj.foo.variables`, etc.

Tenga en cuenta que esta interfaz *no* cubre el enfoque de agregar una función tf.function directamente como `tf.foo`.

Se espera que los usuarios de SavedModel reutilizables solo manejen un nivel de anidamiento (`obj.bar` pero no `obj.bar.baz`). (Las revisiones futuras de esta interfaz pueden permitir un anidamiento más profundo y quizas eliminan el requisito de que el objeto de nivel superior sea invocable).

## Comentarios finales

### Relación con las API en el proceso

En este documento se describe una interfaz de una clase de Python que consta de primitivos como tf.function y tf.Variable que sobreviven a un viaje de ida y vuelta a través de la serialización a través de `tf.saved_model.save()` y `tf.saved_model.load()`. Sin embargo, la interfaz ya estaba presente en el objeto original que se pasó a `tf.saved_model.save()`. La adaptación a esa interfaz permite el intercambio de las partes del modelo entre la API de creación de modelos dentro de un único programa de TensorFlow.
