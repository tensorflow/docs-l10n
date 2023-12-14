# Formato TF1 Hub

En su lanzamiento en 2018, TensorFlow Hub ofrecía un único tipo de activo: el formato TF1 Hub para importar a programas de TensorFlow 1.

En esta página se explica cómo usar el formato TF1 Hub en TF1 (o el modo de compatibilidad TF1 de TF2) con la clase `hub.Module` y las API asociadas. (El uso típico es construir un `tf.Graph`, posiblemente dentro de un TF1 `Estimator`, combinando uno o más modelos en formato TF1 Hub con `tf.compat.layers` o `tf.layers`).

Los usuarios de TensorFlow 2 (fuera del modo de compatibilidad TF1) deben usar [la API nueva con `hub.load()` o `hub.KerasLayer`](tf2_saved_model.md). La API nueva carga el nuevo tipo de activo TF2 SavedModel, pero también tiene [compatibilidad limitada para cargar el formato TF1 Hub en TF2](migration_tf2.md).

## Usar un modelo en formato TF1 Hub

### Creación de instancias de un modelo en formato TF1 Hub

Un modelo en formato TF1 Hub se importa a un programa de TensorFlow y se crea un objeto `hub.Module` a partir de una cadena de texto con su URL o ruta del sistema de archivos, como por ejemplo:

```python
m = hub.Module("path/to/a/module_dir")
```

**Nota:** Puede encontrar más información sobre otros tipos de identificadores válidos [aquí](tf2_saved_model.md#model_handles).

Así se agregan las variables del módulo al gráfico actual de TensorFlow. Al ejecutar sus inicializadores se leerán sus valores preentrenados del disco. De la misma forma, se agregan tablas y otros estados al gráfico.

### Almacenar módulos en caché

Al crear un módulo a partir de una URL, el contenido del módulo se descarga y se almacena en caché en el directorio temporal del sistema local. La ubicación donde se almacenan en caché los módulos se puede anular con la variable de entorno `TFHUB_CACHE_DIR`. Para obtener más información, consulte [Almacenamiento en caché](caching.md).

### Aplicar un módulo

Una vez que se crea una instancia, se puede llamar a un módulo `m` cero o más veces como una función de Python desde entradas de tensores a salidas de tensores:

```python
y = m(x)
```

Cada una de estas llamadas agrega operaciones al gráfico actual de TensorFlow para calcular `y` a partir de `x`. Si esto involucra variables con pesos entrenados, estos se comparten entre todas las aplicaciones.

Los módulos pueden definir varias *signaturas* con nombre para permitir su aplicación en más de una forma (similar a cómo los objetos Python tienen *métodos*). La documentación de un módulo debe describir las signaturas disponibles. La llamada anterior aplica la signatura denominada `"default"`. Se puede seleccionar cualquier signatura al pasar su nombre al argumento opcional `signature=`.

Si una signatura tiene varias entradas, se deben pasar como un dict, con las claves que define la signatura. Del mismo modo, si una signatura tiene varias salidas, estas se pueden recuperar como un dict pasando `as_dict=True`, bajo las claves que define la signatura (la clave `"default"` es para la única salida que se devuelve si `as_dict=False`). Entonces, la forma más general de aplicar un módulo es la siguiente:

```python
outputs = m(dict(apples=x1, oranges=x2), signature="fruit_to_pet", as_dict=True)
y1 = outputs["cats"]
y2 = outputs["dogs"]
```

Un llamador debe proporcionar todas las entradas que define una signatura, pero no es necesario usar todas las salidas de un módulo. TensorFlow ejecutará solo aquellas partes del módulo que terminen como dependencias de un destino en `tf.Session.run()`. De hecho, los editores de módulos pueden optar por proporcionar varias salidas para usos avanzados (como activaciones de capas intermedias) junto con las salidas principales. Los consumidores de módulos deben controlar las salidas adicionales con elegancia.

### Probar módulos alternativos

Siempre que haya varios módulos para la misma tarea, TensorFlow Hub recomienda equiparlos con signaturas (interfaces) compatibles, para que probar diferentes módulos sea tan fácil como variar el identificador del módulo como un hiperparámetro con valor de cadena de texto.

Con este fin, mantenemos una colección de [signaturas comunes](common_signatures/index.md) recomendadas para tareas populares.

## Crear un módulo nuevo

### Nota de compatibilidad

El formato TF1 Hub está orientado a TensorFlow 1. TF Hub solo lo admite parcialmente en TensorFlow 2. Considere publicar en el nuevo formato [TF2 SavedModel](tf2_saved_model.md).

El formato TF1 Hub es similar al formato SavedModel de TensorFlow 1 en un nivel sintáctico (los mismos nombres de archivo y mensajes de protocolo) pero semánticamente diferente para permitir la reutilización, composición y reentrenamiento del módulo (por ejemplo, almacenamiento diferente de inicializadores de recursos, convenciones de etiquetado diferentes para los metagráficos). La forma más fácil de diferenciarlos en el disco es la presencia o ausencia del archivo `tfhub_module.pb`.

### Enfoque general

Para definir un módulo nuevo, un editor llama a `hub.create_module_spec()` con una función `module_fn`. Esta función construye un gráfico que representa la estructura interna del módulo y usa `tf.placeholder()` para las entradas que debe proporcionar la persona que llama. Luego define signaturas al llamar a `hub.add_signature(name, inputs, outputs)` una o más veces.

Por ejemplo:

```python
def module_fn():
  inputs = tf.placeholder(dtype=tf.float32, shape=[None, 50])
  layer1 = tf.layers.dense(inputs, 200)
  layer2 = tf.layers.dense(layer1, 100)
  outputs = dict(default=layer2, hidden_activations=layer1)
  # Add default signature.
  hub.add_signature(inputs=inputs, outputs=outputs)

...
spec = hub.create_module_spec(module_fn)
```

El resultado de `hub.create_module_spec()` se puede usar, en lugar de una ruta, para crear una instancia de un objeto de módulo dentro de un gráfico de TensorFlow particular. En tal caso, no hay ningún punto de verificación y la instancia del módulo usará los inicializadores de variables en su lugar.

Cualquier instancia de módulo se puede serializar en el disco mediante su método `export(path, session)`. Al exportar un módulo, se serializa su definición junto con el estado actual de sus variables en `session` en la ruta que se pasa. Esto se puede usar al exportar un módulo por primera vez, así como al exportar un módulo con ajustes.

Para que sea compatible con TensorFlow Estimators, `hub.LatestModuleExporter` exporta los módulos desde el último punto de verificación, de la misma manera que `tf.estimator.LatestExporter` exporta el modelo completo desde el último punto de verificación.

Los editores de módulos deben implementar una [signatura común](common_signatures/index.md) cuando sea posible, para que los consumidores puedan intercambiar módulos fácilmente y encontrar el mejor para su problema.

### Ejemplo real

Eche un vistazo a nuestro [exportador de módulos de incrustación de texto](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings/export.py) para ver un ejemplo real de cómo crear un módulo a partir de un formato de incrustación de texto común.

## Ajuste

Entrenar las variables de un módulo importado junto con las del modelo que lo envuelve se llama *ajuste*. Un ajuste puede dar lugar a una mejor calidad, pero añade complicaciones nuevas. Aconsejamos a los consumidores que consideren realizar ajustes solo después de explorar los arreglos de calidad más simples y solo si el editor del módulo lo recomienda.

### Para consumidores

Para habilitar el ajuste, cree una instancia del módulo con `hub.Module(..., trainable=True)` para que sus variables sean entrenables e importe `REGULARIZATION_LOSSES` de TensorFlow. Si el módulo tiene varias variantes de gráficos, asegúrese de elegir la adecuada para el entrenamiento. Por lo general, es la que tiene etiquetas `{"train"}`.

Elija un régimen de entrenamiento que no arruine los pesos preentrenados, por ejemplo, una tasa de aprendizaje menor que un entrenamiento desde cero.

### Para editores

Para facilitar el ajuste para los consumidores, tenga en cuenta lo siguiente:

- El ajuste necesita regularización. Su módulo se exporta con la colección `REGULARIZATION_LOSSES`, que es lo que coloca su elección de `tf.layers.dense(..., kernel_regularizer=...)` etc. en lo que el consumidor obtiene de `tf.losses.get_regularization_losses()`. Esta forma de definir las pérdidas de regularización L1/L2 es preferible.

- En el modelo de editor, evite definir la regularización L1/L2 mediante los parámetros `l1_` y `l2_regularization_strength` de `tf.train.FtrlOptimizer`, `tf.train.ProximalGradientDescentOptimizer` y otros optimizadores proximales. Estos no se exportan junto con el módulo, y establecer niveles de regularización globalmente puede no ser adecuadi para el consumidor. Excepto por la regularización L1 en modelos amplios (es decir, lineales dispersos) o amplios y profundos, debería ser posible usar pérdidas de regularización individuales en su lugar.

- Si usa abandono, normalización por lotes o técnicas de entrenamiento similares, establezca sus hiperparámetros en valores que tengan sentido en muchos usos esperados. Es posible que la tasa de abandono deba ajustarse a la predisposición del problema objetivo al sobreajuste. En la normalización por lotes, el impulso (también conocido como coeficiente de caída) debe ser lo suficientemente pequeño como para permitir un ajuste con conjuntos de datos pequeños o lotes grandes. Para los consumidores avanzados, considere agregar una signatura que exponga el control sobre los hiperparámetros críticos.
