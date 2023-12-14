# Exportar modelos en formato TF1 Hub

Puede leer más sobre este formato en [formato TF1 Hub](tf1_hub_module.md)

## Nota de compatibilidad

El formato TF1 Hub está orientado a TensorFlow 1. TF Hub solo lo admite parcialmente en TensorFlow 2. Considere publicar en el nuevo formato [TF2 SavedModel](tf2_saved_model.md) siguiendo la guía de [Exportar un modelo](exporting_tf2_saved_model).

El formato TF1 Hub es similar al formato SavedModel de TensorFlow 1 en un nivel sintáctico (los mismos nombres de archivo y mensajes de protocolo) pero semánticamente diferente para permitir la reutilización, composición y reentrenamiento del módulo (por ejemplo, almacenamiento diferente de inicializadores de recursos, convenciones de etiquetado diferentes para los metagráficos). La forma más fácil de diferenciarlos en el disco es la presencia o ausencia del archivo `tfhub_module.pb`.

## Enfoque general

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

## Ejemplo real

Eche un vistazo a nuestro [exportador de módulos de incrustación de texto](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings/export.py) para ver un ejemplo real de cómo crear un módulo a partir de un formato de incrustación de texto común.

## Consejos para editores

Para facilitar el ajuste para los consumidores, tenga en cuenta lo siguiente:

- El ajuste necesita regularización. Su módulo se exporta con la colección `REGULARIZATION_LOSSES`, que es lo que coloca su elección de `tf.layers.dense(..., kernel_regularizer=...)` etc. en lo que el consumidor obtiene de `tf.losses.get_regularization_losses()`. Esta forma de definir las pérdidas de regularización L1/L2 es preferible.

- En el modelo de editor, evite definir la regularización L1/L2 mediante los parámetros `l1_` y `l2_regularization_strength` de `tf.train.FtrlOptimizer`, `tf.train.ProximalGradientDescentOptimizer` y otros optimizadores proximales. Estos no se exportan junto con el módulo, y establecer niveles de regularización globalmente puede no ser adecuadi para el consumidor. Excepto por la regularización L1 en modelos amplios (es decir, lineales dispersos) o amplios y profundos, debería ser posible usar pérdidas de regularización individuales en su lugar.

- Si usa abandono, normalización por lotes o técnicas de entrenamiento similares, establezca sus hiperparámetros en valores que tengan sentido en muchos usos esperados. Es posible que la tasa de abandono deba ajustarse a la predisposición del problema objetivo al sobreajuste. En la normalización por lotes, el impulso (también conocido como coeficiente de caída) debe ser lo suficientemente pequeño como para permitir un ajuste con conjuntos de datos pequeños o lotes grandes. Para los consumidores avanzados, considere agregar una signatura que exponga el control sobre los hiperparámetros críticos.
