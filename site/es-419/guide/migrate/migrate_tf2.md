# Visión general de migración TF1.x -&gt; TF2

TensorFlow 2 es fundamentalmente diferente de TF1.x en varios aspectos. Puedes seguir ejecutando código TF1.x sin modificar ([excepto en el caso de contrib](https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md)) en instalaciones binarias de TF2 de esta forma:

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

Sin embargo, así *no* ejecutarás comportamientos y API de TF2, y puede que no funcione como se espera con código escrito para TF2. Si de por sí no utilizas los comportamientos de TF2, en la práctica estás utilizando TF1.x sobre una instalación de TF2. Lee la guía [Comportamientos TF1 vs TF2](./tf1_vs_tf2.ipynb) para más detalles sobre las diferencias entre TF2 y TF1.x.

Esta guía te ofrece una visión general del proceso para que migres tu código TF1.x a TF2. Esto te permite aprovechar las nuevas y futuras mejoras de las funciones y también hacer que tu código sea más sencillo, eficaz y fácil de mantener.

Si usas las API de alto nivel de `tf.keras` y entrenas exclusivamente con `model.fit`, tu código debería ser más o menos totalmente compatible con TF2, salvo por las siguientes advertencias:

- TF2 tiene nuevas [tasas de aprendizaje predeterminadas](../../guide/effective_tf2.ipynb#optimizer_defaults) para los optimizadores Keras.
- TF2 [puede haber cambiado](../../guide/effective_tf2.ipynb#keras_metric_names) el "nombre" con el que se registran las métricas.

## Proceso de migración a TF2

Antes de migrar, lee la [guía](./tf1_vs_tf2.ipynb) para conocer el comportamiento y las diferencias de la API entre TF1.x y TF2.

1. Ejecuta el script automatizado para convertir parte de tu uso de API TF1.x a `tf.compat.v1`.
2. Elimina los antiguos símbolos `tf.contrib` (revisa [TF Addons](https://github.com/tensorflow/addons) y [TF-Slim](https://github.com/google-research/tf-slim)).
3. Haz que tus pasadas hacia delante del modelo TF1.x se ejecuten en TF2 con ejecución eager activada.
4. Actualiza tu código de TF1.x para utilizar bucles de entrenamiento y modelos de guardado/carga a equivalentes de TF2.
5. (Opcional) Migra tus API compatibles con TF2 `tf.compat.v1` a las API idiomáticas de TF2.

Las secciones siguientes amplían los pasos descritos anteriormente.

## Ejecutar el script de conversión de símbolos

Esto ejecuta una primera pasada en la reescritura de los símbolos de tu código para que funcione con los binarios de TF 2.x, pero no hará que tu código sea idiomático para TF 2.x ni hará que tu código sea automáticamente compatible con los comportamientos de TF2.

Lo más probable es que tu código siga usando los puntos finales `tf.compat.v1` para acceder a marcadores de posición, sesiones, colecciones y otras funciones del estilo TF1.x.

Lee la [guía](./upgrade.ipynb) para saber más sobre las mejores prácticas para usar el script de conversión de símbolos.

## Eliminar el uso de `tf.contrib`

El módulo `tf.contrib` ha desaparecido y varios de sus submódulos se han integrado en el núcleo de la API de TF2. Los demás submódulos se han separado en otros proyectos como [TF IO](https://github.com/tensorflow/io) y [TF Addons](https://www.tensorflow.org/addons/overview).

Una gran cantidad de código antiguo de TF1.x usa la librería [Slim](https://ai.googleblog.com/2016/08/tf-slim-high-level-library-to-define.html), que se empaquetó con TF1.x como `tf.contrib.layers`. Cuando migres tu código Slim a TF2, cambia tus usos de la API Slim para que apunten al paquete [tf-slim pip](https://pypi.org/project/tf-slim/). Después, lee la [guía de mapeo de modelos](https://tensorflow.org/guide/migrate/model_mapping#a_note_on_slim_and_contriblayers) para aprender a convertir el código Slim.

Otra posibilidad, si usas modelos preentrenados de Slim, es probar los modelos preentrenados de Keras desde `tf.keras.applications` o los modelos TF2 `SavedModel` de [TF Hub](https://tfhub.dev/s?tf-version=tf2&q=slim) exportados desde el código original de Slim.

## Hacer que las pasadas hacia delante del modelo TF1.x se ejecuten con los comportamientos TF2 activados

### Da seguimiento de variables y pérdidas

[TF2 no admite recolecciones globales.](./tf1_vs_tf2.ipynb#no_more_globals)

La ejecución eager en TF2 no admite APIs basadas en recolección `tf.Graph`. Y esto afecta al modo en que construyes y rastreas las variables.

Para el nuevo código TF2, usarías `tf.Variable` en lugar de `v1.get_variable` y usarías objetos Python para recopilar y dar seguimiento a las variables en lugar de `tf.compat.v1.variable_scope`. Normalmente sería uno de los siguientes:

- `tf.keras.layers.Layer`
- `tf.keras.Model`
- `tf.Module`

Agrega listas de variables (como `tf.Graph.get_collection(tf.GraphKeys.VARIABLES)`) con los atributos `.variables` y `.trainable_variables` de los objetos `Layer`, `Module` o `Model`.

Las clases `Layer` y `Model` implementan algunas otras propiedades que eliminan la necesidad de recolecciones globales. Su propiedad `.losses` puede ser un repuesto para usar la recolección `tf.GraphKeys.LOSSES`.

Lee la [guía de mapeo de modelos](./model_mapping.ipynb) para encontrar más información sobre cómo usar las plantillas de modelado de código TF2 para incrustar tu código existente basado en `get_variable` y `variable_scope` dentro de `Layers`, `Models` y `Modules`. Esto te permitirá ejecutar pasadas hacia delante con ejecución eager activada sin tener que reescribir mucho.

### Adaptar a otros cambios de comportamiento

Si la [guía de mapeo de modelos](./model_mapping.ipynb) por sí sola no es suficiente para que tu modelo haga pasadas hacia adelante ejecutando otros cambios de comportamiento que pueden ser más detallados, consulta la guía sobre [Comportamientos de TF1.x vs TF2](./tf1_vs_tf2.ipynb) para conocer los otros cambios de comportamiento y cómo puedes adaptarte a ellos. Consulta también la guía [Cómo crear nuevas Capas y Modelos mediante subclases](https://tensorflow.org/guide/keras/custom_layers_and_models.ipynb) para obtener más detalles.

### Validar tus resultados

Consulta la [guía de validación de modelos](./validate_correctness.ipynb) para conocer herramientas y orientación sencillas sobre cómo puedes validar (numéricamente) que tu modelo se comporta correctamente cuando está activada la ejecución eager. Puedes encontrarla especialmente útil si la combinas con la [guía de mapeo de modelos](./model_mapping.ipynb).

## Actualizar el entrenamiento, la evaluación y el código de importación/exportación

Los bucles de entrenamiento de TF1.x creados con `tf.estimator.Estimator` en estilo `v1.Session` y otros enfoques basados en colecciones no son compatibles con los nuevos comportamientos de TF2. Es importante que migres todo tu código de entrenamiento de TF1.x, ya que combinarlo con código de TF2 puede causar comportamientos inesperados.

Para ello, puedes elegir entre varias estrategias.

El enfoque de más alto nivel es usar `tf.keras`. Las funciones de alto nivel de Keras administran muchos de los detalles de bajo nivel que es fácil pasar por alto si escribes tu propio bucle de entrenamiento. Por ejemplo, recopilan automáticamente las pérdidas de regularización y configuran el argumento `training=True` al llamar al modelo.

Consulta la [Guía de migración de Estimator](./migrating_estimator.ipynb) para saber cómo puedes migrar el código de `tf.estimator.Estimator` para usar los bucles de entrenamiento <code>tf.keras</code> [predeterminado](./migrating_estimator.ipynb#tf2_keras_training_api) y [personalizado](./migrating_estimator.ipynb#tf2_keras_training_api_with_custom_training_step).

Los bucles de entrenamiento personalizados te proporcionan un control más preciso sobre tu modelo, como el seguimiento de las ponderaciones de capas individuales. Lee la guía sobre [construcción de bucles de entrenamiento desde cero](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch) para aprender a usar `tf.GradientTape` para recuperar las ponderaciones del modelo y usarlas para actualizarlo.

### Convertir optimizadores TF1.x en optimizadores Keras

Los optimizadores de `tf.compat.v1.train`, como el [optimizador Adam](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/AdamOptimizer) y el [optimizador de descenso gradiente](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/GradientDescentOptimizer), tienen equivalentes en `tf.keras.optimizers`.

La tabla siguiente resume cómo puedes convertir estos optimizadores heredados en sus equivalentes de Keras. Puedes reemplazar directamente la versión TF1.x por la versión TF2, a menos que se requieran pasos adicionales (como [actualizar la tasa de aprendizaje predeterminada](../../guide/effective_tf2.ipynb#optimizer_defaults)).

Ten en cuenta que convertir tus optimizadores [puede hacer incompatibles los puntos de verificación previos](./migrating_checkpoints.ipynb).

<table>
  <tr>
    <th>TF1.x</th>
    <th>TF2</th>
    <th>Pasos adicionales</th>
  </tr>
  <tr>
    <td>`tf.v1.train.GradientDescentOptimizer`</td>
    <td>`tf.keras.optimizers.SGD`</td>
    <td>Ninguno</td>
  </tr>
  <tr>
    <td>`tf.v1.train.MomentumOptimizer`</td>
    <td>`tf.keras.optimizers.SGD`</td>
    <td>Incluye el argumento "momentum"</td>
  </tr>
  <tr>
    <td>`tf.v1.train.AdamOptimizer`</td>
    <td>`tf.keras.optimizers.Adam`</td>
    <td>Cambia el nombre de los argumentos "beta1" y "beta2" a "beta_1" y "beta_2"</td>
  </tr>
  <tr>
    <td>`tf.v1.train.RMSPropOptimizer`</td>
    <td>`tf.keras.optimizers.RMSprop`</td>
    <td>Cambia el nombre del argumento "decay" por "rho"</td>
  </tr>
  <tr>
    <td>`tf.v1.train.AdadeltaOptimizer`</td>
    <td>`tf.keras.optimizers.Adadelta`</td>
    <td>Ninguno</td>
  </tr>
  <tr>
    <td>`tf.v1.train.AdagradOptimizer`</td>
    <td>`tf.keras.optimizers.Adagrad`</td>
    <td>Ninguno</td>
  </tr>
  <tr>
    <td>`tf.v1.train.FtrlOptimizer`</td>
    <td>`tf.keras.optimizers.Ftrl`</td>
    <td>Elimina los argumentos "accum_name" y "linear_name"</td>
  </tr>
  <tr>
    <td>`tf.contrib.AdamaxOptimizer`</td>
    <td>`tf.keras.optimizers.Adamax`</td>
    <td>Cambia el nombre de los argumentos "beta1" y "beta2" a "beta_1" y "beta_2"</td>
  </tr>
  <tr>
    <td>`tf.contrib.Nadam`</td>
    <td>`tf.keras.optimizers.Nadam`</td>
    <td>Cambia el nombre de los argumentos "beta1" y "beta2" a "beta_1" y "beta_2"</td>
  </tr>
</table>

Nota: En TF2, todos los épsilones (constantes numéricas de estabilidad) ahora están predeterminados a `1e-7` en lugar de `1e-8`. Esta diferencia es insignificante en la mayoría de los casos que se usan.

### Actualizar las canalizaciones de entrada de datos

Hay muchas formas de introducir datos en un modelo `tf.keras`. Aceptan generadores Python y arreglos Numpy como entrada.

La forma recomendada de introducir datos en un modelo es usar el paquete `tf.data`, que contiene una recolección de clases de alto rendimiento para manipular datos. Los `dataset` pertenecientes a `tf.data` son eficaces, expresivas y se integran bien con TF2.

Pueden pasarse directamente al método `tf.keras.Model.fit`.

```python
model.fit(dataset, epochs=5)
```

Pueden iterarse directamente sobre Python estándar:

```python
for example_batch, label_batch in dataset:
    break
```

Si sigues usando `tf.queue`, ahora sólo se admiten como estructuras de datos, no como canalizaciones de entrada.

También debes migrar todo el código de preprocesamiento de características que use `tf.feature_columns`. Lee la [guía de migración](./migrating_feature_columns.ipynb) para más detalles.

### Modelos de guardado y carga

TF2 usa puntos de verificación basados en objetos. Lee la [guía de migración de puntos de verificación](./migrating_checkpoints.ipynb) para saber más sobre la migración de puntos de verificación de TF1.x basados en nombres. Lee también la [guía de puntos de verificación](https://www.tensorflow.org/guide/checkpoint) de la documentación básica de TensorFlow.

No hay problemas significativos de compatibilidad para los modelos guardados. Lee la guía [`SavedModel`](./saved_model.ipynb) para saber más sobre la migración de los `SavedModel` de TF1.x a TF2. En general,

- saved_models de TF1.x  funcionan en TF2.
- Los saved_models de TF2 funcionan en TF1.x si todas las ops son compatibles.

Consulta también la [sección `GraphDef`](./saved_model.ipynb#graphdef_and_metagraphdef) de la guía de migración `SavedModel` para saber más sobre cómo trabajar con los objetos `Graph.pb` y `Graph.pbtxt`.

## (Opcional) Migrar para descartar símbolos `tf.compat.v1`.

El módulo `tf.compat.v1` contiene la API TF1.x completa, con su semántica original.

Incluso después de seguir los pasos anteriores y acabar con un código totalmente compatible con todos los comportamientos de TF2, es probable que haya muchas menciones a APIs `compat.v1` que resulten ser compatibles con TF2. Debes evitar usar estas APIs `compat.v1` heredadas para cualquier código nuevo que escribas, aunque seguirán funcionando para tu código ya escrito.

Sin embargo, puedes elegir migrar los usos existentes a API de TF2 no heredadas. Las docstrings de cada uno de los símbolos `compat.v1` suelen explicar cómo migrarlos a API de TF2 no heredadas. Además, la sección de la guía para mapear [modelos sobre la migración gradual a las API idiomáticas de TF2](./model_mapping.ipynb#incremental_migration_to_native_tf2) también puede ser de ayuda.

## Recursos y lecturas complementarias

Como se ha mencionado anteriormente, es una buena práctica migrar todo tu código TF1.x a TF2. Lee las guías de la sección [Migrar a TF2](https://tensorflow.org/guide/migrate) de la guía de TensorFlow para obtener más información.
