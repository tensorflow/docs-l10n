# Migrar de TF1 a TF2 con TensorFlow Hub

En esta página se explica cómo seguir usando TensorFlow Hub mientras migras tu código de TensorFlow de TensorFlow 1 a TensorFlow 2. Funciona como complemento de la [guía general de migración](https://www.tensorflow.org/guide/migrate) de TensorFlow.

Para TF2, TF Hub se ha alejado de la API `hub.Module` heredada para crear un `tf.compat.v1.Graph` como lo hace `tf.contrib.v1.layers`. En su lugar, ahora hay un `hub.KerasLayer` para usar junto con otras capas de Keras para construir un `tf.keras.Model` (normalmente en el nuevo [entorno de ejecución eager](https://www.tensorflow.org/api_docs/python/tf/executing_eagerly) de TF2) y su método `hub.load()` subyacente para el código de TensorFlow de bajo nivel.

La API `hub.Module` aún está disponible en la biblioteca `tensorflow_hub` para su uso en TF1 y en el modo de compatibilidad TF1 de TF2. Solo puede cargar modelos en [formato TF1 Hub](tf1_hub_module.md).

La nueva API de `hub.load()` y `hub.KerasLayer` funciona para TensorFlow 1.15 (en modo eager y graph) y en TensorFlow 2. Esta API nueva puede cargar los nuevos activos [TF2 SavedModel](tf2_saved_model.md) y, con las restricciones establecidas la [guía de compatibilidad de modelos](model_compatibility.md), los modelos heredados en formato TF1 Hub.

En general, se recomienda usar una API nueva siempre que sea posible.

## Resumen de la API nueva

`hub.load()` es la nueva función de bajo nivel para cargar un SavedModel desde TensorFlow Hub (o servicios compatibles). Contiene `tf.saved_model.load()` de TF2; en la [guía de SavedModel](https://www.tensorflow.org/guide/saved_model) de TensorFlow se describe lo que puede hacer con el resultado.

```python
m = hub.load(handle)
outputs = m(inputs)
```

La clase `hub.KerasLayer` llama a `hub.load()` y adapta el resultado para usarlo en Keras junto con otras capas de Keras. (Incluso puede ser un contenedor conveniente de SavedModels que se cargaron y que se usan de otras maneras).

```python
model = tf.keras.Sequential([
    hub.KerasLayer(handle),
    ...])
```

Muchos tutoriales muestran estas API en acción. Consulte los siguientes

- [Cuaderno de ejemplo de clasificación de texto](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/tf2_text_classification.ipynb)
- [Cuaderno de ejemplo de clasificación de imágenes](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/tf2_image_retraining.ipynb)

### Usar API nueva en el entrenamiento de Estimator

Si usa un TF2 SavedModel en un Estimador para entrenar con servidores de parámetros (o de otro modo en una sesión TF1 con variables que están en dispositivos remotos), debe configurar `experimental.share_cluster_devices_in_session` en el ConfigProto de tf.Session, o de lo contrario obtendrá un error como "Assigned device '/job:ps/replica:0/task:0/device:CPU:0' does not match any device" (El dispositivo asignado '/job:ps/replica:0/task:0/device:CPU:0' no coincide con ningún dispositivo).

La opción necesaria se puede configurar así

```python
session_config = tf.compat.v1.ConfigProto()
session_config.experimental.share_cluster_devices_in_session = True
run_config = tf.estimator.RunConfig(..., session_config=session_config)
estimator = tf.estimator.Estimator(..., config=run_config)
```

A partir de TF2.2, esta opción ya no es experimental y se puede eliminar la parte de `.experimental`.

## Cargar modelos heredados en formato TF1 Hub

Es posible que aún no esté disponible un TF2 SavedModel nuevo para su caso de uso que y necesite cargar un modelo heredado en formato TF1 Hub. A partir de la versión 0.7 `tensorflow_hub`, puede usar el modelo heredado en formato TF1 Hub junto con `hub.KerasLayer` como se muestra a continuación:

```python
m = hub.KerasLayer(handle)
tensor_out = m(tensor_in)
```

Además, `KerasLayer` expone la capacidad de especificar `tags`, `signature`, `output_key` y `signature_outputs_as_dict` para usos más específicos de modelos heredados en formato TF1 Hub y SavedModels heredados.

Para obtener más información sobre la compatibilidad del formato TF1 Hub, consulte la [guía de compatibilidad de modelos](model_compatibility.md).

## Usar unas API de nivel inferior

Los modelos de formato Legacy TF1 Hub se pueden cargar a través de `tf.saved_model.load`. En lugar de

```python
# DEPRECATED: TensorFlow 1
m = hub.Module(handle, tags={"foo", "bar"})
tensors_out_dict = m(dict(x1=..., x2=...), signature="sig", as_dict=True)
```

se recomienda usar:

```python
# TensorFlow 2
m = hub.load(path, tags={"foo", "bar"})
tensors_out_dict = m.signatures["sig"](x1=..., x2=...)
```

En estos ejemplos, `m.signatures` es un dict de [funciones concretas](https://www.tensorflow.org/tutorials/customization/performance#tracing) de TensorFlow codificadas por nombres de signatura. Al llamar a dicha función se calculan todas sus salidas, incluso si no se usaron. (Esto es diferente a la evaluación diferida del modo gráfico de TF1).
