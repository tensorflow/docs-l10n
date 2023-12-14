# Problemas comunes

Si no encuentra su problema en esta lista, búsquelo en los [problemas de github](https://github.com/tensorflow/hub/issues) antes de presentar uno nuevo.

**Nota:** En esta documentación se usan identificadores del URL de TFhub.dev en los ejemplos. Puede ver más información sobre otros tipos de identificadores válidos [aquí](tf2_saved_model.md#model_handles).

## TypeError: no se puede llamar al objeto 'AutoTrackable'

```python
# BAD: Raises error
embed = hub.load('https://tfhub.dev/google/nnlm-en-dim128/1')
embed(['my text', 'batch'])
```

Este error suele ocurrir al cargar modelos en formato TF1 Hub con la API `hub.load()` en TF2. Agregar la signatura correcta debería solucionar este problema. Consulte la [guía de migración de TF-Hub para TF2](migration_tf2.md) para obtener más detalles sobre cómo pasar a TF2 y el uso de los modelos en formato TF1 Hub en TF2.

```python

embed = hub.load('https://tfhub.dev/google/nnlm-en-dim128/1')
embed.signatures['default'](['my text', 'batch'])
```

## No se puede descargar un módulo

En el proceso de usar un módulo desde una URL, pueden aparecer muchos errores debido a la pila de red. A menudo, este es un problema específico de la máquina que ejecuta el código y no un problema con la biblioteca. Aquí hay una lista de los errores más comunes:

- **"EOF occurred in violation of protocol"** (ocurrió un fin de archivo en violación del protocolo): es probable que este problema se genere si la versión de Python instalada no admite los requisitos de TLS del servidor que aloja el módulo. En particular, se sabe que Python 2.7.5 no resuelve módulos del dominio tfhub.dev. **SOLUCIÓN**: actualice Python a una versión más reciente.

- **"cannot verify tfhub.dev's certificate"** (no se puede verificar el certificado de tfhub.dev): es probable que este problema se genere si hay algo en la red que intenta actuar como gTLD de desarrollo. Antes de que .dev se usara como gTLD, los desarrolladores y los marcos a veces usaban nombres .dev para ayudar a probar el código. **SOLUCIÓN:** Identifique y reconfigure el software que intercepta la resolución de nombres en el dominio ".dev".

- Errores al escribir en el directorio de caché `/tmp/tfhub_modules` (o similar): consulte [Almacenamiento en caché](caching.md) para saber qué es y cómo cambiar su ubicación.

Si los errores y soluciones anteriores no funcionan, se puede intentar descargar manualmente un módulo y simular el protocolo de adjuntar `?tf-hub-format=compressed` a la URL para descargar un archivo comprimido tar que debe descomprimirse manualmente en un archivo local. Luego se puede usar la ruta del archivo local en lugar de la URL. Aquí hay un ejemplo breve:

```bash
# Create a folder for the TF hub module.
$ mkdir /tmp/moduleA
# Download the module, and uncompress it to the destination folder. You might want to do this manually.
$ curl -L "https://tfhub.dev/google/universal-sentence-encoder/2?tf-hub-format=compressed" | tar -zxvC /tmp/moduleA
# Test to make sure it works.
$ python
> import tensorflow_hub as hub
> hub.Module("/tmp/moduleA")
```

## Ejecutar inferencia en un módulo preinicializado

Si está escribiendo un programa de Python que aplica un módulo muchas veces a los datos de entrada, puede aplicar las siguientes recetas. (Nota: para atender solicitudes en servicios de producción, considere [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) u otras soluciones escalables sin Python).

Suponiendo que su modelo de caso de uso sea la **inicialización** y **las solicitudes** posteriores (por ejemplo, Django, Flask, servidor HTTP personalizado, etc.), puede configurar el servicio de la siguiente manera:

### TF2 SavedModels

- En la parte de inicialización:
    - Cargue el modelo TF2.0.

```python
import tensorflow_hub as hub

embedding_fn = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
```

- En la parte de la solicitud:
    - Use la función de incrustación para ejecutar la inferencia.

```python
embedding_fn(["Hello world"])
```

Esta llamada de una función tf.function está optimizada para el rendimiento; consulte [la guía de tf.function](https://www.tensorflow.org/guide/function).

### Módulos de TF1 Hub

- En la parte de inicialización:
    - Construya el gráfico con un **marcador de posición**: punto de entrada al gráfico.
    - Inicialice la sesión.

```python
import tensorflow as tf
import tensorflow_hub as hub

# Create graph and finalize (finalizing optional but recommended).
g = tf.Graph()
with g.as_default():
  # We will be feeding 1D tensors of text into the graph.
  text_input = tf.placeholder(dtype=tf.string, shape=[None])
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
  embedded_text = embed(text_input)
  init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

# Create session and initialize.
session = tf.Session(graph=g)
session.run(init_op)
```

- En la parte de la solicitud:
    - Use la sesión para ingresar datos en el gráfico a través del marcador de posición.

```python
result = session.run(embedded_text, feed_dict={text_input: ["Hello world"]})
```

## No se puede cambiar el tipo de modelo (por ejemplo, float32 a bfloat16)

Los SavedModels de TensorFlow (que se comparten en TF Hub o de otro modo) contienen operaciones que funcionan en tipos de datos fijos (a menudo, float32 para los pesos y activaciones intermedias de redes neuronales). No se pueden cambiar después del hecho al cargar SavedModel (pero los editores de modelos pueden optar por publicar diferentes modelos con diferentes tipos de datos).

## Actualizar una versión de modelo

Se pueden actualizar los metadatos de la documentación de las versiones del modelo. Sin embargo, los activos de la versión (archivos modelo) son inmutables. Si desea cambiar los activos del modelo, puede publicar una versión más nueva del modelo. Es una buena práctica ampliar la documentación con un registro de cambios que describa los cambios entre versiones.
