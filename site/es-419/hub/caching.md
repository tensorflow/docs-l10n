# Almacenar en caché las descargas de modelos desde TF Hub

## Descripción general

Actualmente, la biblioteca `tensorflow_hub` admite dos modos para descargar modelos. De forma predeterminada, un modelo se descarga como un archivo comprimido y se almacena en caché en el disco. En segundo lugar, los modelos se pueden leer directamente desde el almacenamiento remoto en TensorFlow. De cualquier manera, las llamadas a las funciones `tensorflow_hub` en el código Python real pueden y deben continuar usando las URL canónicas tfhub.dev de los modelos, que son portátiles entre sistemas y navegables para documentación. En el caso inusual en el que el código de usuario necesite la ubicación real del sistema de archivos (después de descargarlo y descomprimirlo, o después de resolver un identificador de modelo en una ruta del sistema de archivos), se puede obtener mediante la función `hub.resolve(handle)`.

### Almacenar en caché de descargas comprimidas

La biblioteca `tensorflow_hub` almacena en caché de forma predeterminada los modelos en el filesystem cuando se descargan de tfhub.dev (u otros [sitios de alojamiento](hosting.md)) y se descomprimen. Se recomienda este modo para la mayoría de los entornos, excepto si el espacio en disco es escaso pero el ancho de banda y la latencia de la red son excelentes.

La ubicación de descarga predeterminada es un directorio temporal local, pero se puede personalizar configurando la variable de entorno `TFHUB_CACHE_DIR` (recomendado) o pasando el indicador de línea de comandos `--tfhub_cache_dir`. La ubicación de caché predeterminada `/tmp/tfhub_modules` (o cualquier cosa que se evalúe `os.path.join(tempfile.gettempdir(), "tfhub_modules")`) debería funcionar en la mayoría de los casos.

Los usuarios que prefieren el almacenamiento en caché persistente durante los reinicios del sistema pueden configurar `TFHUB_CACHE_DIR` en una ubicación en su directorio de inicio. Por ejemplo, un usuario de shell bash en un sistema Linux puede agregar una línea como la siguiente a `~/.bashrc`

```bash
export TFHUB_CACHE_DIR=$HOME/.cache/tfhub_modules
```

... reinicie el shell y luego se usará esta ubicación. Cuando use una ubicación persistente, tenga en cuenta que no existe una limpieza automática.

### Leeer desde un almacenamiento remoto

Los usuarios pueden indicarle a la biblioteca `tensorflow_hub` que lea directamente los modelos desde el almacenamiento remoto (Google Cloud Storage, GCS) en lugar de descargar los modelos localmente con

```shell
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED"
```

o configurando el indicador de línea de comando `--tfhub_model_load_format` en `UNCOMPRESSED`. De esta manera, no se necesita ningún directorio de almacenamiento en caché, lo que resulta especialmente útil en entornos que ofrecen poco espacio en disco pero una conexión rápida a Internet.

### Ejecutar en TPU en cuadernos Colab

En [colab.research.google.com](https://colab.research.google.com), la descarga de modelos comprimidos estárá en conflicto con el tiempo de ejecución de TPU ya que la carga de trabajo de cálculo se delega a otra máquina que no tiene acceso a la ubicación de la caché de forma predeterminada. Hay dos soluciones para esta situación:

#### 1) Usar un depósito de GCS al que el trabajador de TPU pueda acceder

La solución más fácil es indicarle a la biblioteca `tensorflow_hub` que lea los modelos del depósito GCS de TF Hub como se explicó anteriormente. Los usuarios con su propio depósito de GCS pueden especificar un directorio en su depósito como ubicación de caché con un código como

```python
import os
os.environ["TFHUB_CACHE_DIR"] = "gs://my-bucket/tfhub-modules-cache"
```

...antes de llamar a la biblioteca `tensorflow_hub`.

#### 2) Redirigir todas las lecturas a través del host de Colab

Otra solución es redirigir todas las lecturas (incluso las de variables grandes) a través del host de Colab:

```python
load_options =
tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
reloaded_model = hub.load("https://tfhub.dev/...", options=load_options)
```

**Nota:** consulte más información sobre los identificadores válidos [aquí](tf2_saved_model.md#model_handles).
