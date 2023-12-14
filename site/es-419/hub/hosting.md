# Protocolo de alojamiento del <br>modelo

En este documento se describen las convenciones de URL usadas al alojar todos los tipos de modelos en [tfhub.dev](https://tfhub.dev): modelos TFJS, TF Lite y TensorFlow. También se describe el protocolo basado en HTTP(S) que implementa la biblioteca `tensorflow_hub` para cargar modelos de TensorFlow desde [tfhub.dev](https://tfhub.dev) y servicios compatibles en programas de TensorFlow.

Su característica clave es usar la misma URL en el código para cargar un modelo y en un navegador para ver la documentación del modelo.

## Convenciones generales de URL

[tfhub.dev](https://tfhub.dev) admite los siguientes formatos de URL:

- Los editores de TF Hub siguen `https://tfhub.dev/<publisher>`
- Las colecciones de TF Hub siguen `https://tfhub.dev/<publisher>/collection/<collection_name>`
- Los modelos TF Hub tienen una URL versionada `https://tfhub.dev/<publisher>/<model_name>/<version>` y una URL no versionada `https://tfhub.dev/<publisher>/<model_name>` que se resuelve en la última versión del modelo.

Los modelos de TF Hub se pueden descargar como activos comprimidos agregando parámetros de URL a la URL del modelo [tfhub.dev](https://tfhub.dev). Sin embargo, los parámetros de URL necesarios para lograrlo dependen del tipo de modelo:

- Modelos de TensorFlow (formatos SavedModel y TF1 Hub): agregue `?tf-hub-format=compressed` a la URL del modelo de TensorFlow.
- Modelos TFJS: agregue `?tfjs-format=compressed` a la URL del modelo TFJS para descargar el archivo comprimido o `/model.json?tfjs-format=file` para leerlo desde el almacenamiento remoto.
- Modelos TF Lite: agregue `?lite-format=tflite` a la URL del modelo TF Lite.

Por ejemplo:

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 10%">
    <col style="width: 20%">
    <col style="width: 15%">
    <col style="width: 30%">
    <col style="width: 25%">
    <td style="text-align: center; background-color: #D0D0D0">Tipo</td>
    <td style="text-align: center; background-color: #D0D0D0">URL del modelo</td>
    <td style="text-align: center; background-color: #D0D0D0">Tipo de descarga</td>
    <td style="text-align: center; background-color: #D0D0D0">Parámetro de URL</td>
    <td style="text-align: center; background-color: #D0D0D0">Descargar URL</td>
  </tr>
  <tr>
    <td>TensorFlow (SavedModel, formato TF1 Hub)</td>
    <td>https://tfhub.dev/google/spice/2</td>
    <td>.tar.gz</td>
    <td>?tf-hub-format=compressed</td>
    <td>https://tfhub.dev/google/spice/2?tf-hub-format=compressed</td>
  </tr>
  <tr>
    <td>TF Lite</td>
    <td>https://tfhub.dev/google/lite-model/spice/1</td>
    <td>.tflite</td>
    <td>?lite-format=tflite</td>
    <td>https://tfhub.dev/google/lite-model/spice/1?lite-format=tflite</td>
  </tr>
  <tr>
    <td>TF.js</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1</td>
    <td>.tar.gz</td>
    <td>?tfjs-format=compressed</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1?tfjs-format=compressed</td>
  </tr>
</table>

Además, algunos modelos también están alojados en un formato que se puede leer directamente desde el almacenamiento remoto sin necesidad de descargarlos. Esto es especialmente útil si no hay almacenamiento local disponible, como ejecutar un modelo TF.js en el navegador o cargar un SavedModel en [Colab](https://colab.research.google.com/). Tenga en cuenta que leer modelos alojados de forma remota sin descargarlos localmente puede aumentar la latencia.

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 10%">
    <col style="width: 20%">
    <col style="width: 15%">
    <col style="width: 30%">
    <col style="width: 25%">
    <td style="text-align: center; background-color: #D0D0D0">Tipo</td>
    <td style="text-align: center; background-color: #D0D0D0">URL del modelo</td>
    <td style="text-align: center; background-color: #D0D0D0">Tipo de respuesta</td>
    <td style="text-align: center; background-color: #D0D0D0">Parámetro de URL</td>
    <td style="text-align: center; background-color: #D0D0D0">URL de la solicitud</td>
  </tr>
  <tr>
    <td>TensorFlow (SavedModel, formato TF1 Hub)</td>
    <td>https://tfhub.dev/google/spice/2</td>
    <td>Cadena de texto (Ruta a la carpeta de GCS donde se almacena el modelo sin comprimir)</td>
    <td>?tf-hub-format=uncompressed</td>
    <td>https://tfhub.dev/google/spice/2?tf-hub-format=uncompressed</td>
  </tr>
  <tr>
    <td>TF.js</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1</td>
    <td>.json</td>
    <td>?tfjs-format=file</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1/model.json?tfjs-format=file</td>
  </tr>
</table>

## Protocolo de la biblioteca tensorflow_hub

En esta sección se describe cómo alojamos modelos en [tfhub.dev](https://tfhub.dev) para usarlos con la biblioteca tensorflow_hub. Si quiere alojar su propio repositorio de modelos para trabajar con la biblioteca tensorflow_hub, su servicio de distribución HTTP debe proporcionar una implementación de este protocolo.

Tenga en cuenta que esta sección no aborda el alojamiento de los modelos TF Lite y TFJS, ya que no se descargan a través de la biblioteca `tensorflow_hub`. Para obtener más información sobre cómo alojar estos tipos de modelos, consulte lo que se menciona [anteriormente](#general-url-conventions).

### Alojamiento comprimido

Los modelos se almacenan en [tfhub.dev](https://tfhub.dev) como archivos comprimidos tar.gz. De forma predeterminada, la biblioteca tensorflow_hub descarga el modelo comprimido automáticamente. También se pueden descargar manualmente al agregar `?tf-hub-format=compressed` a la URL del modelo, por ejemplo:

```shell
wget https://tfhub.dev/tensorflow/albert_en_xxlarge/1?tf-hub-format=compressed
```

La raíz del archivo es la raíz del directorio del modelo y debe contener un SavedModel, como en este ejemplo:

```shell
# Create a compressed model from a SavedModel directory.
$ tar -cz -f model.tar.gz --owner=0 --group=0 -C /tmp/export-model/ .

# Inspect files inside a compressed model
$ tar -tf model.tar.gz
./
./variables/
./variables/variables.data-00000-of-00001
./variables/variables.index
./assets/
./saved_model.pb
```

Los archivos tarball para usar con el [formato TF1 Hub](https://www.tensorflow.org/hub/tf1_hub_module) heredado también tendrán un archivo `./tfhub_module.pb`.

Cuando se invoca una de las API de carga de modelos de la biblioteca `tensorflow_hub` ([hub.KerasLayer](https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer), [hub.load](https://www.tensorflow.org/hub/api_docs/python/hub/load), etc.), la biblioteca descarga el modelo, lo descomprime y lo almacena en caché localmente. La biblioteca `tensorflow_hub` requiere que las URL del modelo tengan versiones y que el contenido del modelo de una versión determinada sea inmutable, de modo que pueda almacenarse en caché indefinidamente. Obtenga más información sobre [almacenar modelos en caché](caching.md).

![](https://raw.githubusercontent.com/tensorflow/hub/master/docs/images/library_download_cache.png)

### Alojamiento sin comprimir

Cuando la variable de entorno `TFHUB_MODEL_LOAD_FORMAT` o el indicador de línea de comandos `--tfhub_model_load_format` se establece en `UNCOMPRESSED`, el modelo se lee directamente desde el almacenamiento remoto (GCS) en lugar de descargarlo y descomprimirlo localmente. Cuando este comportamiento está habilitado, la biblioteca agrega `?tf-hub-format=uncompressed` a la URL del modelo. Esa solicitud devuelve la ruta a la carpeta en GCS que contiene los archivos del modelo sin comprimir. Como ejemplo, <br> `https://tfhub.dev/google/spice/2?tf-hub-format=uncompressed`<br> devoluciones<br> `gs://tfhub-modules/google/spice/2/uncompressed` en el cuerpo de la respuesta 303. Luego, la biblioteca lee el modelo desde ese destino GCS.
