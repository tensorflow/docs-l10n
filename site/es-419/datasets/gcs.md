# tfds y Google Cloud Storage

Se puede usar Google Cloud Storage (GCS) con tfds por varios motivos:

- Almacenamiento de datos preprocesados
- Acceso a conjuntos de datos con datos almacenados en GCS

## Acceder a través del depósito de GCS de TFDS

Algunos conjuntos de datos están disponibles directamente en nuestro depósito de GCS [`gs://tfds-data/datasets/`](https://console.cloud.google.com/storage/browser/tfds-data) sin ninguna autenticación:

- Si `tfds.load(..., try_gcs=False)` (predeterminado), el conjunto de datos se copiará localmente en `~/tensorflow_datasets` durante `download_and_prepare`.
- Si `tfds.load(..., try_gcs=True)`, el conjunto de datos se transmitirá directamente desde GCS (se omitirá `download_and_prepare`).

Se puede comprobar si un conjunto de datos está alojado en el depósito público con `tfds.is_dataset_on_gcs('mnist')`.

## Autenticación

Antes de comenzar, debe decidir cómo quiere realizar la autenticación. Hay tres opciones:

- sin autenticación (también conocido como acceso anónimo)
- con su cuenta de Google
- con una cuenta de servicio (se puede compartir fácilmente con otros miembros de su equipo)

Puede encontrar información detallada en la [documentación de Google Cloud.](https://cloud.google.com/docs/authentication/getting-started)

### Instrucciones simplificadas

Si ejecuta el código desde colab, puede realizar la autenticación con su cuenta, pero debe ejecutar:

```python
from google.colab import auth
auth.authenticate_user()
```

Si ejecuta el código en su máquina local (o en VM), puede realizar la autenticación con su cuenta al ejecutar:

```shell
gcloud auth application-default login
```

Si desea iniciar sesión con una cuenta de servicio, descargue la clave del archivo JSON y establezca

```shell
export GOOGLE_APPLICATION_CREDENTIALS=<JSON_FILE_PATH>
```

## Usar Google Cloud Storage para almacenar datos preprocesados

Normalmente, cuando se usan conjuntos de datos de TensorFlow, los datos descargados y preparados se almacenan en caché en un directorio local (de forma predeterminada `~/tensorflow_datasets`).

En algunos entornos donde el disco local puede ser efímero (un servidor temporal en la nube o un [bloc de notas de Colab](https://colab.research.google.com)) o cuando se necesita que varias máquinas puedan acceder a los datos, es útil configurar `data_dir` en un sistema de almacenamiento en la nube, como el depósito de Google Cloud Storage (GCS).

### ¿Cómo?

[Cree un depósito de GCS](https://cloud.google.com/storage/docs/creating-buckets) y asegúrese de que usted (o su cuenta de servicio) tenga los permisos de lectura/escritura (consulte las instrucciones de autorización que se mencionan anteriormente)

Cuando usa `tfds`, puede configurar `data_dir` en `"gs://YOUR_BUCKET_NAME"`

```python
ds_train, ds_test = tfds.load(name="mnist", split=["train", "test"], data_dir="gs://YOUR_BUCKET_NAME")
```

### Precauciones:

- Este enfoque funciona para conjuntos de datos que solo usan `tf.io.gfile` para acceder a los datos. Esto es cierto para la mayoría de los conjuntos de datos, pero no para todos.
- Recuerde que al acceder a GCS se accede a un servidor remoto y se transmiten datos desde él, por lo que puede incurrir en costos de red.

## Acceder a conjuntos de datos almacenados en GCS

Si los propietarios del conjunto de datos permitieron el acceso anónimo, puede seguir adelante y ejecutar el código tfds.load, y funcionará como una descarga normal de Internet.

Si el conjunto de datos requiere autenticación, siga las instrucciones anteriores para decidir qué opción desea (cuenta propia o cuenta de servicio) y comunique el nombre de la cuenta (también conocido como correo electrónico) al propietario del conjunto de datos. Después de que le permitan acceder al directorio GCS, debería poder ejecutar el código de descarga de tfds.
