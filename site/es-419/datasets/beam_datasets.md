# Generar conjuntos de datos grandes con Apache Beam

Algunos conjuntos de datos son demasiado grandes para procesarlos en una sola máquina. `tfds` admite la generación de datos en muchas máquinas mediante el uso de [Apache Beam](https://beam.apache.org/).

Este documento tiene dos secciones:

- Para usuarios que desean generar un conjunto de datos Beam existente
- Para desarrolladores que quieran crear un conjunto de datos Beam nuevo

## Generar un conjunto de datos Beam

A continuación, se muestran diferentes ejemplos de generación de un conjunto de datos Beam, tanto en la nube como de forma local.

**Advertencia**: al generar el conjunto de datos con la [ interfaz de la línea de comandos (CLI) `tfds build`](https://www.tensorflow.org/datasets/cli#tfds_build_download_and_prepare_a_dataset), asegúrese de especificar la configuración del conjunto de datos que desea generar, o este generará todas las configuraciones existentes de forma predeterminada. Por ejemplo, para [wikipedia](https://www.tensorflow.org/datasets/catalog/wikipedia), use `tfds build wikipedia/20200301.en` en lugar de `tfds build wikipedia`.

### En Google Cloud Dataflow

Para ejecutar la canalización mediante el uso de [Google Cloud Dataflow](https://cloud.google.com/dataflow/) y aprovechar el cálculo distribuido, primero siga las [instrucciones de inicio rápido](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python).

Una vez que esté configurado su entorno, puede ejecutar la [CLI `tfds build`](https://www.tensorflow.org/datasets/cli#tfds_build_download_and_prepare_a_dataset) con un directorio de datos en [Google Cloud Storage (GCS)](https://cloud.google.com/storage/) y al especificar las [opciones obligatorias](https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#configuring-pipelineoptions-for-execution-on-the-cloud-dataflow-service) para el indicador `--beam_pipeline_options`.

Para facilitar el inicio del script, es útil definir las siguientes variables con los valores reales para su configuración de GCP/GCS y el conjunto de datos que desea generar:

```sh
DATASET_NAME=<dataset-name>
DATASET_CONFIG=<dataset-config>
GCP_PROJECT=my-project-id
GCS_BUCKET=gs://my-gcs-bucket
```

Luego, deberá crear un archivo para indicarle a Dataflow que instale `tfds` en los trabajadores:

```sh
echo "tensorflow_datasets[$DATASET_NAME]" > /tmp/beam_requirements.txt
```

Si usa `tfds-nightly`, asegúrese de hacer eco desde `tfds-nightly` en caso de que el conjunto de datos se haya actualizado desde la última versión.

```sh
echo "tfds-nightly[$DATASET_NAME]" > /tmp/beam_requirements.txt
```

Por último, puede iniciar el trabajo con el siguiente comando:

```sh
tfds build $DATASET_NAME/$DATASET_CONFIG \
  --data_dir=$GCS_BUCKET/tensorflow_datasets \
  --beam_pipeline_options=\
"runner=DataflowRunner,project=$GCP_PROJECT,job_name=$DATASET_NAME-gen,"\
"staging_location=$GCS_BUCKET/binaries,temp_location=$GCS_BUCKET/temp,"\
"requirements_file=/tmp/beam_requirements.txt"
```

### Localmente

Para ejecutar su script localmente con el [ejecutor Apache Beam predeterminado](https://beam.apache.org/documentation/runners/direct/) (deben entrar todos los datos en la memoria), el comando es el mismo que para otros conjuntos de datos:

```sh
tfds build my_dataset
```

**Advertencia**: los conjuntos de datos de Beam pueden ser **gigantes** (terabytes o más) y suelen requerir una cantidad significativa de recursos para generarse (pueden tardar semanas en una computadora local). Se recomienda generar los conjuntos de datos con un entorno distribuido. Eche un vistazo a la [documentación de Apache Beam](https://beam.apache.org/) para obtener una lista de tiempos de ejecución compatibles.

### Con Apache Flink

Para ejecutar la canalización con [Apache Flink,](https://flink.apache.org/) puede leer la [documentación oficial](https://beam.apache.org/documentation/runners/flink). Asegúrese de que su Beam cumpla con [la compatibilidad de versiones de Flink](https://beam.apache.org/documentation/runners/flink/#flink-version-compatibility)

Para facilitar el inicio del script, es útil definir las siguientes variables con los valores reales para la configuración de Flink y el conjunto de datos que desea generar:

```sh
DATASET_NAME=<dataset-name>
DATASET_CONFIG=<dataset-config>
FLINK_CONFIG_DIR=<flink-config-directory>
FLINK_VERSION=<flink-version>
```

Para ejecutarlo en un clúster incrustado de Flink, puede iniciar el trabajo con el siguiente comando:

```sh
tfds build $DATASET_NAME/$DATASET_CONFIG \
  --beam_pipeline_options=\
"runner=FlinkRunner,flink_version=$FLINK_VERSION,flink_conf_dir=$FLINK_CONFIG_DIR"
```

### Con un script personalizado

Para generar el conjunto de datos en Beam, la API es la misma que para otros conjuntos de datos. Puede personalizar [`beam.Pipeline`](https://beam.apache.org/documentation/programming-guide/#creating-a-pipeline) con los argumentos `beam_options` (y `beam_runner`) de `DownloadConfig`.

```python
# If you are running on Dataflow, Spark,..., you may have to set-up runtime
# flags. Otherwise, you can leave flags empty [].
flags = ['--runner=DataflowRunner', '--project=<project-name>', ...]

# `beam_options` (and `beam_runner`) will be forwarded to `beam.Pipeline`
dl_config = tfds.download.DownloadConfig(
    beam_options=beam.options.pipeline_options.PipelineOptions(flags=flags)
)
data_dir = 'gs://my-gcs-bucket/tensorflow_datasets'
builder = tfds.builder('wikipedia/20190301.en', data_dir=data_dir)
builder.download_and_prepare(download_config=dl_config)
```

## Implementar un conjunto de datos Beam

### Requisitos previos

Para escribir conjuntos de datos de Apache Beam, debe conocer los siguientes conceptos:

- Familiarícese con la [guía de creación de conjuntos de datos `tfds`](https://github.com/tensorflow/datasets/blob/master/docs/add_dataset.md), ya que la mayor parte del contenido todavía se aplica a los conjuntos de datos Beam.
- Puede empezar a aprender Apache Beam con la [guía de programación de Beam](https://beam.apache.org/documentation/programming-guide/).
- Si desea generar su conjunto de datos con Cloud Dataflow, lea la [documentación de Google Cloud](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python) y la [guía de dependencia de Apache Beam](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/).

### Instrucciones

Si conoce la [guía de creación de conjuntos de datos](https://github.com/tensorflow/datasets/blob/master/docs/add_dataset.md), para agregar un conjunto de datos Beam solo requiere que se modifique la función `_generate_examples`. La función debería devolver un objeto de Beam, en lugar de un generador:

Conjunto de datos que no son Beam:

```python
def _generate_examples(self, path):
  for f in path.iterdir():
    yield _process_example(f)
```

Conjunto de datos beam:

```python
def _generate_examples(self, path):
  return (
      beam.Create(path.iterdir())
      | beam.Map(_process_example)
  )
```

Todo lo demás puede ser 100 % idéntico, incluidas las pruebas.

Algunas consideraciones adicionales:

- Use `tfds.core.lazy_imports` para importar Apache Beam. Al utilizar una dependencia diferida, los usuarios aún pueden leer el conjunto de datos una vez generado sin tener que instalar Beam.
- Tenga cuidado con las interrupciones de Python. Al ejecutar la canalización, se serializan las funciones `beam.Map` y `beam.DoFn` mediante `pickle` y se envían a todos los trabajadores. No use objetos mutables dentro de una `beam.PTransform` si los trabajadores comparten el estado.
- Debido a la forma en que se serializa `tfds.core.DatasetBuilder` con pickle, los trabajadores ignorarán la mutación `tfds.core.DatasetBuilder` durante la creación de datos (por ejemplo, no es posible establecer `self.info.metadata['offset'] = 123` en `_split_generators` y acceder a él desde los trabajadores como `beam.Map(lambda x: x + self.info.metadata['offset'])`)
- Si necesita compartir algunos pasos de la canalización entre las divisiones, puede agregar un kwarg `pipeline: beam.Pipeline` en `_split_generator` y controlar la canalización de generación completa. Consulte la documentación `_generate_examples` de `tfds.core.GeneratorBasedBuilder`.

### Ejemplo

A continuación se muestra un ejemplo de un conjunto de datos Beam.

```python
class DummyBeamDataset(tfds.core.GeneratorBasedBuilder):

  VERSION = tfds.core.Version('1.0.0')

  def _info(self):
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(16, 16, 1)),
            'label': tfds.features.ClassLabel(names=['dog', 'cat']),
        }),
    )

  def _split_generators(self, dl_manager):
    ...
    return {
        'train': self._generate_examples(file_dir='path/to/train_data/'),
        'test': self._generate_examples(file_dir='path/to/test_data/'),
    }

  def _generate_examples(self, file_dir: str):
    """Generate examples as dicts."""
    beam = tfds.core.lazy_imports.apache_beam

    def _process_example(filename):
      # Use filename as key
      return filename, {
          'image': os.path.join(file_dir, filename),
          'label': filename.split('.')[1],  # Extract label: "0010102.dog.jpeg"
      }

    return (
        beam.Create(tf.io.gfile.listdir(file_dir))
        | beam.Map(_process_example)
    )

```

### Ejecutar su canalización

Para ejecutar la canalización, eche un vistazo a la sección anterior.

**Nota**: Al igual que con los conjuntos de datos que no son beam, no olvide registrar las sumas de comprobación de descarga con `--register_checksums` (solo la primera vez para registrar las descargas).

```sh
tfds build my_dataset --register_checksums
```

## Canalización con TFDS como entrada

Si quiere crear una canalización beam que tome un conjunto de datos TFDS como origen, puede usar `tfds.beam.ReadFromTFDS`:

```python
builder = tfds.builder('my_dataset')

_ = (
    pipeline
    | tfds.beam.ReadFromTFDS(builder, split='train')
    | beam.Map(tfds.as_numpy)
    | ...
)
```

Procesará cada fragmento del conjunto de datos en paralelo.

Nota: Esto requiere que el conjunto de datos ya esté generado. Para generar conjuntos de datos con beam, consulte las otras secciones.
