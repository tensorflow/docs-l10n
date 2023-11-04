# Cargar tfrecord externo con TFDS

Si tiene un protocolo `tf.train.Example` (dentro de `.tfrecord`, `.riegeli` ,...), que fue generado con herramientas de terceros y lo quiere cargar directamente con la API de tfds, entonces esta página es para usted.

Para cargar sus archivos `.tfrecord`, sólo debe:

- Seguir la convención de nomenclatura de TFDS.
- Agregar archivos de metadatos (`dataset_info.json`, `features.json`) junto con sus archivos tfrecord.

Limitaciones:

- `tf.train.SequenceExample` no es compatible, solo `tf.train.Example`.
- Se debe poder expresar `tf.train.Example` en términos de `tfds.features` (vea la sección a continuación).

## Convención de nomenclatura de archivos

TFDS admite la definición de una plantilla para los nombres de archivos, lo que proporciona flexibilidad para usar diferentes esquemas de nombres de archivos. La plantilla se representa con `tfds.core.ShardedFileTemplate` y admite las siguientes variables: `{DATASET}`, `{SPLIT}`, `{FILEFORMAT}`, `{SHARD_INDEX}`, `{NUM_SHARDS}` y `{SHARD_X_OF_Y}`. Por ejemplo, el esquema de nomenclatura de archivos predeterminado de TFDS es: `{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}`. Para MNIST, esto significa que [los nombres de los archivos](https://console.cloud.google.com/storage/browser/tfds-data/datasets/mnist/3.0.1) se ven así:

- `mnist-test.tfrecord-00000-of-00001`
- `mnist-train.tfrecord-00000-of-00001`

## Agregar metadatos

### Proporcionar la estructura de las funciones

Para que TFDS pueda decodificar el protocolo `tf.train.Example`, hay proporcionar la estructura `tfds.features` que coincida con sus especificaciones. Por ejemplo:

```python
features = tfds.features.FeaturesDict({
    'image':
        tfds.features.Image(
            shape=(256, 256, 3),
            doc='Picture taken by smartphone, downscaled.'),
    'label':
        tfds.features.ClassLabel(names=['dog', 'cat']),
    'objects':
        tfds.features.Sequence({
            'camera/K': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
        }),
})
```

Corresponde a las siguientes especificaciones `tf.train.Example`:

```python
{
    'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
    'label': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
    'objects/camera/K': tf.io.FixedLenSequenceFeature(shape=(3,), dtype=tf.int64),
}
```

La especificación de las funciones permite que TFDS decodifique automáticamente imágenes, videos,... Como cualquier otro conjunto de datos TFDS, los metadatos de las funciones (por ejemplo, los nombres de etiquetas,...) estarán a la vista de los usuarios (por ejemplo `info.features['label'].names`).

#### Si se controla la canalización de generación

Si se generan conjuntos de datos fuera de TFDS pero aún se controla la canalización de generación, puede usar `tfds.features.FeatureConnector.serialize_example` para codificar sus datos desde `dict[np.ndarray]` al proto de `tf.train.Example` `bytes`:

```python
with tf.io.TFRecordWriter('path/to/file.tfrecord') as writer:
  for ex in all_exs:
    ex_bytes = features.serialize_example(data)
    writer.write(ex_bytes)
```

Esto garantizará que las funciones sean compatibles con TFDS.

De manera similar, existe un `feature.deserialize_example` para decodificar el proto de ([ejemplo](https://www.tensorflow.org/datasets/features#serializedeserialize_to_proto))

#### Si no se controla la canalización de generación

Si desea ver cómo se representan las `tfds.features` en un `tf.train.Example`, puede examinarlo en colab:

- Para traducir `tfds.features` a la estructura legible para los humanos de `tf.train.Example`, puede llamar `features.get_serialized_info()`.
- Para obtener la `FixedLenFeature` exacta,... la especificaciónque se pasa a `tf.io.parse_single_example`, puede usar `spec = features.tf_example_spec`

Nota: Si usa un conector de funciones personalizado, asegúrese de implementar `to_json_content`/`from_json_content` y probarlo con `self.assertFeature` (consulte [la guía del conector de funciones](https://www.tensorflow.org/datasets/features#create_your_own_tfdsfeaturesfeatureconnector)).

### Obtener estadísticas de las divisiones

TFDS debe conocer la cantidad exacta de ejemplos dentro de cada partición. Es necesario para las funciones como `len(ds)` o la [API subplit](https://www.tensorflow.org/datasets/splits): `split='train[75%:]'`.

- Si tiene esta información, puede crear explícitamente una lista de `tfds.core.SplitInfo` y pasar a la siguiente sección:

    ```python
    split_infos = [
        tfds.core.SplitInfo(
            name='train',
            shard_lengths=[1024, ...],  # Num of examples in shard0, shard1,...
            num_bytes=0,  # Total size of your dataset (if unknown, set to 0)
        ),
        tfds.core.SplitInfo(name='test', ...),
    ]
    ```

- Si no tiene esta información, puede calcularla con el script `compute_split_info.py` (o en su propio script con `tfds.folder_dataset.compute_split_info`). Se iniciará una canalización beam que leerá todas las particiones en el directorio dado y calculará la información.

### Agregar archivos de metadatos

Para agregar automáticamente los archivos de metadatos adecuados a su conjunto de datos, use `tfds.folder_dataset.write_metadata`:

```python
tfds.folder_dataset.write_metadata(
    data_dir='/path/to/my/dataset/1.0.0/',
    features=features,
    # Pass the `out_dir` argument of compute_split_info (see section above)
    # You can also explicitly pass a list of `tfds.core.SplitInfo`.
    split_infos='/path/to/my/dataset/1.0.0/',
    # Pass a custom file name template or use None for the default TFDS
    # file name template.
    filename_template='{SPLIT}-{SHARD_X_OF_Y}.{FILEFORMAT}',

    # Optionally, additional DatasetInfo metadata can be provided
    # See:
    # https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetInfo
    description="""Multi-line description."""
    homepage='http://my-project.org',
    supervised_keys=('image', 'label'),
    citation="""BibTex citation.""",
)
```

Cuando se llame a la función en su directorio de conjunto de datos una vez, se agregarán los archivos de metadatos (`dataset_info.json` ,...), y sus conjuntos de datos estarán listos para cargarse con TFDS (vea la siguiente sección).

## Cargar conjunto de datos con TFDS

### Directamente desde la carpeta

Cuando se generan los metadatos, los conjuntos de datos se pueden cargar con `tfds.builder_from_directory`, que devuelve un `tfds.core.DatasetBuilder` con la API TFDS estándar (como `tfds.builder`):

```python
builder = tfds.builder_from_directory('~/path/to/my_dataset/3.0.0/')

# Metadata are available as usual
builder.info.splits['train'].num_examples

# Construct the tf.data.Dataset pipeline
ds = builder.as_dataset(split='train[75%:]')
for ex in ds:
  ...
```

### Directamente desde varias carpetas

También es posible cargar datos desde varias carpetas. Se puede hacer, por ejemplo, en el aprendizaje de refuerzo cuando cada uno de los varios agentes genera un conjunto de datos separado, se busca cargarlos todos juntos. Otros casos de uso son cuando se produce un nuevo conjunto de datos de forma regular, por ejemplo, un nuevo conjunto de datos por día, y busca cargar los datos de un rango de fechas.

Para cargar datos desde varias carpetas, use `tfds.builder_from_directories`, que devuelve un `tfds.core.DatasetBuilder` con la API TFDS estándar (como `tfds.builder`):

```python
builder = tfds.builder_from_directories(builder_dirs=[
    '~/path/my_dataset/agent1/1.0.0/',
    '~/path/my_dataset/agent2/1.0.0/',
    '~/path/my_dataset/agent3/1.0.0/',
])

# Metadata are available as usual
builder.info.splits['train'].num_examples

# Construct the tf.data.Dataset pipeline
ds = builder.as_dataset(split='train[75%:]')
for ex in ds:
  ...
```

Nota: cada carpeta debe tener sus propios metadatos, porque estos contienen información sobre las divisiones.

### Estructura de carpetas (opcional)

Para una mejor compatibilidad con TFDS, puede organizar sus datos en `<data_dir>/<dataset_name>[/<dataset_config>]/<dataset_version>`. Por ejemplo:

```
data_dir/
    dataset0/
        1.0.0/
        1.0.1/
    dataset1/
        config0/
            2.0.0/
        config1/
            2.0.0/
```

Así sus conjuntos de datos serán compatibles con la API `tfds.load`/`tfds.builder`, al simplemente proporcionar `data_dir/`:

```python
ds0 = tfds.load('dataset0', data_dir='data_dir/')
ds1 = tfds.load('dataset1/config0', data_dir='data_dir/')
```
