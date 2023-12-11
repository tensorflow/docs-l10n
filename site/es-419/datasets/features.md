# FeatureConnector

La API `tfds.features.FeatureConnector`:

- Define la estructura, formas y tipos del `tf.data.Dataset` final.
- Abstraer la serialización hacia/desde el disco.
- Expone metadatos adicionales (por ejemplo, nombres de etiquetas, frecuencia de las muestras de audio,...)

## Descripción general

`tfds.features.FeatureConnector` define la estructura de funciones del conjunto de datos (en `tfds.core.DatasetInfo`):

```python
tfds.core.DatasetInfo(
    features=tfds.features.FeaturesDict({
        'image': tfds.features.Image(shape=(28, 28, 1), doc='Grayscale image'),
        'label': tfds.features.ClassLabel(
            names=['no', 'yes'],
            doc=tfds.features.Documentation(
                desc='Whether this is a picture of a cat',
                value_range='yes or no'
            ),
        ),
        'metadata': {
            'id': tf.int64,
            'timestamp': tfds.features.Scalar(
                tf.int64,
                doc='Timestamp when this picture was taken as seconds since epoch'),
            'language': tf.string,
        },
    }),
)
```

Las funciones se pueden documentar con una sola descripción textual (`doc='description'`) o directamente con `tfds.features.Documentation` para proporcionar una descripción de funciones más detallada.

Las funciones pueden ser:

- Valores escalares: `tf.bool`, `tf.string`, `tf.float32`,... Cuando quiera documentar la función, también puede usar `tfds.features.Scalar(tf.int64, doc='description')` .
- `tfds.features.Audio`, `tfds.features.Video`,... (consulte [la lista](https://www.tensorflow.org/datasets/api_docs/python/tfds/features?version=nightly) de funciones disponibles)
- `dict` anidado de funciones: `{'metadata': {'image': Image(), 'description': tf.string}}`,...
- `tfds.features.Sequence` anidado: `Sequence({'image': ..., 'id': ...})`, `Sequence(Sequence(tf.int64))`,...

Durante la generación, `FeatureConnector.encode_example` serializará automáticamente los ejemplos en un formato adecuado para el disco (actualmente, búfers de protocolo `tf.train.Example`):

```python
yield {
    'image': '/path/to/img0.png',  # `np.array`, file bytes,... also accepted
    'label': 'yes',  # int (0-num_classes) also accepted
    'metadata': {
        'id': 43,
        'language': 'en',
    },
}
```

Al leer el conjunto de datos (por ejemplo, con `tfds.load`), los datos se decodifican automáticamente con `FeatureConnector.decode_example`. El `tf.data.Dataset` que se devuelve coincidirá con la estructura de `dict` definida en `tfds.core.DatasetInfo`:

```python
ds = tfds.load(...)
ds.element_spec == {
    'image': tf.TensorSpec(shape=(28, 28, 1), tf.uint8),
    'label': tf.TensorSpec(shape=(), tf.int64),
    'metadata': {
        'id': tf.TensorSpec(shape=(), tf.int64),
        'language': tf.TensorSpec(shape=(), tf.string),
    },
}
```

## Serializar/deserializar a proto

TFDS expone una API de bajo nivel para serializar/deserializar ejemplos en el proto `tf.train.Example`.

Para serializar `dict[np.ndarray | Path | str | ...]` para `bytes`, use `features.serialize_example`:

```python
with tf.io.TFRecordWriter('path/to/file.tfrecord') as writer:
  for ex in all_exs:
    ex_bytes = features.serialize_example(data)
    f.write(ex_bytes)
```

Para deserializar proto `bytes` a `tf.Tensor`, use `features.deserialize_example`:

```python
ds = tf.data.TFRecordDataset('path/to/file.tfrecord')
ds = ds.map(features.deserialize_example)
```

## Acceder a los metadatos

Consulte el [documento de introducción](https://www.tensorflow.org/datasets/overview#access_the_dataset_metadata) para acceder a los metadatos de las funciones (nombres de etiquetas, forma, tipo,...). Ejemplo:

```python
ds, info = tfds.load(..., with_info=True)

info.features['label'].names  # ['cat', 'dog', ...]
info.features['label'].str2int('cat')  # 0
```

## Crear su propio `tfds.features.FeatureConnector`

Si cree que falta una función en las [funciones disponibles](https://www.tensorflow.org/datasets/api_docs/python/tfds/features#classes), abra un [nuevoa problema](https://github.com/tensorflow/datasets/issues).

Para crear su propio conector de funciones, debe heredar de `tfds.features.FeatureConnector` e implementar los métodos abstractos.

- Si su función es un valor de tensor único, es mejor heredar de `tfds.features.Tensor` y usar `super()` cuando sea necesario. Consulte el código fuente `tfds.features.BBoxFeature` para ver un ejemplo.
- Si su función es un contenedor de varios tensores, es mejor heredar de `tfds.features.FeaturesDict` y usar `super()` para codificar automáticamente los subconectores.

El objeto `tfds.features.FeatureConnector` abstrae cómo se codifica la característica en el disco de cómo se presenta al usuario. A continuación se muestra un diagrama que muestra las capas de abstracción del conjunto de datos y la transformación de los archivos del conjunto de datos sin procesar al objeto `tf.data.Dataset`.

<p align="center">   <img src="dataset_layers.png" width="700" alt="Capas de abstracción de DatasetBuilder"></p>

Para crear su propio conector de funciones, subclasifíque `tfds.features.FeatureConnector` e implemente los métodos abstractos:

- `encode_example(data)`: define cómo codificar los datos proporcionados en el generador `_generate_examples()` en datos compatibles con `tf.train.Example`. Puede devolver un valor único o un `dict` de valores.
- `decode_example(data)`: define cómo decodificar los datos del tensor que se lee desde `tf.train.Example` en el tensor de usuario que devuelve `tf.data.Dataset`.
- `get_tensor_info()`: Indica la forma/tipo de los tensores que devuelve `tf.data.Dataset`. Puede ser opcional si se hereda de otro `tfds.features`.
- (opcional) `get_serialized_info()`: si la información que devuelve `get_tensor_info()` es diferente de cómo se escriben realmente los datos en el disco, entonces necesita sobrescribir `get_serialized_info()` para que coincida con las especificaciones de `tf.train.Example`
- `to_json_content`/`from_json_content`: se necesita para permitir que su conjunto de datos se cargue sin el código fuente original. Consulte [la función de audio](https://github.com/tensorflow/datasets/blob/65a76cb53c8ff7f327a3749175bc4f8c12ff465e/tensorflow_datasets/core/features/audio_feature.py#L121) para ver un ejemplo.

Nota: asegúrese de probar sus conectores de las funciones con `self.assertFeature` y `tfds.testing.FeatureExpectationItem`. Eche un vistazo a [los ejemplos de prueba](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/features/image_feature_test.py):

Para obtener más información, consulte la documentación `tfds.features.FeatureConnector`. También es mejor ver [ejemplos reales](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/features).
