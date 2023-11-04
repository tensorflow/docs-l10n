# FeatureConnector

A API `tfds.features.FeatureConnector`:

- Define a estrutura, formatos e dtypes do `tf.data.Dataset` final
- Abstrai a serialização de/para o disco.
- Expõe metadados adicionais (por exemplo, nomes de rótulos, taxa de amostragem de áudio,...)

## Visão geral

O `tfds.features.FeatureConnector` define a estrutura de características do dataset (em `tfds.core.DatasetInfo`):

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

Características podem ser documentadas usando apenas uma descrição textual (`doc='description'`) ou usando `tfds.features.Documentation` diretamente para fornecer uma descrição mais detalhada da característica.

Características podem ser:

- Valores escalares: `tf.bool`, `tf.string`, `tf.float32`,... Quando quiser documentar a característica, você também pode usar `tfds.features.Scalar(tf.int64, doc='description')`.
- `tfds.features.Audio`, `tfds.features.Video`,... (veja [a lista](https://www.tensorflow.org/datasets/api_docs/python/tfds/features?version=nightly) de características disponíveis)
- `dict` aninhado de características: `{'metadata': {'image': Image(), 'description': tf.string}}`,...
- `tfds.features.Sequence` aninhados: `Sequence({'image': ..., 'id': ...})`, `Sequence(Sequence(tf.int64))`,...

Durante a geração, os exemplos serão serializados automaticamente por `FeatureConnector.encode_example` num formato adequado ao disco (atualmente buffers de protocolo `tf.train.Example`):

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

Ao ler o dataset (por exemplo, com `tfds.load`), os dados são decodificados automaticamente com `FeatureConnector.decode_example`. O `tf.data.Dataset` retornado corresponderá à estrutura do `dict` definida em `tfds.core.DatasetInfo`:

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

## Serializar/desserializar para proto

O TFDS expõe uma API de baixo nível para serializar/desserializar exemplos para o proto `tf.train.Example`.

Para serializar `dict[np.ndarray | Path | str | ...]` para proto `bytes`, use `features.serialize_example`:

```python
with tf.io.TFRecordWriter('path/to/file.tfrecord') as writer:
  for ex in all_exs:
    ex_bytes = features.serialize_example(data)
    f.write(ex_bytes)
```

Para desserializar para proto `bytes` para `tf.Tensor`, use `features.deserialize_example`:

```python
ds = tf.data.TFRecordDataset('path/to/file.tfrecord')
ds = ds.map(features.deserialize_example)
```

## Acesso de metadados

ConsVejaulte o [documento de introdução](https://www.tensorflow.org/datasets/overview#access_the_dataset_metadata) para acessar os metadados das característica (nomes dos rótulos, forma, dtype,...). Exemplo:

```python
ds, info = tfds.load(..., with_info=True)

info.features['label'].names  # ['cat', 'dog', ...]
info.features['label'].str2int('cat')  # 0
```

## Crie seu próprio `tfds.features.FeatureConnector`

Se você acredita que uma característica está faltando entre as [características disponíveis](https://www.tensorflow.org/datasets/api_docs/python/tfds/features#classes), abra um [novo issue](https://github.com/tensorflow/datasets/issues).

Para criar seu próprio conector de características, você precisa herdar de `tfds.features.FeatureConnector` e implementar os métodos abstratos.

- Se sua característica for um valor de tensor único, é melhor herdar de `tfds.features.Tensor` e usar `super()` quando necessário. Veja o código-fonte `tfds.features.BBoxFeature` para um exemplo.
- Se seu recurso for um container de múltiplos tensores, é melhor herdar de `tfds.features.FeaturesDict` e usar `super()` para codificar automaticamente os subconectores.

O objeto `tfds.features.FeatureConnector` abstrai como o recurso é codificado no disco e como ele é apresentado ao usuário. Abaixo está um diagrama que mostra as camadas de abstração do dataset e a transformação dos arquivos brutos do dataset para o objeto `tf.data.Dataset`.

<p align="center">   <img src="dataset_layers.png" width="700" alt="Camadas de abstração do DatasetBuilder"></p>

Para criar seu próprio conector de características, crie uma subclasse de `tfds.features.FeatureConnector` e implemente os métodos abstratos a seguir:

- `encode_example(data)`: define como codificar os dados fornecidos no gerador `_generate_examples()` em dados compatíveis com `tf.train.Example`. Pode retornar um único valor ou um `dict` de valores.
- `decode_example(data)`: define como decodificar os dados do tensor lido de `tf.train.Example` no tensor do usuário retornado por `tf.data.Dataset`.
- `get_tensor_info()`: indica o formato/tipo do(s) tensor(es) retornado(s) por `tf.data.Dataset`. Pode ser opcional se herdar de outro `tfds.features`.
- (opcionalmente) `get_serialized_info()`: se as informações retornadas por `get_tensor_info()` forem diferentes de como os dados são realmente gravados no disco, então você precisa sobrescrever `get_serialized_info()` para corresponder às especificações do `tf.train.Example`
- `to_json_content` / `from_json_content`: é necessário para permitir que seu dataset seja carregado sem o código-fonte original. Veja [Recurso de áudio](https://github.com/tensorflow/datasets/blob/65a76cb53c8ff7f327a3749175bc4f8c12ff465e/tensorflow_datasets/core/features/audio_feature.py#L121) para um exemplo.

Observação: certifique-se de testar seus conectores de características com `self.assertFeature` e `tfds.testing.FeatureExpectationItem`. Dê uma olhada nos [exemplos de teste](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/features/image_feature_test.py):

Para obter mais informações, dê uma olhada na documentação `tfds.features.FeatureConnector`. Também é melhor ver alguns [exemplos reais](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/features).
