# Carregando um tfrecord externo com TFDS

Se você possui um proto `tf.train.Example` (dentro de `.tfrecord`, `.riegeli`,...), que foi gerado por ferramentas de terceiros, que você gostaria de carregar diretamente com a API tfds, então esta página é para você.

Para carregar seus arquivos `.tfrecord` , você só precisa:

- Siga a convenção de nomenclatura TFDS.
- Acrescentar arquivos de metadados (`dataset_info.json`, `features.json`) junto com seus arquivos tfrecord.

Limitações

- `tf.train.SequenceExample` não é suportado, apenas `tf.train.Example`.
- Você precisa ser capaz de expressar `tf.train.Example` em termos de `tfds.features` (veja a seção abaixo).

## Convenção de nomenclatura de arquivos

O TFDS suporta a definição de um modelo para nomes de arquivos, o que fornece flexibilidade para usar diferentes esquemas de nomenclatura de arquivos. O modelo é representado por um `tfds.core.ShardedFileTemplate` e suporta as seguintes variáveis: `{DATASET}`, `{SPLIT}`, `{FILEFORMAT}`, `{SHARD_INDEX}`, `{NUM_SHARDS}` e `{SHARD_X_OF_Y}`. Por exemplo, o esquema de nomenclatura de arquivo padrão do TFDS é: `{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}`. Para MNIST, isso significa que [os nomes dos arquivos](https://console.cloud.google.com/storage/browser/tfds-data/datasets/mnist/3.0.1) têm a seguinte aparência:

- `mnist-test.tfrecord-00000-of-00001`
- `mnist-train.tfrecord-00000-of-00001`

## Adicione metadados

### Forneça a estrutura de características

Para que o TFDS seja capaz de decodificar o proto `tf.train.Example`, você precisa fornecer a estrutura `tfds.features` que corresponda às suas especificações. Por exemplo:

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

Corresponde às seguintes especificações `tf.train.Example`:

```python
{
    'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
    'label': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
    'objects/camera/K': tf.io.FixedLenSequenceFeature(shape=(3,), dtype=tf.int64),
}
```

A especificação das características permite que o TFDS decodifique automaticamente imagens, vídeos,... Como qualquer outro dataset do TFDS, os metadados das características (por exemplo, nomes de rótulos,...) serão expostos ao usuário (por exemplo `info.features['label'].names`).

#### Se você controla o pipeline de geração

Se você gerar datasets fora do TFDS, mas ainda controla o pipeline de geração, poderá usar `tfds.features.FeatureConnector.serialize_example` para codificar seus dados de `dict[np.ndarray]` para `tf.train.Example` proto `bytes`:

```python
with tf.io.TFRecordWriter('path/to/file.tfrecord') as writer:
  for ex in all_exs:
    ex_bytes = features.serialize_example(data)
    writer.write(ex_bytes)
```

Isso garantirá a compatibilidade das características com o TFDS.

Da mesma forma, existe um `feature.deserialize_example` para decodificar o proto ([exemplo](https://www.tensorflow.org/datasets/features#serializedeserialize_to_proto))

#### Se você não controla o pipeline de geração

Se você quiser ver como `tfds.features` são representados num `tf.train.Example`, você pode examinar isto no colab:

- Para traduzir `tfds.features` numa estrutura legível por humanos do `tf.train.Example`, você pode chamar `features.get_serialized_info()`.
- Para obter a especificação exata `FixedLenFeature`,... passada para `tf.io.parse_single_example`, você pode usar `spec = features.tf_example_spec`

Observação: se você estiver usando um conector de recursos personalizado, certifique-se de implementar `to_json_content` / `from_json_content` e testar com `self.assertFeature` (consulte [o guia do conector de recursos](https://www.tensorflow.org/datasets/features#create_your_own_tfdsfeaturesfeatureconnector) )

### Obtenha estatísticas das divisões

O TFDS requer saber o número exato de exemplos dentro de cada fragmento. Isto é necessário para recursos como `len(ds)` ou a [API subsplit](https://www.tensorflow.org/datasets/splits): `split='train[75%:]'` .

- Se você tiver essas informações, poderá criar explicitamente uma lista de `tfds.core.SplitInfo` e pular para a próxima seção:

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

- Se você não souber sobre essas informações, poderá calculá-las usando o script `compute_split_info.py` (ou em seu próprio script com `tfds.folder_dataset.compute_split_info`). Ele iniciará um pipeline de feixe que lerá todos os fragmentos num determinado diretório e calculará as informações.

### Adicione arquivos de metadados

Para adicionar automaticamente os arquivos de metadados adequados ao seu dataset, use `tfds.folder_dataset.write_metadata`:

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

Depois que a função for chamada uma vez na pasta do seru dataset, os arquivos de metadados (`dataset_info.json`,...) foram adicionados e seus datasets estão prontos para serem carregados com TFDS (veja a próxima seção).

## Carregue o dataset com TFDS

### Diretamente da pasta

Depois que os metadados forem gerados, os datasets podem ser carregados usando `tfds.builder_from_directory` que retorna um `tfds.core.DatasetBuilder` com a API TFDS padrão (como `tfds.builder`):

```python
builder = tfds.builder_from_directory('~/path/to/my_dataset/3.0.0/')

# Metadata are available as usual
builder.info.splits['train'].num_examples

# Construct the tf.data.Dataset pipeline
ds = builder.as_dataset(split='train[75%:]')
for ex in ds:
  ...
```

### Diretamente de múltiplas pastas

Também é possível carregar dados de múltiplas pastas. Isso pode acontecer, por exemplo, no aprendizado por reforço, quando vários agentes estão gerando, cada um, um dataset separado e você deseja carregar todos eles juntos. Outros casos de uso são quando um novo dataset é produzido regularmente, por exemplo, um novo dataset por dia, e você deseja carregar dados de um intervalo de datas.

Para carregar dados de múltiplas pastas, use `tfds.builder_from_directories`, que retorna um `tfds.core.DatasetBuilder` com a API TFDS padrão (como `tfds.builder`):

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

Observação: cada pasta deve ter seus próprios metadados, pois estes contêm informações sobre as divisões.

### Estrutura de pastas (opcional)

Para melhor compatibilidade com TFDS, você pode organizar seus dados como `<data_dir>/<dataset_name>[/<dataset_config>]/<dataset_version>`. Por exemplo:

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

Isso deixará seus datasets compatíveis com a API `tfds.load` / `tfds.builder`, simplesmente fornecendo `data_dir/`:

```python
ds0 = tfds.load('dataset0', data_dir='data_dir/')
ds1 = tfds.load('dataset1/config0', data_dir='data_dir/')
```
