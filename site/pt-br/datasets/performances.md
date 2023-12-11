# Dicas de desempenho

Este documento fornece dicas de desempenho específicas do TensorFlow Datasets (TFDS). Observe que o TFDS fornece datasets como objetos `tf.data.Dataset`, portanto, a recomendação do [guia `tf.data`](https://www.tensorflow.org/guide/data_performance#optimize_performance) ainda se aplica.

## Benchmark de datasets

Use `tfds.benchmark(ds)` para fazer um benchmark de qualquer objeto `tf.data.Dataset`.

Não deixe de indicar `batch_size=` para normalizar os resultados (por exemplo, 100 iter/seg -&gt; 3200 ex/seg). Isto funciona com qualquer iterável (por exemplo `tfds.benchmark(tfds.as_numpy(ds))`).

```python
ds = tfds.load('mnist', split='train').batch(32).prefetch()
# Display some benchmark statistics
tfds.benchmark(ds, batch_size=32)
# Second iteration is much faster, due to auto-caching
tfds.benchmark(ds, batch_size=32)
```

## Pequenos datasets (menos de 1 GB)

Todos os datasets TFDS armazenam os dados em disco no formato [`TFRecord`](https://www.tensorflow.org/tutorials/load_data/tfrecord). Para datasets pequenos (por exemplo, MNIST, CIFAR-10/-100), a leitura de `.tfrecord` pode acrescentar uma sobrecarga significativa.

Como esses datasets cabem na memória, é possível melhorar significativamente o desempenho armazenando previamente em cache ou pré-carregando o dataset. Observe que o TFDS armazena automaticamente em cache pequenos datasets (a seção a seguir contém os detalhes).

### Armazenando o dataset em cache

Aqui está um exemplo de pipeline de dados que armazena explicitamente o dataset em cache após normalizar as imagens.

```python
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


ds, ds_info = tfds.load(
    'mnist',
    split='train',
    as_supervised=True,  # returns `(img, label)` instead of dict(image=, ...)
    with_info=True,
)
# Applying normalization before `ds.cache()` to re-use it.
# Note: Random transformations (e.g. images augmentations) should be applied
# after both `ds.cache()` (to avoid caching randomness) and `ds.batch()` (for
# vectorization [1]).
ds = ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.cache()
# For true randomness, we set the shuffle buffer to the full dataset size.
ds = ds.shuffle(ds_info.splits['train'].num_examples)
# Batch after shuffling to get unique batches at each epoch.
ds = ds.batch(128)
ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
```

- [[1] Vetorização do mapeamento](https://www.tensorflow.org/guide/data_performance#vectorizing_mapping)

Ao iterar sobre este dataset, a segunda iteração será muito mais rápida que a primeira graças ao cache.

### Cache automático

Por padrão, o TFDS armazena em cache automático (com `ds.cache()`) datasets que satisfazem as seguintes restrições:

- O tamanho total do dataset (todas as divisões) é definido e menor que 250 MiB
- `shuffle_files` está desativado ou apenas um único fragmento é lido

É possível desativar o cache automático passando `try_autocaching=False` para `tfds.ReadConfig` em `tfds.load`. Dê uma olhada na documentação do catálogo do dataset para ver se um dataset específico usará cache automático.

### Carregando os dados completos como um único Tensor

Se o seu dataset couber na memória, você também pode carregar o dataset completo como um único array NumPy ou Tensor. É possível fazer isso definindo `batch_size=-1` para agrupar todos os exemplos num único `tf.Tensor`. Em seguida, use `tfds.as_numpy` para a conversão de `tf.Tensor` para `np.array`.

```python
(img_train, label_train), (img_test, label_test) = tfds.as_numpy(tfds.load(
    'mnist',
    split=['train', 'test'],
    batch_size=-1,
    as_supervised=True,
))
```

## Grandes datasets

Grandes dataset são fragmentados (divididos em múltiplos arquivos) e normalmente não cabem na memória, portanto, não devem ser armazenados em cache.

### Embaralhamento e treinamento

Durante o treinamento, é importante embaralhar bem os dados – dados mal embaralhados podem resultar em menor precisão do treinamento.

Além de usar `ds.shuffle` para embaralhar registros, você também deve definir `shuffle_files=True` para obter um bom comportamento de embaralhamento para datasets maiores que são fragmentados em múltiplos arquivos. Caso contrário, as épocas lerão os fragmentos na mesma ordem e, portanto, os dados não serão verdadeiramente aleatórios.

```python
ds = tfds.load('imagenet2012', split='train', shuffle_files=True)
```

Além disso, quando `shuffle_files=True`, o TFDS desativa [`options.deterministic`](https://www.tensorflow.org/api_docs/python/tf/data/Options#deterministic), o que pode proporcionar um ligeiro aumento de desempenho. Para conseguir embaralhamento determinístico, é possível desativar esse recurso com `tfds.ReadConfig`: configurando `read_config.shuffle_seed` ou substituindo `read_config.options.deterministic`.

### Fragmente automaticamente seus dados entre workers (TF)

Ao treinar múltiplos workers, você pode usar o argumento `input_context` de `tfds.ReadConfig`, para que cada worker leia um subconjunto dos dados.

```python
input_context = tf.distribute.InputContext(
    input_pipeline_id=1,  # Worker id
    num_input_pipelines=4,  # Total number of workers
)
read_config = tfds.ReadConfig(
    input_context=input_context,
)
ds = tfds.load('dataset', split='train', read_config=read_config)
```

Isso é complementar à API subsplit. Primeiro, a API subsplit é aplicada: `train[:50%]` é convertido numa lista de arquivos para leitura. Em seguida, uma operação `ds.shard()` é aplicada a esses arquivos. Por exemplo, ao usar `train[:50%]` com `num_input_pipelines=2`, cada um dos 2 workers lerá 1/4 dos dados.

Quando `shuffle_files=True`, os arquivos são embaralhados dentro de um worker, mas não entre workers. Cada worker lerá o mesmo subconjunto de arquivos entre épocas.

Observação: Ao usar `tf.distribute.Strategy`, o `input_context` pode ser criado automaticamente com [distribui_datasets_from_function](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy#distribute_datasets_from_function)

### Fragmente automaticamente seus dados entre workers (Jax)

Com o Jax, você pode usar a API `tfds.split_for_jax_process` ou `tfds.even_splits` para distribuir seus dados entre workers. Consulte o [guia da API de divisões](https://www.tensorflow.org/datasets/splits).

```python
split = tfds.split_for_jax_process('train', drop_remainder=True)
ds = tfds.load('my_dataset', split=split)
```

`tfds.split_for_jax_process` é um alias simples para:

```python
# The current `process_index` loads only `1 / process_count` of the data.
splits = tfds.even_splits('train', n=jax.process_count(), drop_remainder=True)
split = splits[jax.process_index()]
```

### Decodificação de imagens mais rápida

Por padrão, o TFDS decodifica imagens automaticamente. No entanto, há casos em que pode ser mais eficiente pular a decodificação da imagem com `tfds.decode.SkipDecoding` e aplicar manualmente a operação `tf.io.decode_image`:

- Ao filtrar exemplos (com `tf.data.Dataset.filter`), para decodificar imagens após a filtragem dos exemplos.
- Ao recortar imagens, use a operação combinada `tf.image.decode_and_crop_jpeg`.

O código para ambos os exemplos está disponível no [guia de decodificação](https://www.tensorflow.org/datasets/decode#usage_examples).

### Ignore características não utilizadas

Se você estiver usando apenas um subconjunto das características, é possível ignorar completamente algumas delas. Se o seu dataset tiver muitas características não utilizadas, não decodificá-las poderá melhorar significativamente o desempenho. Veja https://www.tensorflow.org/datasets/decode#only_decode_a_sub-set_of_the_features.
