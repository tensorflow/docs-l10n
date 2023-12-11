# Divisões e fatiamento

Todos os datasets TFDS expõem diversas divisões de dados (por exemplo, `'train'`, `'test'`) que podem ser exploradas no [catálogo](https://www.tensorflow.org/datasets/catalog/overview). Qualquer string alfabética pode ser usada como nome da divisão, exceto `all` (que é um termo reservado que corresponde à união de todas as divisões, veja abaixo).

Além das divisões "oficiais" do dataset, o TFDS permite selecionar fatia(s) de divisão(ões) e várias combinações.

## API de fatiamento

Instruções de fatiamento são especificadas em `tfds.load` ou `tfds.DatasetBuilder.as_dataset` através do kwarg `split=`.

```python
ds = tfds.load('my_dataset', split='train[:75%]')
```

```python
builder = tfds.builder('my_dataset')
ds = builder.as_dataset(split='test+train[:75%]')
```

Uma divisão (split) pode ser:

- **Nomes de divisão comuns** (uma string como `'train'`, `'test'`, ...): Todos os exemplos dentro da divisão selecionada.
- **Fatias**: As fatias têm a mesma semântica que [a notação de fatia python](https://docs.python.org/3/library/stdtypes.html#common-sequence-operations). Fatias podem ser:
    - **Absolutas** ( `'train[123:450]'`, `train[:4000]`): (veja a nota abaixo para a ressalva sobre a ordem de leitura)
    - **Porcentuais** ( `'train[:75%]'`, `'train[25%:75%]'`): divida os dados completos em fatias pares. Se os dados não forem divisíveis igualmente, alguma porcentagem poderá conter exemplos adicionais. Porcentagens fracionárias são suportadas.
    - **Fragmentos** (`train[:4shard]` , `train[4shard]`): selecione todos os exemplos no fragmento solicitado. (veja `info.splits['train'].num_shards` para obter o número de fragmentos da divisão)
- **União de divisões** (`'train+test'`, `'train[:25%]+test'`): as divisões serão intercaladas.
- **Dataset completo** (`'all'`): `'all'` é um nome de divisão especial correspondente à união de todas as divisões (equivalente a `'train+test+...'` ).
- **Lista de divisões** (`['train', 'test']`): Múltiplos `tf.data.Dataset` são retornados separadamente:

```python
# Returns both train and test split separately
train_ds, test_ds = tfds.load('mnist', split=['train', 'test[:50%]'])
```

Observação: devido à [intercalação](https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=nightly#interleave) dos fragmentos, não é garantido que a ordem seja consistente entre subdivisões. Em outras palavras, ler `test[0:100]` seguido por `test[100:200]` pode produzir exemplos em uma ordem diferente da leitura `test[:200]`. Veja o [guia de determinismo](https://www.tensorflow.org/datasets/determinism#determinism_when_reading) para entender a ordem em que o TFDS lê os exemplos.

## `tfds.even_splits` e treinamento multi-host

`tfds.even_splits` gera uma lista de subdivisões não sobrepostas do mesmo tamanho.

```python
# Divide the dataset into 3 even parts, each containing 1/3 of the data
split0, split1, split2 = tfds.even_splits('train', n=3)

ds = tfds.load('my_dataset', split=split2)
```

Isto pode ser útil ao treinar num ambiente distribuído, onde cada host deve receber uma fatia dos dados originais.

Com `Jax`, isto pode ser simplificado ainda mais usando `tfds.split_for_jax_process`:

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

`tfds.even_splits` , `tfds.split_for_jax_process` aceita qualquer valor de divisão como entrada (por exemplo `'train[75%:]+test'`)

## Fatiamento e metadados

É possível obter informações adicionais sobre as divisões/subdivisões (`num_examples`, `file_instructions`,...) usando o [dataset info](https://www.tensorflow.org/datasets/overview#access_the_dataset_metadata):

```python
builder = tfds.builder('my_dataset')
builder.info.splits['train'].num_examples  # 10_000
builder.info.splits['train[:75%]'].num_examples  # 7_500 (also works with slices)
builder.info.splits.keys()  # ['train', 'test']
```

## Validação cruzada

Exemplos de validação cruzada de 10 vezes usando a API string:

```python
vals_ds = tfds.load('mnist', split=[
    f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)
])
trains_ds = tfds.load('mnist', split=[
    f'train[:{k}%]+train[{k+10}%:]' for k in range(0, 100, 10)
])
```

Cada dataset de validação será 10%: `[0%:10%]`, `[10%:20%]`, ..., `[90%:100%]`. E cada dataset de treinamento será 90% complementar: `[10%:100%]` (para um conjunto de validação correspondente de `[0%:10%]`), `[0%:10%]

- [20%:100%]`(para um conjunto de validação de ` [10%:20%]`),...

## `tfds.core.ReadInstruction` e arredondamento

Em vez de `str`, é possível passar divisões como `tfds.core.ReadInstruction`:

Por exemplo, `split = 'train[50%:75%] + test'` é equivalente a:

```python
split = (
    tfds.core.ReadInstruction(
        'train',
        from_=50,
        to=75,
        unit='%',
    )
    + tfds.core.ReadInstruction('test')
)
ds = tfds.load('my_dataset', split=split)
```

`unit` pode ser:

- `abs`: fatiamento absoluto
- `%`: porcentagem de fatiamento
- `shard`: fatiamento de fragmento

`tfds.ReadInstruction` também possui um argumento de arredondamento. Se o número de exemplos no dataset não for dividido igualmente:

- `rounding='closest'` (padrão): os exemplos restantes são distribuídos dentro da porcentagem, portanto, alguma porcentagem poderá conter exemplos adicionais.
- `rounding='pct1_dropremainder'`: os exemplos restantes serão eliminados, mas isso garante que todas as porcentagens contenham exatamente o mesmo número de exemplos (por exemplo: `len(5%) == 5 * len(1%)`).

### Reprodutibilidade e determinismo

Durante a geração, para uma determinada versão de dataset, o TFDS garante que os exemplos sejam embaralhados deterministicamente no disco. Portanto, gerar o dataset duas vezes (em 2 computadores diferentes) não alterará a ordem dos exemplos.

Da mesma forma, a API subsplit (subdivisões) sempre selecionará o mesmo `set` de exemplos, independentemente da plataforma, arquitetura, etc. Isto significa `set('train[:20%]') == set('train[:10%]') + set('train[10%:20%]')`.

No entanto, a ordem em que os exemplos são lidos pode **não** ser determinística. Isto depende de outros parâmetros (por exemplo, se `shuffle_files=True`).
