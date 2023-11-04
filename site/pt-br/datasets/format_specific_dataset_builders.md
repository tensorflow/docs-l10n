# Construtores de datasets de formato específico

[TOC]

Este guia documenta todos os construtores de datasets de formato específico atualmente disponíveis no TFDS.

Construtores de datasets de formato específico são subclasses de [`tfds.core.GeneratorBasedBuilder`](https://www.tensorflow.org/datasets/api_docs/python/tfds/core/GeneratorBasedBuilder) que cuidam da maior parte do processamento de dados para um formato de dados específico.

## Datasets baseados em `tf.data.Dataset`

Se você quiser criar um dataset TFDS a partir de um dataset que está no formato `tf.data.Dataset` ([referência](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)), você pode usar `tfds.dataset_builders.TfDataBuilder` (veja a [Documentação da API](https://www.tensorflow.org/datasets/api_docs/python/tfds/dataset_builders/TfDataBuilder)).

Imaginamos dois usos típicos desta classe:

- Criação de datasets experimentais num ambiente semelhante a um notebook
- Definindo um construtor de datasets em código

### Criando um novo dataset a partir de um notebook

Suponha que você esteja trabalhando num notebook, carregou alguns dados como `tf.data.Dataset`, aplicou diversas transformações (mapa, filtro, etc) e agora deseja armazenar esses dados e compartilhá-los facilmente com colegas de equipe ou carregá-los em outros notebooks. Em vez de definir uma nova classe de construtor de datasets, você também pode instanciar um `tfds.dataset_builders.TfDataBuilder` e chamar `download_and_prepare` para armazenar seu dataset como um dataset TFDS.

Por ser um conjunto de dados TFDS, você pode versioná-lo, usar configurações, ter divisões (splits) diferentes e documentá-lo para facilitar o uso posterior. Isto significa que você também precisa informar ao TFDS quais são as características do seu dataset.

Eis um exemplo fictício de como você poderia usá-lo.

```python
import tensorflow as tf
import tensorflow_datasets as tfds

my_ds_train = tf.data.Dataset.from_tensor_slices({"number": [1, 2, 3]})
my_ds_test = tf.data.Dataset.from_tensor_slices({"number": [4, 5]})

# Optionally define a custom `data_dir`.
# If None, then the default data dir is used.
custom_data_dir = "/my/folder"

# Define the builder.
single_number_builder = tfds.dataset_builders.TfDataBuilder(
    name="my_dataset",
    config="single_number",
    version="1.0.0",
    data_dir=custom_data_dir,
    split_datasets={
        "train": my_ds_train,
        "test": my_ds_test,
    },
    features=tfds.features.FeaturesDict({
        "number": tfds.features.Scalar(dtype=tf.int64),
    }),
    description="My dataset with a single number.",
    release_notes={
        "1.0.0": "Initial release with numbers up to 5!",
    }
)

# Make the builder store the data as a TFDS dataset.
single_number_builder.download_and_prepare()
```

O método `download_and_prepare` irá iterar sobre os `tf.data.Dataset` de entrada e armazenarrá o dataset TFDS correspondente em `/my/folder/my_dataset/single_number/1.0.0`, que conterá as divisões de treinamento e teste.

O argumento `config` é opcional e pode ser útil se você quiser armazenar configurações diferentes no mesmo dataset.

O argumento `data_dir` pode ser usado para armazenar o dataset TFDS gerado numa pasta diferente, por exemplo, na sua própria sandbox, se você não quiser compartilhar isso com outras pessoas (ainda). Observe que ao fazer isso, você também precisa passar `data_dir` para `tfds.load`. Se o argumento `data_dir` não for especificado, o diretório de dados TFDS padrão será usado.

#### Carregando seu dataset

Depois que o dataset TFDS for armazenado, ele poderá ser carregado a partir de outros scripts ou por colegas de equipe se eles tiverem acesso aos dados:

```python
# If no custom data dir was specified:
ds_test = tfds.load("my_dataset/single_number", split="test")

# When there are multiple versions, you can also specify the version.
ds_test = tfds.load("my_dataset/single_number:1.0.0", split="test")

# If the TFDS was stored in a custom folder, then it can be loaded as follows:
custom_data_dir = "/my/folder"
ds_test = tfds.load("my_dataset/single_number:1.0.0", split="test", data_dir=custom_data_dir)
```

#### Adicionando uma nova versão ou configuração

Depois de iterar ainda mais no seu dataset, você pode ter adicionado ou alterado algumas das transformações dos dados fonte. Para armazenar e compartilhar este dataset, você pode armazená-lo facilmente como uma nova versão.

```python
def add_one(example):
  example["number"] = example["number"] + 1
  return example

my_ds_train_v2 = my_ds_train.map(add_one)
my_ds_test_v2 = my_ds_test.map(add_one)

single_number_builder_v2 = tfds.dataset_builders.TfDataBuilder(
    name="my_dataset",
    config="single_number",
    version="1.1.0",
    data_dir=custom_data_dir,
    split_datasets={
        "train": my_ds_train_v2,
        "test": my_ds_test_v2,
    },
    features=tfds.features.FeaturesDict({
        "number": tfds.features.Scalar(dtype=tf.int64, doc="Some number"),
    }),
    description="My dataset with a single number.",
    release_notes={
        "1.1.0": "Initial release with numbers up to 6!",
        "1.0.0": "Initial release with numbers up to 5!",
    }
)

# Make the builder store the data as a TFDS dataset.
single_number_builder_v2.download_and_prepare()
```

### Definindo uma nova classe de construtor de dataset

Você também pode definir um novo `DatasetBuilder` baseado nesta classe.

```python
import tensorflow as tf
import tensorflow_datasets as tfds

class MyDatasetBuilder(tfds.dataset_builders.TfDataBuilder):
  def __init__(self):
    ds_train = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    ds_test = tf.data.Dataset.from_tensor_slices([4, 5])
    super().__init__(
        name="my_dataset",
        version="1.0.0",
        split_datasets={
            "train": ds_train,
            "test": ds_test,
        },
        features=tfds.features.FeaturesDict({
            "number": tfds.features.Scalar(dtype=tf.int64),
        }),
        config="single_number",
        description="My dataset with a single number.",
        release_notes={
            "1.0.0": "Initial release with numbers up to 5!",
        })
```

## CoNLL

### O formato

O [CoNLL](https://aclanthology.org/W03-0419.pdf) é um formato popular usado para representar dados de texto anotados.

Os dados formatados em CoNLL geralmente contêm um token com suas anotações linguísticas por linha; dentro da mesma linha, as anotações geralmente são separadas por espaços ou tabulações. As linhas vazias representam os limites das frases.

Considere como exemplo a seguinte frase do dataset [conll2003](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/conll2003/conll2003.py), que segue o formato de anotação CoNLL:

```markdown
U.N. NNP I-NP I-ORG official
NN I-NP O
Ekeus NNP I-NP I-PER
heads VBZ I-VP O
for IN I-PP O
Baghdad NNP I-NP
I-LOC . . O O
```

### `ConllDatasetBuilder`

Para adicionar um novo dataset baseado em CoNLL ao TFDS, você pode basear sua classe de construtor de datasets em `tfds.dataset_builders.ConllDatasetBuilder`. Esta classe base contém o código comum para lidar com as especificidades dos datasets CoNLL (iterando sobre o formato baseado em colunas, listas pré-compiladas de características e tags, ...).

O `tfds.dataset_builders.ConllDatasetBuilder` implementa um `GeneratorBasedBuilder` específico do CoNLL. Consulte a classe a seguir como um exemplo mínimo de um construtor de datasets CoNLL:

```python
from tensorflow_datasets.core.dataset_builders.conll import conll_dataset_builder_utils as conll_lib
import tensorflow_datasets.public_api as tfds

class MyCoNNLDataset(tfds.dataset_builders.ConllDatasetBuilder):
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'Initial release.'}

  # conllu_lib contains a set of ready-to-use CONLL-specific configs.
  BUILDER_CONFIGS = [conll_lib.CONLL_2003_CONFIG]

  def _info(self) -> tfds.core.DatasetInfo:
    return self.create_dataset_info(
        # ...
    )

  def _split_generators(self, dl_manager):
    path = dl_manager.download_and_extract('https://data-url')

    return {'train': self._generate_examples(path=path / 'train.txt'),
            'test': self._generate_examples(path=path / 'train.txt'),
    }
```

Quanto aos construtores de datasets padrão, é necessário substituir os métodos de classe `_info` e `_split_generators`. Dependendo do dataset, pode ser necessário atualizar também [conllu_dataset_builder_utils.py](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_builders/conll/conll_dataset_builder_utils.py) para incluir as características e a lista de tags específicas do seu dataset.

O método `_generate_examples` não deve exigir sobreposição adicional, a menos que seu dataset precise de alguma implementação específica.

### Exemplos

Considere o [conll2003](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/conll2003/conll2003.py) como um exemplo de dataset implementado usando o construtor de dataset específico do CoNLL.

### CLI

A maneira mais fácil de escrever um novo dataset baseado em CoNLL é usar o [TFDS CLI](https://www.tensorflow.org/datasets/cli):

```sh
cd path/to/my/project/datasets/
tfds new my_dataset --format=conll   # Create `my_dataset/my_dataset.py` CoNLL-specific template files
```

## CoNLL-U

### O formato

[CoNLL-U](https://universaldependencies.org/format.html) é um formato popular usado para representar dados de texto anotados.

O CoNLL-U aprimora o formato CoNLL adicionando uma série de recursos, como suporte para [palavras com múltiplos tokens](https://universaldependencies.org/u/overview/tokenization.html). Os dados formatados em CoNLL-U geralmente contêm um token com suas anotações linguísticas por linha; dentro da mesma linha, as anotações geralmente são separadas por caracteres de tabulação individuais. As linhas vazias representam os limites das frases.

Normalmente, cada linha de palavra anotada em CoNLL-U contém os seguintes campos, conforme relatado na [documentação oficial](https://universaldependencies.org/format.html):

- ID: índice da palavra, inteiro começando em 1 para cada nova frase; pode ser um intervalo para tokens multipalavras; pode ser um número decimal para nós vazios (os números decimais podem ser menores que 1, mas devem ser maiores que 0).
- FORM: forma da palavra ou símbolo de pontuação.
- LEMMA: lema ou radical da forma da palavra.
- UPOS: tag universal de classe gramatical.
- XPOS: tag de classe gramatical específica do idioma; sublinhado se não estiver disponível.
- FEATS: lista de características morfológicas do inventário de características universais ou de uma extensão definida, específica de um idioma; sublinhado se não estiver disponível.
- HEAD: cabeçalho da palavra atual, que é um valor de ID ou zero (0).
- DEPREL: relação de dependência universal com o HEAD (raiz, se HEAD = 0) ou um subtipo definido de um idioma específico.
- DEPS: grafo de dependências aprimorado na forma de uma lista de pares head-deprel.
- MISC: qualquer outra anotação.

Considere como exemplo a seguinte frase, anotada com CoNLL-U, da [documentação oficial](https://universaldependencies.org/format.html):

```markdown
1-2    vámonos   _
1      vamos     ir
2      nos       nosotros
3-4    al        _
3      a         a
4      el        el
5      mar       mar
```

### `ConllUDatasetBuilder`

Para adicionar um novo dataset baseado em CoNLL-U ao TFDS, você pode basear sua classe de construtor de datasets em `tfds.dataset_builders.ConllUDatasetBuilder`. Esta classe base contém o código comum para lidar com as especificidades dos datasets CoNLL-U (iterando sobre o formato baseado em colunas, listas pré-compiladas de características e tags, ...).

O `tfds.dataset_builders.ConllUDatasetBuilder` implementa um `GeneratorBasedBuilder` específico do CoNLL-U. Consulte a classe a seguir como um exemplo mínimo de um construtor de datasets CoNLL-U:

```python
from tensorflow_datasets.core.dataset_builders.conll import conllu_dataset_builder_utils as conllu_lib
import tensorflow_datasets.public_api as tfds

class MyCoNNLUDataset(tfds.dataset_builders.ConllUDatasetBuilder):
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'Initial release.'}

  # conllu_lib contains a set of ready-to-use features.
  BUILDER_CONFIGS = [
      conllu_lib.get_universal_morphology_config(
          language='en',
          features=conllu_lib.UNIVERSAL_DEPENDENCIES_FEATURES,
      )
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    return self.create_dataset_info(
        # ...
    )

  def _split_generators(self, dl_manager):
    path = dl_manager.download_and_extract('https://data-url')

    return {
        'train':
            self._generate_examples(
                path=path / 'train.txt',
                # If necessary, add optional custom processing (see conllu_lib
                # for examples).
                # process_example_fn=...,
            )
    }
```

Quanto aos construtores de datasets padrão, é necessário substituir os métodos de classe `_info` e `_split_generators`. Dependendo do dataset, pode ser necessário atualizar também [conllu_dataset_builder_utils.py](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_builders/conll/conllu_dataset_builder_utils.py) para incluir as características e a lista de tags específicas do seu dataset.

O método `_generate_examples` não deve exigir sobreposição adicional, a menos que seu dataset precise de alguma implementação específica. Observe que, se o seu dataset exigir pré-processamento específico - por exemplo, se considerar [características de dependência universal](https://universaldependencies.org/guidelines.html) não clássicas - você talvez tenha que alterar o atributo `process_example_fn` da sua função [`generate_examples`](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_builders/conll/conllu_dataset_builder.py#L192) (veja o dataset [xtreme_pos](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/universal_dependencies/xtreme_pos.py) como exemplo).

### Exemplos

Considere os seguintes datasets, que usam o construtor de dataset específico CoNNL-U, como exemplos:

- [universal_dependencies](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/universal_dependencies/universal_dependencies.py)
- [xtreme_pos](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/universal_dependencies/xtreme_pos.py)

### CLI

A maneira mais fácil de escrever um novo dataset baseado em CoNLL-U é usar o [TFDS CLI](https://www.tensorflow.org/datasets/cli):

```sh
cd path/to/my/project/datasets/
tfds new my_dataset --format=conllu   # Create `my_dataset/my_dataset.py` CoNLL-U specific template files
```
