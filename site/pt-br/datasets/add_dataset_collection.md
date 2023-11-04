# Adicione uma nova coleção de datasets

Siga este guia para criar uma nova coleção de datasets (no TFDS ou no seu próprio repositório).

## Visão geral

Para adicionar uma nova coleção de datasets `my_collection` ao TFDS, os usuários precisam gerar uma pasta `my_collection` contendo os seguintes arquivos:

```sh
my_collection/
  __init__.py
  my_collection.py # Dataset collection definition
  my_collection_test.py # (Optional) test
  description.md # (Optional) collection description (if not included in my_collection.py)
  citations.md # (Optional) collection citations (if not included in my_collection.py)
```

Como convenção, novas coleções de datasets devem ser adicionadas à pasta `tensorflow_datasets/dataset_collections/` no repositório TFDS.

## Escreva sua coleção de datasets

Todas as coleções de datasets são subclasses implementadas de [`tfds.core.dataset_collection_builder.DatasetCollection`](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_collection_builder.py).

Aqui está um exemplo mínimo de um construtor de coleção de datasets, definido no arquivo `my_collection.py`:

```python
import collections
from typing import Mapping
from tensorflow_datasets.core import dataset_collection_builder
from tensorflow_datasets.core import naming

class MyCollection(dataset_collection_builder.DatasetCollection):
  """Dataset collection builder my_dataset_collection."""

  @property
  def info(self) -> dataset_collection_builder.DatasetCollectionInfo:
    return dataset_collection_builder.DatasetCollectionInfo.from_cls(
        dataset_collection_class=self.__class__,
        description="my_dataset_collection description.",
        release_notes={
            "1.0.0": "Initial release",
        },
    )

  @property
  def datasets(
      self,
  ) -> Mapping[str, Mapping[str, naming.DatasetReference]]:
    return collections.OrderedDict({
        "1.0.0":
            naming.references_for({
                "dataset_1": "natural_questions/default:0.0.2",
                "dataset_2": "media_sum:1.0.0",
            }),
        "1.1.0":
            naming.references_for({
                "dataset_1": "natural_questions/longt5:0.1.0",
                "dataset_2": "media_sum:1.0.0",
                "dataset_3": "squad:3.0.0"
            })
    })
```

As próximas seções descrevem os 2 métodos abstratos a serem sobrepostos.

### `info`: metadados da coleção de datasets

O método `info` retorna o [`dataset_collection_builder.DatasetCollectionInfo`](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/core/dataset_collection_builder.py#L66) contendo os metadados da coleção.

Um dataset collection info contém quatro campos:

- nome: o nome da coleção de datasets.
- descrição: uma descrição formatada em markdown da coleção de datasets. Existem duas maneiras de definir a descrição de uma coleção de datasets: (1) Como uma string (multilinhas) diretamente no arquivo `my_collection.py` da coleção - da mesma forma como já é feito para conjuntos de dados TFDS; (2) Em um arquivo `description.md`, que deve ser colocado na pasta da coleção de datasets.
- release_notes: um mapeamento da versão da coleção de datasets para as notas de lançamento correspondentes.
- citation: uma (lista de) citações opcionais do BibTeX para a coleção de datasets. Existem duas maneiras de definir a citação de uma coleção de datasets: (1) Como uma string (multilinhas) diretamente no arquivo `my_collection.py` da coleção - da mesma forma como já é feito para datasets TFDS; (2) Num arquivo `citations.bib`, que deve ser colocado na pasta da coleção de datasets.

### `datasets`: defina os datasets da coleção

O método `datasets` retorna os datasets TFDS na coleção.

É definido como um dicionário de versões, que descreve a evolução da coleção do dataset.

Para cada versão, os datasets TFDS incluídos são armazenados como um dicionário associando nomes dos datasets a [`naming.DatasetReference`](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/core/naming.py#L187). Por exemplo:

```python
class MyCollection(dataset_collection_builder.DatasetCollection):
  ...
  @property
  def datasets(self):
    return {
        "1.0.0": {
            "yes_no":
                naming.DatasetReference(
                    dataset_name="yes_no", version="1.0.0"),
            "sst2":
                naming.DatasetReference(
                    dataset_name="glue", config="sst2", version="2.0.0"),
            "assin2":
                naming.DatasetReference(
                    dataset_name="assin2", version="1.0.0"),
        },
        ...
    }
```

O método [`naming.references_for`](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/core/naming.py#L257) fornece uma maneira mais compacta de expressar o mesmo que o código acima:

```python
class MyCollection(dataset_collection_builder.DatasetCollection):
  ...
  @property
  def datasets(self):
    return {
        "1.0.0":
            naming.references_for({
                "yes_no": "yes_no:1.0.0",
                "sst2": "glue/sst:2.0.0",
                "assin2": "assin2:1.0.0",
            }),
        ...
    }
```

## Faça testes de unidade na sua coleção de datasets

A [DatasetCollectionTestBase](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/testing/dataset_collection_builder_testing.py#L28) é uma classe de teste base para coleções de datasets. Ela fornece uma série de verificações simples para garantir que a coleção de datasets esteja registrada corretamente e que seus datasets existam no TFDS.

O único atributo de classe a ser definido é `DATASET_COLLECTION_CLASS`, que especifica o objeto de classe da coleção de datasets a ser testado.

Além disso, os usuários podem definir os seguintes atributos de classe:

- `VERSION`: A versão da coleção de datasets usada para executar o teste (o padrão é a versão mais recente).
- `DATASETS_TO_TEST` : Lista contendo os datasets s serem testados quanto à existência no TFDS (o padrão é todos os datasets da coleção).
- `CHECK_DATASETS_VERSION`: se deve ou não verificar a existência dos datasets versionados na coleção de datasets ou suas versões padrão (o padrão é true).

O teste válido mais simples para uma coleção de datasets seria:

```python
from tensorflow_datasets.testing.dataset_collection_builder_testing import DatasetCollectionTestBase
from . import my_collection

class TestMyCollection(DatasetCollectionTestBase):
  DATASET_COLLECTION_CLASS = my_collection.MyCollection
```

Execute o comando a seguir para testar a coleção de datasets.

```sh
python my_dataset_test.py
```

## Feedback

Tentamos continuamente melhorar o workflow de criação de datasets, mas só poderemos fazê-lo se estivermos cientes dos problemas. Quais problemas ou erros você encontrou ao criar a coleção do dataset? Houve alguma parte que foi confusa ou não funcionou de primeira?

Compartilhe seu feedback no [GitHub](https://github.com/tensorflow/datasets/issues).
