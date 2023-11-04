# Escrevendo datasets personalizados

Siga este guia para criar um novo dataset (no TFDS ou no seu próprio repositório).

Confira nossa [lista de datasets](catalog/overview.md) para ver se o dataset que você procura já está presente.

## Resumo

A maneira mais fácil de escrever um novo dataset é usar o [TFDS CLI](https://www.tensorflow.org/datasets/cli):

```sh
cd path/to/my/project/datasets/
tfds new my_dataset  # Create `my_dataset/my_dataset.py` template files
# [...] Manually modify `my_dataset/my_dataset_dataset_builder.py` to implement your dataset.
cd my_dataset/
tfds build  # Download and prepare the dataset to `~/tensorflow_datasets/`
```

Para usar o novo dataset com `tfds.load('my_dataset')`:

- `tfds.load` vai automaticamente detectar e carregar o dataset gerado em `~/tensorflow_datasets/my_dataset/` (por exemplo, via `tfds build`).
- Alternativamente, você pode explicitamente importar (`import my.project.datasets.my_dataset`) para registrar seu dataset:

```python
import my.project.datasets.my_dataset  # Register `my_dataset`

ds = tfds.load('my_dataset')  # `my_dataset` registered
```

## Visão geral

Datasets são distribuídos em todos os tipos de formatos e em todos os tipos de lugares, e nem sempre são armazenados num formato que esteja pronto para alimentar um pipeline de aprendizado de máquina. Então temos o TFDS.

O TFDS processa esses datasets para um formato padrão (dados externos -&gt; arquivos serializados), que podem então ser carregados como pipeline de aprendizado de máquina (arquivos serializados -&gt; `tf.data.Dataset`). A serialização é feita apenas uma vez. O acesso subsequente vai ler esses arquivos pré-processados diretamente.

A maior parte do pré-processamento é feita automaticamente. Cada dataset implementa uma subclasse de `tfds.core.DatasetBuilder`, que especifica:

- De onde vêm os dados (ou seja, suas URLs);
- Como se apresenta o dataset (ou seja, suas características);
- Como os dados devem ser divididos (ex. `TRAIN` e `TEST`);
- e os exemplos individuais no dataset.

## Escreva seu dataset

### Modelo padrão: `tfds new`

Use o [TFDS CLI](https://www.tensorflow.org/datasets/cli) para gerar os arquivos modelo Python necessários.

```sh
cd path/to/project/datasets/  # Or use `--dir=path/to/project/datasets/` below
tfds new my_dataset
```

Este comando irá gerar uma nova pasta `my_dataset/` com a seguinte estrutura:

```sh
my_dataset/
    __init__.py
    README.md # Markdown description of the dataset.
    CITATIONS.bib # Bibtex citation for the dataset.
    TAGS.txt # List of tags describing the dataset.
    my_dataset_dataset_builder.py # Dataset definition
    my_dataset_dataset_builder_test.py # Test
    dummy_data/ # (optional) Fake data (used for testing)
    checksum.tsv # (optional) URL checksums (see `checksums` section).
```

Aqui, procure por `TODO(my_dataset)` aqui e modifique de acordo.

### Exemplo de dataset

Todos os datasets são subclasses implementadas de `tfds.core.DatasetBuilder`, que contém a maior parte do código basico. Essa classe suporta:

- Datasets pequenos/médios que podem ser gerados numa única máquina (este tutorial).
- Datasets muito grandes que requerem geração distribuída (usando o [Apache Beam](https://beam.apache.org/), veja nosso [guia para datasets gigantes](https://www.tensorflow.org/datasets/beam_datasets#implementing_a_beam_dataset))

Aqui está um exemplo mínimo de um construtor de dataset baseado em `tfds.core.GeneratorBasedBuilder`:

```python
class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata (homepage, citation,...)."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(256, 256, 3)),
            'label': tfds.features.ClassLabel(
                names=['no', 'yes'],
                doc='Whether this is a picture of a cat'),
        }),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    extracted_path = dl_manager.download_and_extract('http://data.org/data.zip')
    # dl_manager returns pathlib-like objects with `path.read_text()`,
    # `path.iterdir()`,...
    return {
        'train': self._generate_examples(path=extracted_path / 'train_images'),
        'test': self._generate_examples(path=extracted_path / 'test_images'),
    }

  def _generate_examples(self, path) -> Iterator[Tuple[Key, Example]]:
    """Generator of examples for each split."""
    for img_path in path.glob('*.jpeg'):
      # Yields (key, example)
      yield img_path.name, {
          'image': img_path,
          'label': 'yes' if img_path.name.startswith('yes_') else 'no',
      }
```

Observe que, para alguns formatos de dados específicos, fornecemos [construtores de datasets](https://www.tensorflow.org/datasets/format_specific_dataset_builders) prontos para uso para cuidar da maior parte do processamento de dados.

Vamos ver, em detalhes, os 3 métodos abstratos que precisamos sobrescrever.

### `_info` : metadados do dataset

`_info` retorna `tfds.core.DatasetInfo` contendo os [metadados do dataset](https://www.tensorflow.org/datasets/overview#access_the_dataset_metadata).

```python
def _info(self):
  # The `dataset_info_from_configs` base method will construct the
  # `tfds.core.DatasetInfo` object using the passed-in parameters and
  # adding: builder (self), description/citations/tags from the config
  # files located in the same package.
  return self.dataset_info_from_configs(
      homepage='https://dataset-homepage.org',
      features=tfds.features.FeaturesDict({
          'image_description': tfds.features.Text(),
          'image': tfds.features.Image(),
          # Here, 'label' can be 0-4.
          'label': tfds.features.ClassLabel(num_classes=5),
      }),
      # If there's a common `(input, target)` tuple from the features,
      # specify them here. They'll be used if as_supervised=True in
      # builder.as_dataset.
      supervised_keys=('image', 'label'),
      # Specify whether to disable shuffling on the examples. Set to False by default.
      disable_shuffling=False,
  )
```

A maior parte dos campos deve ser autoexplicativa. Algumas explicações:

- `features`: especifica a estrutura do dataset, formato, ... Suporta tipos de dados complexos (áudio, vídeo, sequências aninhadas, ...). Para mais informações, veja os guias [características disponíveis](https://www.tensorflow.org/datasets/api_docs/python/tfds/features#classes) ou o [guia do conector de características](https://www.tensorflow.org/datasets/features).
- `disable_shuffling`: Veja a seção [Mantenha a ordem do dataset](#maintain-dataset-order).

Escrevendo o arquivo `BibText` `CITATIONS.bib`:

- Pesquise no site do dataset por instruções de citação (use no formato BibTex).
- Para artigos [arXiv](https://arxiv.org/): encontre o artigo e clique no link `BibText` no lado direito.
- Encontre o artigo no [Google Scholar](https://scholar.google.com) e clique nas aspas duplas abaixo do título e no pop-up, clique em `BibTeX`.
- Se não houver nenhum documento associado (por exemplo, há apenas um site), você pode usar o [Editor Online do BibTeX](https://truben.no/latex/bibtex/) para criar uma entrada BibTeX personalizada (o menu suspenso tem um tipo de entrada `Online`).

Atualizando o arquivo `TAGS.txt`:

- Todas as tags permitidas são pré-preenchidas no arquivo gerado.
- Remova todas as tags que não se aplicam ao dataset.
- Tags válidas estão listadas em [tensorflow_datasets/core/valid_tags.txt](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/valid_tags.txt).
- Para acrescentar uma tag a essa lista, envie um pull request.

#### Mantenha a ordem do dataset

Por padrão, os registros dos datasets são embaralhados quando armazenados para tornar a distribuição das classes mais uniforme no dataset, uma vez que muitas vezes os registros pertencentes à mesma classe são contíguos. Para especificar que o dataset deve ser classificado pela chave gerada fornecida por `_generate_examples` o campo `disable_shuffling` deve ser definido como `True`. Por padrão, está definido como `False`.

```python
def _info(self):
  return self.dataset_info_from_configs(
    # [...]
    disable_shuffling=True,
    # [...]
  )
```

Tenha em mente que desabilitar o embaralhamento tem um impacto no desempenho, já que os fragmentos não podem mais ser lidos em paralelo.

### `_split_generators` : baixa e divide dados

#### Baixando e extraindo dados de fonte

A maioria dos datasets precisa baixar dados da web. Isto é feito usando o argumento de entrada `tfds.download.DownloadManager` de `_split_generators`. `dl_manager` possui os seguintes métodos:

- `download`: suporta `http(s)://`, `ftp(s)://`
- `extract`: atualmente suporta arquivos `.zip` , `.gz` e `.tar`.
- `download_and_extract`: O mesmo que `dl_manager.extract(dl_manager.download(urls))`

Todos esses métodos retornam `tfds.core.Path` (aliases para [`epath.Path`](https://github.com/google/etils)), que são objetos [semelhantes a pathlib.Path](https://docs.python.org/3/library/pathlib.html).

Esses métodos suportam estruturas aninhadas arbitrárias (`list`, `dict`), como:

```python
extracted_paths = dl_manager.download_and_extract({
    'foo': 'https://example.com/foo.zip',
    'bar': 'https://example.com/bar.zip',
})
# This returns:
assert extracted_paths == {
    'foo': Path('/path/to/extracted_foo/'),
    'bar': Path('/path/extracted_bar/'),
}
```

#### Download e extração manuais

Alguns dados não podem ser baixados automaticamente (por exemplo, requerem um login); neste caso, o usuário baixará manualmente os dados da fonte e os colocará em `manual_dir/` (o padrão é `~/tensorflow_datasets/downloads/manual/`).

Os arquivos podem depois ser acessados ​​através de `dl_manager.manual_dir`:

```python
class MyDataset(tfds.core.GeneratorBasedBuilder):

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Register into https://example.org/login to get the data. Place the `data.zip`
  file in the `manual_dir/`.
  """

  def _split_generators(self, dl_manager):
    # data_path is a pathlib-like `Path('<manual_dir>/data.zip')`
    archive_path = dl_manager.manual_dir / 'data.zip'
    # Extract the manually downloaded `data.zip`
    extracted_path = dl_manager.extract(archive_path)
    ...
```

A localização `manual_dir` pode ser personalizada com `tfds build --manual_dir=` ou usando `tfds.download.DownloadConfig`.

#### Leia o arquivo diretamente

`dl_manager.iter_archive` lê arquivos sequencialmente sem extraí-los. Isso pode economizar espaço de armazenamento e melhorar o desempenho em alguns sistemas de arquivos.

```python
for filename, fobj in dl_manager.iter_archive('path/to/archive.zip'):
  ...
```

`fobj` tem os mesmos métodos `with open('rb') as fobj:` (por exemplo, `fobj.read()`)

#### Especificando divisões em datasets

Se o dataset vier com divisões (splits) predefinidas (por exemplo, `MNIST` tem as divisões `train` e `test`), mantenha-as. Caso contrário, especifique apenas uma única divisão `all`. Os usuários podem criar dinamicamente suas próprias subdivisões com a [subsplit API](https://www.tensorflow.org/datasets/splits) (por exemplo `split='train[80%:]'`). Observe que qualquer string alfabética pode ser usada como nome de divisão, exceto o já mencionado `all` .

```python
def _split_generators(self, dl_manager):
  # Download source data
  extracted_path = dl_manager.download_and_extract(...)

  # Specify the splits
  return {
      'train': self._generate_examples(
          images_path=extracted_path / 'train_imgs',
          label_path=extracted_path / 'train_labels.csv',
      ),
      'test': self._generate_examples(
          images_path=extracted_path / 'test_imgs',
          label_path=extracted_path / 'test_labels.csv',
      ),
  }
```

### `_generate_examples`: gerador de exemplos

`_generate_examples` gera os exemplos para cada divisão dos dados de fonte.

Este método normalmente irá ler artefatos do dataset fonte (por exemplo, um arquivo CSV) e produzir tuplas `(key, feature_dict)`:

- `key` : identificador de exemplo. Usado para embaralhar deterministicamente os exemplos usando `hash(key)` ou para classificar por chave quando o embaralhamento estiver desativado (veja a seção [Mantenha a ordem do dataset](#maintain-dataset-order)). Deve ser:
    - **exclusiva**: se dois exemplos usarem a mesma chave, uma exceção será gerada.
    - **determinístic** : não deve depender da ordem `download_dir`, `os.path.listdir`,... Gerar os dados duas vezes deve produzir a mesma chave.
    - **comparável**: se o embaralhamento estiver desabilitado, a chave será usada para classificar o dataset.
- `feature_dict` : um `dict` contendo os valores de exemplo.
    - A estrutura deve corresponder à estrutura `features=` definida em `tfds.core.DatasetInfo`.
    - Tipos de dados complexos (imagem, vídeo, áudio,...) serão codificados automaticamente.
    - Cada característica geralmente aceita múltiplos tipos de entrada (por exemplo, vídeo aceita `/path/to/vid.mp4`, `np.array(shape=(l, h, w, c))`, `List[paths]`, `List[np.array(shape=(h, w, c)]`, `List[img_bytes]`,...)
    - Consulte o [guia do conector de características](https://www.tensorflow.org/datasets/features) para mais informações.

```python
def _generate_examples(self, images_path, label_path):
  # Read the input data out of the source files
  with label_path.open() as f:
    for row in csv.DictReader(f):
      image_id = row['image_id']
      # And yield (key, feature_dict)
      yield image_id, {
          'image_description': row['description'],
          'image': images_path / f'{image_id}.jpeg',
          'label': row['label'],
      }
```

Importante: ao processar valores booleanos de strings ou números inteiros, use a função util `tfds.core.utils.bool_utils.parse_bool` para evitar erros de processamento (por exemplo, `bool("False") == True`).

#### Acesso a arquivos e `tf.io.gfile`

Para oferecer suporte a sistemas de armazenamento em nuvem, evite o uso das operações de E/S nativas do Python.

Em vez disso, o `dl_manager` retorna objetos [similares a pathlib](https://docs.python.org/3/library/pathlib.html) diretamente compatíveis com o armazenamento do Google Cloud:

```python
path = dl_manager.download_and_extract('http://some-website/my_data.zip')

json_path = path / 'data/file.json'

json.loads(json_path.read_text())
```

Como alternativa, use a API `tf.io.gfile` em vez da API nativa para realizar operações com arquivos:

- `open` -&gt; `tf.io.gfile.GFile`
- `os.rename` -&gt; `tf.io.gfile.rename`
- ...

Deve-se dar preferência a Pathlib em vez de `tf.io.gfile` (veja [rational](https://www.tensorflow.org/datasets/common_gotchas#prefer_to_use_pathlib_api).

#### Dependências extras

Alguns datasets requerem dependências adicionais do Python apenas durante a geração. Por exemplo, o dataset SVHN usa `scipy` para carregar alguns dados.

Se você estiver adicionando um dataset ao repositório TFDS, use `tfds.core.lazy_imports` para manter o pacote `tensorflow-datasets` pequeno. Os usuários instalarão dependências adicionais somente conforme necessário.

Para usar `lazy_imports` :

- Adicione uma entrada para seu dataset em `DATASET_EXTRAS` em [`setup.py`](https://github.com/tensorflow/datasets/tree/master/setup.py). Isso permitirá que os usuários possam fazer, por exemplo, `pip install 'tensorflow-datasets[svhn]'` para instalar as dependências extras.
- Adicione uma entrada para sua importação ao [`LazyImporter`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/lazy_imports_lib.py) e ao [`LazyImportsTest`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/lazy_imports_lib_test.py).
- Use `tfds.core.lazy_imports` para acessar a dependência (por exemplo, `tfds.core.lazy_imports.scipy`) em seu `DatasetBuilder`.

#### Dados corrompidos

Alguns conjuntos de dados não estão perfeitamente limpos e contêm alguns dados corrompidos (por exemplo, as imagens estão em arquivos JPEG, mas algumas podem ser JPEGs inválidos). Esses exemplos devem ser ignorados, mas deixe uma nota na descrição do dataset sobre quantos exemplos foram descartados e por quê.

### Configuração/variantes do dataset (tfds.core.BuilderConfig)

Alguns datasets podem ter múltiplas variantes ou opções de como os dados são pré-processados ​​e gravados no disco. Por exemplo, [cycle_gan](https://www.tensorflow.org/datasets/catalog/cycle_gan) tem uma configuração para cada par de objetos (`cycle_gan/horse2zebra`, `cycle_gan/monet2photo`,...).

Isto é feito através de `tfds.core.BuilderConfig`:

1. Defina seu objeto de configuração como uma subclasse de `tfds.core.BuilderConfig`. Por exemplo, `MyDatasetConfig`.

    ```python
    @dataclasses.dataclass
    class MyDatasetConfig(tfds.core.BuilderConfig):
      img_size: Tuple[int, int] = (0, 0)
    ```

    Observação: Os valores padrão são obrigatórios devido a https://bugs.python.org/issue33129.

2. Defina o membro de classe `BUILDER_CONFIGS = []` em `MyDataset` que lista os `MyDatasetConfig` que o dataset expõe.

    ```python
    class MyDataset(tfds.core.GeneratorBasedBuilder):
      VERSION = tfds.core.Version('1.0.0')
      # pytype: disable=wrong-keyword-args
      BUILDER_CONFIGS = [
          # `name` (and optionally `description`) are required for each config
          MyDatasetConfig(name='small', description='Small ...', img_size=(8, 8)),
          MyDatasetConfig(name='big', description='Big ...', img_size=(32, 32)),
      ]
      # pytype: enable=wrong-keyword-args
    ```

    Observação: `# pytype: disable=wrong-keyword-args` é necessário devido ao [bug do Pytype](https://github.com/google/pytype/issues/628) com herança de classes de dados.

3. Use `self.builder_config` em `MyDataset` para configurar a geração de dados (por exemplo, `shape=self.builder_config.img_size`). Isto pode incluir a definição de valores diferentes em `_info()` ou a alteração do acesso aos dados de download.

Observações:

- Cada configuração tem um nome exclusivo. O nome totalmente qualificado de uma configuração é `dataset_name/config_name` (por exemplo, `coco/2017`).
- Se não for especificado, a primeira configuração em `BUILDER_CONFIGS` será usada (por exemplo `tfds.load('c4')` padrão para `c4/en`)

Veja [`anli`](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/anli.py#L69) para um exemplo de dataset que usa `BuilderConfig`.

### Versão

A versão pode ter dois significados diferentes:

- A versão dos dados originais "external": por exemplo, COCO v2019, v2017,...
- A versão "internal" do código TFDS: por exemplo, renomeia uma característica em `tfds.features.FeaturesDict`, corrige um bug em `_generate_examples`

Para atualizar um dataset:

- Para atualização de dados "external": vários usuários podem querer acessar um ano/versão específico simultaneamente. Isso é feito usando um `tfds.core.BuilderConfig` por versão (por exemplo, `coco/2017`, `coco/2019`) ou uma classe por versão (por exemplo, `Voc2007`, `Voc2012`).
- Para atualização de código "internal": os usuários baixam apenas a versão mais recente. Qualquer atualização de código deve incrementar o atributo da classe `VERSION` (por exemplo, de `1.0.0` para `VERSION = tfds.core.Version('2.0.0')`) após o [versionamento semântico](https://www.tensorflow.org/datasets/datasets_versioning#semantic).

### Adicione uma importação para registro

Não se esqueça de importar o módulo do dataset para o seu projeto `__init__` para ser registrado automaticamente em `tfds.load`, `tfds.builder`.

```python
import my_project.datasets.my_dataset  # Register MyDataset

ds = tfds.load('my_dataset')  # MyDataset available
```

Por exemplo, se você estiver contribuindo para `tensorflow/datasets`, adicione o módulo import ao `__init__.py` de seu subdiretório (por exemplo, [`image/__init__.py`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/image/__init__.py).

### Verifique para detectar problemas comuns de implementação

Dê uma olhada nos [problemas comuns de implementação](https://www.tensorflow.org/datasets/common_gotchas).

## Teste seu dataset

### Baixe e prepare: `tfds build`

Para gerar o dataset, execute `tfds build` no diretório `my_dataset/`:

```sh
cd path/to/datasets/my_dataset/
tfds build --register_checksums
```

Alguns sinalizadores úteis para desenvolvimento:

- `--pdb`: entra no modo de depuração se uma exceção for levantada.
- `--overwrite` : exclui os arquivos existentes se o dataset já tiver sido gerado.
- `--max_examples_per_split`: gera apenas os primeiros X exemplos (padrão 1), em vez do dataset completo.
- `--register_checksums`: registra checksums das URLs baixadas. Só deve ser usado durante o desenvolvimento.

Consulte a [documentação da CLI](https://www.tensorflow.org/datasets/cli#tfds_build_download_and_prepare_a_dataset) para uma lista completa de sinalizadores.

### Checksums

Recomenda-se registrar os checksums dos seus datasets para garantir o determinismo, ajudar com a documentação,... Isto é feito gerando o dataset com o `--register_checksums` (veja a seção anterior).

Se você estiver liberando seus dataset através do PyPI, não esqueça de exportar os arquivos `checksums.tsv` (por exemplo, no `package_data` do seu `setup.py`).

### Faça testes de unidade no seu dataset

`tfds.testing.DatasetBuilderTestCase` é um `TestCase` base para exercitar um dataset por completo. Ele usa "dados fictícios" como dados de teste que imitam a estrutura do dataset fonte.

- Os dados de teste devem ser colocados no diretório `my_dataset/dummy_data/` e devem imitar os artefatos do dataset fonte baixados e extraídos. Pode ser criado manualmente ou automaticamente com um script ([exemplo de script](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/datasets/bccd/dummy_data_generation.py)).
- Certifique-se de usar dados diferentes nas divisões de dados de teste, pois o teste falhará se as divisões do dataset se sobrepuserem.
- **Os dados de teste não devem conter nenhum material protegido por direitos autorais**. Em caso de dúvida, não crie os dados utilizando material do dataset original.

```python
import tensorflow_datasets as tfds
from . import my_dataset_dataset_builder


class MyDatasetTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for my_dataset dataset."""
  DATASET_CLASS = my_dataset_dataset_builder.Builder
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  DL_EXTRACT_RESULT = {
      'name1': 'path/to/file1',  # Relative to my_dataset/dummy_data dir.
      'name2': 'file2',
  }


if __name__ == '__main__':
  tfds.testing.test_main()
```

Execute o comando a seguir para testar o dataset.

```sh
python my_dataset_test.py
```

## Envie feedback

Tentamos continuamente melhorar o workflow de criação de datasets, mas só poderemos fazê-lo se estivermos cientes dos problemas. Quais problemas ou erros você encontrou ao criar o dataset? Houve alguma parte que foi confusa ou não funcionou de primeira?

Compartilhe seu feedback no [GitHub](https://github.com/tensorflow/datasets/issues).
