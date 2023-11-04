# Versionamento de datasets

## Definição

O versionamento pode referir-se a diferentes significados:

- A versão da API TFDS (versão pip): `tfds.__version__`
- A versão do dataset público, independente do TFDS (por exemplo, [Voc2007](https://pjreddie.com/projects/pascal-voc-dataset-mirror/), Voc2012). No TFDS, cada versão do dataset público deve ser implementada como um dataset independente:
    - Ou através das [configurações do construtor](https://www.tensorflow.org/datasets/add_dataset#dataset_configurationvariants_tfdscorebuilderconfig): por exemplo `voc/2007`, `voc/2012`
    - Ou como dois datsets independentes: por exemplo, `wmt13_translate`, `wmt14_translate`
- A versão do código de geração do dataset no TFDS (`my_dataset:1.0.0`): Por exemplo, se um bug for encontrado na implementação TFDS de `voc/2007`, o código de geração `voc.py` será atualizado (`voc/2007:1.0.0` - &gt; `voc/2007:2.0.0` .

O restante deste guia concentra-se apenas na última definição (versão do código do dataset no repositório TFDS).

## Versões suportadas

Como regra geral:

- Somente a última versão atual pode ser gerada.
- Todos os datasets gerados anteriormente podem ser lidos (observação: isso requer datasets gerados com TFDS 4+).

```python
builder = tfds.builder('my_dataset')
builder.info.version  # Current version is: '2.0.0'

# download and load the last available version (2.0.0)
ds = tfds.load('my_dataset')

# Explicitly load a previous version (only works if
# `~/tensorflow_datasets/my_dataset/1.0.0/` already exists)
ds = tfds.load('my_dataset:1.0.0')
```

## Semântica

Cada `DatasetBuilder` definido no TFDS vem com uma versão, por exemplo:

```python
class MNIST(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('2.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release',
      '2.0.0': 'Update dead download url',
  }
```

A versão segue a regra [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html): `MAJOR.MINOR.PATCH`. O objetivo da versão é garantir a reprodutibilidade: carregar um determinado dataset numa versão fixa produz os mesmos dados. Mais especificamente:

- Se a versão `PATCH` for incrementada, os dados lidos pelo cliente serão os mesmos, embora os dados possam ser serializados de forma diferente no disco ou os metadados possam ter sido alterados. Para qualquer fatia, a API de fatiamento retorna o mesmo conjunto de registros.
- Se a versão `MINOR` for incrementada, os dados existentes lidos pelo cliente são os mesmos, mas há dados adicionais (recursos em cada registro). Para qualquer fatia, a API de fatiamento retorna o mesmo conjunto de registros.
- Se a versão `MAJOR` for incrementada, os dados existentes foram alterados e/ou a API de fatiamento não retorna necessariamente o mesmo conjunto de registros para uma determinada fatia.

Quando uma alteração de código é feita na biblioteca TFDS e essa alteração de código afeta a maneira como um dataset está sendo serializado e/ou lido pelo cliente, a versão do construtor correspondente é incrementada de acordo com as diretrizes acima.

Observe que a semântica acima é o melhor esforço e pode haver bugs despercebidos afetando um dataset enquanto a versão não foi incrementada. Esses bugs serão eventualmente corrigidos, mas se você depende muito do controle de versão, recomendamos usar o TFDS de uma versão já lançada (em vez de `HEAD`).

Observe também que alguns datasets possuem outro esquema de versionamento independente da versão do TFDS. Por exemplo, o dataset Open Images possui várias versões e, no TFDS, os construtores correspondentes são `open_images_v4`, `open_images_v5`, ...

## Carregando uma versão específica

Ao carregar um dataset ou `DatasetBuilder`, você pode especificar a versão a ser usada. Por exemplo:

```python
tfds.load('imagenet2012:2.0.1')
tfds.builder('imagenet2012:2.0.1')

tfds.load('imagenet2012:2.0.0')  # Error: unsupported version.

# Resolves to 3.0.0 for now, but would resolve to 3.1.1 if when added.
tfds.load('imagenet2012:3.*.*')
```

Se estiver usando o TFDS para uma publicação, aconselhamos que:

- **mantenha fixo apenas o componente `MAJOR` da versão**;
- **anuncie qual versão do dataset foi usada em seus resultados.**

Isso tornará mais fácil para você mesmo no futuro, seus leitores e revisores, reproduzirem seus resultados.

## BUILDER_CONFIGS e versões

Alguns datasets definem vários `BUILDER_CONFIGS`. Quando for esse o caso, `version` e `supported_versions` são definidos nos próprios objetos de configuração. Fora isso, a semântica e o uso são idênticos. Por exemplo:

```python
class OpenImagesV4(tfds.core.GeneratorBasedBuilder):

  BUILDER_CONFIGS = [
      OpenImagesV4Config(
          name='original',
          version=tfds.core.Version('0.2.0'),
          supported_versions=[
            tfds.core.Version('1.0.0', "Major change in data"),
          ],
          description='Images at their original resolution and quality.'),
      ...
  ]

tfds.load('open_images_v4/original:1.*.*')
```

## Versão experimental

Observação: O que se segue é uma prática não recomendada, propensa a erros e que deve ser desencorajada.

É possível permitir a geração de duas versões ao mesmo tempo. Uma versão padrão e uma versão experimental. Por exemplo:

```python
class MNIST(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version("1.0.0")  # Default version
  SUPPORTED_VERSIONS = [
      tfds.core.Version("2.0.0"),  # Experimental version
  ]


# Download and load default version 1.0.0
builder = tfds.builder('mnist')

#  Download and load experimental version 2.0.0
builder = tfds.builder('mnist', version='experimental_latest')
```

No código, você precisa ter certeza de oferecer suporte às duas versões:

```python
class MNIST(tfds.core.GeneratorBasedBuilder):

  ...

  def _generate_examples(self, path):
    if self.info.version >= '2.0.0':
      ...
    else:
      ...
```
