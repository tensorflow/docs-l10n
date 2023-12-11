# Gerando grandes datasets com o Apache Beam

Alguns datasets são grandes demais para serem processados ​​numa única máquina. `tfds` oferece suporte à geração de dados em muitas máquinas usando o [Apache Beam](https://beam.apache.org/).

Este documento tem duas seções:

- Para usuários que desejam gerar um dataset Beam existente
- Para desenvolvedores que desejam criar um novo dataset Beam

## Gerando um dataset Beam

Abaixo estão diferentes exemplos de geração de um dataset Beam, tanto na nuvem quanto localmente.

**Importante**: ao gerar o dataset com a [CLI do `tfds build`](https://www.tensorflow.org/datasets/cli#tfds_build_download_and_prepare_a_dataset), certifique-se de especificar a configuração do dataset que deseja gerar ou o padrão será gerar todas as configurações existentes. Por exemplo, para [wikipedia](https://www.tensorflow.org/datasets/catalog/wikipedia), use `tfds build wikipedia/20200301.en` em vez de `tfds build wikipedia`.

### No Google Cloud Dataflow

Para executar o pipeline usando o [Google Cloud Dataflow](https://cloud.google.com/dataflow/) e aproveitar a computação distribuída, primeiro siga as [Instruções de início rápido](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python).

Assim que seu ambiente estiver configurado, você poderá executar a [CLI do `tfds build`](https://www.tensorflow.org/datasets/cli#tfds_build_download_and_prepare_a_dataset) usando um diretório de dados no [GCS](https://cloud.google.com/storage/) e especificando as [opções necessárias](https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#configuring-pipelineoptions-for-execution-on-the-cloud-dataflow-service) para a flag `--beam_pipeline_options`.

Para facilitar o lançamento do script, é útil definir as seguintes variáveis ​​usando os valores reais da configuração do GCP/GCS e do dataset que você deseja gerar:

```sh
DATASET_NAME=<dataset-name>
DATASET_CONFIG=<dataset-config>
GCP_PROJECT=my-project-id
GCS_BUCKET=gs://my-gcs-bucket
```

Em seguida, você precisará criar um arquivo para instruir o Dataflow a instalar `tfds` nos workers:

```sh
echo "tensorflow_datasets[$DATASET_NAME]" > /tmp/beam_requirements.txt
```

Se você estiver usando `tfds-nightly`, não deixe de copiar de `tfds-nightly` caso o dataset tenha sido atualizado desde o último lançamento.

```sh
echo "tfds-nightly[$DATASET_NAME]" > /tmp/beam_requirements.txt
```

Finalmente, você pode iniciar a tarefa usando o comando abaixo:

```sh
tfds build $DATASET_NAME/$DATASET_CONFIG \
  --data_dir=$GCS_BUCKET/tensorflow_datasets \
  --beam_pipeline_options=\
"runner=DataflowRunner,project=$GCP_PROJECT,job_name=$DATASET_NAME-gen,"\
"staging_location=$GCS_BUCKET/binaries,temp_location=$GCS_BUCKET/temp,"\
"requirements_file=/tmp/beam_requirements.txt"
```

### Localmente

Para executar seu script localmente usando o [executor padrão do Apache Beam](https://beam.apache.org/documentation/runners/direct/) (ele deve caber todos os dados na memória), o comando é o mesmo que para outros datasets:

```sh
tfds build my_dataset
```

**Importante**: os datasets Beam podem ser **enormes** (terabytes ou mais) e exigir uma quantidade significativa de recursos para serem gerados (pode levar semanas num computador local). Recomenda-se gerar os datasets usando um ambiente distribuído. Dê uma olhada na [documentação do Apache Beam](https://beam.apache.org/) para uma lista de runtimes suportados.

### Com o Apache Flink

Para executar o pipeline usando o [Apache Flink](https://flink.apache.org/) você pode ler a [documentação oficial](https://beam.apache.org/documentation/runners/flink). Certifique-se de que seu Beam esteja em conformidade com a [Compatibilidade de Versão do Flink](https://beam.apache.org/documentation/runners/flink/#flink-version-compatibility)

Para facilitar o lançamento do script, é útil definir as seguintes variáveis ​​usando os valores reais da configuração do Flink e do dataset que você deseja gerar:

```sh
DATASET_NAME=<dataset-name>
DATASET_CONFIG=<dataset-config>
FLINK_CONFIG_DIR=<flink-config-directory>
FLINK_VERSION=<flink-version>
```

Para executar num cluster Flink incorporado, você pode iniciar o trabalho usando o comando abaixo:

```sh
tfds build $DATASET_NAME/$DATASET_CONFIG \
  --beam_pipeline_options=\
"runner=FlinkRunner,flink_version=$FLINK_VERSION,flink_conf_dir=$FLINK_CONFIG_DIR"
```

### Com um script personalizado

Para gerar o dataset no Beam, a API é a mesma de outros datasets. Você pode personalizar o [`beam.Pipeline`](https://beam.apache.org/documentation/programming-guide/#creating-a-pipeline) usando os argumentos `beam_options` (e `beam_runner` ) de `DownloadConfig`.

```python
# If you are running on Dataflow, Spark,..., you may have to set-up runtime
# flags. Otherwise, you can leave flags empty [].
flags = ['--runner=DataflowRunner', '--project=<project-name>', ...]

# `beam_options` (and `beam_runner`) will be forwarded to `beam.Pipeline`
dl_config = tfds.download.DownloadConfig(
    beam_options=beam.options.pipeline_options.PipelineOptions(flags=flags)
)
data_dir = 'gs://my-gcs-bucket/tensorflow_datasets'
builder = tfds.builder('wikipedia/20190301.en', data_dir=data_dir)
builder.download_and_prepare(download_config=dl_config)
```

## Implementando um dataset Beam

### Pré-requisitos

Para escrever datasets do Apache Beam, você precisa estar familiarizado com os seguintes conceitos:

- Familiarize-se com o [guia de criação de datasets `tfds`](https://github.com/tensorflow/datasets/blob/master/docs/add_dataset.md), pois a maior parte do conteúdo ainda se aplica aos datasets do Beam.
- Obtenha uma introdução ao Apache Beam com o [Guia de programação do Beam](https://beam.apache.org/documentation/programming-guide/).
- Se você quiser gerar seu dataset usando o Cloud Dataflow, leia a [documentação do Google Cloud](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python) e o [Guia de dependências do Apache Beam](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/).

### Instruções

Se você estiver familiarizado com o [guia de criação de datasets](https://github.com/tensorflow/datasets/blob/master/docs/add_dataset.md), adicionar um dataset Beam requer apenas a modificação da função `_generate_examples`. A função deve retornar um objeto Beam, em vez de um gerador:

Dataset não Beam:

```python
def _generate_examples(self, path):
  for f in path.iterdir():
    yield _process_example(f)
```

Dataset Beam:

```python
def _generate_examples(self, path):
  return (
      beam.Create(path.iterdir())
      | beam.Map(_process_example)
  )
```

Todo o resto pode ser 100% idêntico, incluindo os testes.

Algumas considerações adicionais:

- Use `tfds.core.lazy_imports` para importar o Apache Beam. Ao usar uma dependência lazy, os usuários ainda poderão ler o dataset após ele ter sido gerado, sem precisar instalar o Beam.
- Tenha cuidado com os closures do Python. Ao executar o pipeline, as funções `beam.Map` e `beam.DoFn` são serializadas usando `pickle` e enviadas a todos os workers. Não use objetos mutáveis ​​dentro de um `beam.PTransform` se o estado tiver que ser compartilhado entre workers.
- Devido à forma como `tfds.core.DatasetBuilder` é serializado com pickle, a mutação de `tfds.core.DatasetBuilder` durante a criação de dados será ignorada nos workers (por exemplo, não é possível definir `self.info.metadata['offset'] = 123` em `_split_generators` e acessá-lo dos workers como `beam.Map(lambda x: x + self.info.metadata['offset'])`
- Se você precisar compartilhar alguns passos do pipeline entre os splits (divisões), você pode adicionar um kwarg `pipeline: beam.Pipeline` para `_split_generator` e controlar o pipeline de geração completo. Consulte a documentação `_generate_examples` de `tfds.core.GeneratorBasedBuilder`.

### Exemplo

Aqui está um exemplo de um dataset Beam.

```python
class DummyBeamDataset(tfds.core.GeneratorBasedBuilder):

  VERSION = tfds.core.Version('1.0.0')

  def _info(self):
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(16, 16, 1)),
            'label': tfds.features.ClassLabel(names=['dog', 'cat']),
        }),
    )

  def _split_generators(self, dl_manager):
    ...
    return {
        'train': self._generate_examples(file_dir='path/to/train_data/'),
        'test': self._generate_examples(file_dir='path/to/test_data/'),
    }

  def _generate_examples(self, file_dir: str):
    """Generate examples as dicts."""
    beam = tfds.core.lazy_imports.apache_beam

    def _process_example(filename):
      # Use filename as key
      return filename, {
          'image': os.path.join(file_dir, filename),
          'label': filename.split('.')[1],  # Extract label: "0010102.dog.jpeg"
      }

    return (
        beam.Create(tf.io.gfile.listdir(file_dir))
        | beam.Map(_process_example)
    )

```

### Executando seu pipeline

Para executar o pipeline, dê uma olhada na seção acima.

**Observação**: assim como para datasets não Beam, não se esqueça de registrar checksums de download com `--register_checksums` (apenas na primeira vez para registrar os downloads).

```sh
tfds build my_dataset --register_checksums
```

## Pipeline usando TFDS como entrada

Se você deseja criar um pipeline do Beam que recebe um dataset TFDS como fonte, você pode usar `tfds.beam.ReadFromTFDS`:

```python
builder = tfds.builder('my_dataset')

_ = (
    pipeline
    | tfds.beam.ReadFromTFDS(builder, split='train')
    | beam.Map(tfds.as_numpy)
    | ...
)
```

Ele processará cada fragmento do dataset em paralelo.

Observação: isto exige que o dataset já tenha sido gerado. Para gerar datasets usando o Beam, veja as outras seções.
