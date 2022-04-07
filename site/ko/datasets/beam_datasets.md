# Apache Beam으로 빅 데이터세트 생성하기

일부 데이터세트는 너무 커서 단일 머신에서 처리할 수 없습니다. `tfds`는 [Apache Beam](https://beam.apache.org/)을 사용하여 많은 머신에서 데이터를 생성하도록 지원합니다.

이 문서에는 두 개의 섹션이 있습니다:

- 기존 Beam 데이터세트를 생성하려는 사용자
- 새로운 Beam 데이터세트를 생성하려는 개발자

## Beam 데이터세트 생성하기

다음은 클라우드 또는 로컬에서 Beam 데이터세트를 생성하는 서로 다른 예제입니다.

**경고**: [`tfds build` CLI](https://www.tensorflow.org/datasets/cli#tfds_build_download_and_prepare_a_dataset)를 사용하여 데이터세트를 생성할 때 생성하려는 데이터세트 구성을 지정해야 합니다. 그렇지 않으면 기본적으로 모든 기존 구성이 생성됩니다. 예를 들어, [위키피디아](https://www.tensorflow.org/datasets/catalog/wikipedia)의 경우에 `tfds build wikipedia` 대신 `tfds build wikipedia/20200301.en`를 사용합니다.

### Google Cloud Dataflow에서

[Google Cloud Dataflow](https://cloud.google.com/dataflow/)를 사용하여 파이프라인을 실행하고 분산 계산을 이용하려면, 먼저 [빠른 시작 지침](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python)을 따르세요.

환경이 설정되면 [GCS](https://cloud.google.com/storage/)의 데이터 디렉토리를 사용하고 `--beam_pipeline_options` 플래그에 [필요한 옵션](https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#configuring-pipelineoptions-for-execution-on-the-cloud-dataflow-service)을 지정하여 [`tfds build` CLI](https://www.tensorflow.org/datasets/cli#tfds_build_download_and_prepare_a_dataset)를 실행할 수 있습니다.

스크립트를 보다 쉽게 시작하려면 GCP/GCS 설정의 실제 값과 생성하려는 데이터세트를 사용하여 다음 변수를 정의하면 도움이 됩니다.

```sh
DATASET_NAME=<dataset-name>
DATASET_CONFIG=<dataset-config>
GCP_PROJECT=my-project-id
GCS_BUCKET=gs://my-gcs-bucket
```

Dataflow가 다음 작업자에서 `tfds`를 설치하도록 지시하는 파일을 작성해야 합니다.

```sh
echo "tensorflow_datasets[$DATASET_NAME]" > /tmp/beam_requirements.txt
```

`tfds-nightly`를 사용하는 경우, 마지막 릴리스 이후 데이터세트가 업데이트된 경우를 위해 `tfds-nightly`에서 echo해야 합니다.

```sh
echo "tfds-nightly[$DATASET_NAME]" > /tmp/beam_requirements.txt
```

마지막으로 아래 명령을 사용하여 작업을 실행할 수 있습니다.

```sh
python -m tensorflow_datasets.scripts.download_and_prepare \
  --datasets=$DATASET_NAME/$DATASET_CONFIG \
  --data_dir=$GCS_BUCKET/tensorflow_datasets \
  --beam_pipeline_options=\
"runner=DataflowRunner,project=$GCP_PROJECT,job_name=$DATASET_NAME-gen,"\
"staging_location=$GCS_BUCKET/binaries,temp_location=$GCS_BUCKET/temp,"\
"requirements_file=/tmp/beam_requirements.txt"
```

### 로컬에서

기본 Apache Beam 러너를 사용하여 스크립트를 로컬로 실행하기 위한 명령은 다른 데이터세트에서와 같습니다.

```sh
tfds build my_dataset
```

**경고**: Beam 데이터세트는 **매우** 클 수 있으며(테라바이트 이상), 상당한 양의 리소스가 생성될 수 있습니다(로컬 컴퓨터에서 몇 주가 걸릴 수 있음). 분산 환경을 사용하여 데이터세트를 생성하는 것이 좋습니다. 지원되는 런타임 목록은 [Apache Beam 설명서](https://beam.apache.org/)를 참조하세요.

### 사용자 정의 스크립트

Beam에서 데이터세트를 생성하기 위해 API는 다른 데이터세트의 경우와 동일합니다. `DownloadConfig`의 `beam_options`(및 `beam_runner`) 인수를 사용하여 [`beam.Pipeline`](https://beam.apache.org/documentation/programming-guide/#creating-a-pipeline)를 사용자 지정할 수 있습니다.

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

## Beam 데이터세트 구현하기

### 전제 조건

Apache Beam 데이터세트를 작성하려면 다음 개념에 익숙해야 합니다.

- 대부분의 내용이 Beam 데이터세트에 적용되므로 [`tfds`데이터세트 작성 가이드](https://github.com/tensorflow/datasets/tree/master/docs/add_dataset.md)를 숙지하세요.
- Apache Beam에 대한 소개는 [Beam 프로그래밍 가이드](https://beam.apache.org/documentation/programming-guide/)를 참조하세요.
- Cloud Dataflow를 사용하여 데이터세트를 생성하려면, [Google 클라우드 설명서](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python) 및 [Apache Beam 종속성 가이드](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/)를 참조하세요.

### 지침

[데이터세트 생성 가이드](https://github.com/tensorflow/datasets/tree/master/docs/add_dataset.md)에 익숙한 경우, Beam 데이터세트를 추가하려면 `_generate_examples` 함수만 수정하면 됩니다. 이 함수는 생성기가 아닌 빔 객체를 반환합니다.

빔이 아닌 데이터세트:

```python
def _generate_examples(self, path):
  for f in path.iterdir():
    yield _process_example(f)
```

빔 데이터세트:

```python
def _generate_examples(self, path):
  return (
      beam.Create(path.iterdir())
      | beam.Map(_process_example)
  )
```

나머지 모두는 테스트를 포함하여 100% 동일할 수 있습니다.

몇 가지 추가 고려 사항:

- `tfds.core.lazy_imports`를 사용하여 Apache Beam을 가져옵니다. 지연 종속성(lazy dependency)을 사용하면 사용자는 Beam을 설치하지 않고도 생성된 데이터세트를 읽을 수 있습니다.
- Python 닫힘에 주의하세요. 파이프라인을 실행할 때 `beam.Map` 및 `beam.DoFn` 함수는 `pickle`를 사용하여 직렬화되어 모든 작업자에게 전송됩니다. 상태를 작업자 간에 공유해야 하는 경우 `beam.PTransform` 내에서 개체를 변경할 수 없습니다.
- `tfds.core.DatasetBuilder`가 피클을 사용하여 직렬화되는 방식으로 인해 데이터 생성 중 `tfds.core.DatasetBuilder` 변경은 작업자에서 무시됩니다(예: `_split_generators`에서 `self.info.metadata['offset'] = 123`를 설정하고 `beam.Map(lambda x: x + self.info.metadata['offset'])`와 같은 작업자로부터 여기에 액세스할 수 없음).
- 분할 사이에 일부 파이프라인 단계를 공유해야 하는 경우 `_split_generator`에 별도의 `pipeline: beam.Pipeline` kwarg를 추가하고 전체 세대의 파이프라인을 제어할 수 있습니다. `tfds.core.GeneratorBasedBuilder`의 `_generate_examples` 문서를 참조하세요.

### 예제

다음은 Beam 데이터세트의 예입니다.

```python
class DummyBeamDataset(tfds.core.GeneratorBasedBuilder):

  VERSION = tfds.core.Version('1.0.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
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

### 파이프라인 실행하기

파이프라인을 실행하려면, 위 섹션을 살펴보세요.

**참고**: 빔이 아닌 데이터세트와 마찬가지로 `--register_checksums`으로 다운로드 체크섬을 등록하는 것을 잊지 마세요(다운로드를 처음 등록할 때만).

```sh
tfds build my_dataset --register_checksums
```

## TFDS를 입력으로 사용하는 파이프라인

TFDS 데이터세트를 소스로 사용하는 빔 파이프라인을 생성하려면 `tfds.beam.ReadFromTFDS`를 사용할 수 있습니다.

```python
builder = tfds.builder('my_dataset')

_ = (
    pipeline
    | tfds.beam.ReadFromTFDS(builder, split='train')
    | beam.Map(tfds.as_numpy)
    | ...
)
```

데이터세트의 각 샤드를 병렬로 처리합니다.

참고: 이를 위해서는 데이터세트가 이미 생성되어 있어야 합니다. 빔을 사용하여 데이터세트를 생성하려면 다른 섹션을 참조하세요.
