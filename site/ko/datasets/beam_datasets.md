# Apache Beam으로 빅 데이터세트 생성하기

일부 데이터세트는 너무 커서 단일 머신에서 처리할 수 없습니다. `tfds`는 [Apache Beam](https://beam.apache.org/)을 사용하여 많은 머신에서 데이터를 생성하도록 지원합니다.

이 문서에는 두 개의 섹션이 있습니다:

- 기존 Beam 데이터세트를 생성하려는 사용자
- 새로운 Beam 데이터세트를 생성하려는 개발자

목차:

- [Beam 데이터세트 생성하기](#generating-a-beam-dataset)
    - [Google Cloud Dataflow에서](#on-google-cloud-dataflow)
    - [로컬에서](#locally)
    - [사용자 정의 스크립트 내](#with-a-custom-script)
- [Beam 데이터세트 구현하기](#implementing-a-beam-dataset)
    - [전제 조건](#prerequisites)
    - [지침](#instructions)
    - [예제](#example)
    - [파이프라인 실행](#run-your-pipeline)

## Beam 데이터세트 생성하기

다음은 클라우드 또는 로컬에서 Beam 데이터세트를 생성하는 서로 다른 예제입니다.

**경고**: `tensorflow_datasets.scripts.download_and_prepare` 스크립트를 사용하여 데이터세트를 생성할 때 생성하려는 데이터세트 구성을 지정해야 합니다. 그렇지 않으면, 기본적으로 모든 기존 구성이 생성됩니다. 예를 들어, [wikipedia](https://www.tensorflow.org/datasets/catalog/wikipedia)를 위해서는 `--dataset=wikipedia` 대신 `--dataset=wikipedia/20200301.en`를 사용하세요.

### Google Cloud Dataflow에서

[Google Cloud Dataflow](https://cloud.google.com/dataflow/)를 사용하여 파이프라인을 실행하고 분산 계산을 이용하려면, 먼저 [빠른 시작 지침](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python)을 따르세요.

환경이 설정되면, [GCS](https://cloud.google.com/storage/)의 데이터 디렉토리를 사용하고 `--beam_pipeline_options` 플래그에 [필요한 옵션](https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#configuring-pipelineoptions-for-execution-on-the-cloud-dataflow-service)을 지정하여 `download_and_prepare` 스크립트를 실행할 수 있습니다.

스크립트를 보다 쉽게 실행하려면, GCP/GCS 설정의 실제 값과 생성하려는 데이터세트를 사용하여 다음 변수를 정의하면 도움이 됩니다.

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
python -m tensorflow_datasets.scripts.download_and_prepare \
  --datasets=my_new_dataset
```

**경고**: Beam 데이터세트는 **매우** 클 수 있으며(테라바이트), 상당한 양의 리소스가 생성될 수 있습니다(로컬 컴퓨터에서 몇 주가 걸릴 수 있음). 분산 환경을 사용하여 데이터세트를 생성하는 것이 좋습니다. 지원되는 런타임 목록은 [Apache Beam 설명서](https://beam.apache.org/)를 참조하세요.

### 사용자 정의 스크립트

Beam에서 데이터세트를 생성하기 위한 API는 다른 데이터세트에서와 같지만, Beam 옵션 또는 러너를 `DownloadConfig`로 전달해야 합니다.

```py
# If you are running on Dataflow, Spark,..., you may have to set-up runtime
# flags. Otherwise, you can leave flags empty [].
flags = ['--runner=DataflowRunner', '--project=<project-name>', ...]

# To use Beam, you have to set at least one of `beam_options` or `beam_runner`
dl_config = tfds.download.DownloadConfig(
    beam_options=beam.options.pipeline_options.PipelineOptions(flags=flags)
)

data_dir = 'gs://my-gcs-bucket/tensorflow_datasets'
builder = tfds.builder('wikipedia/20190301.en', data_dir=data_dir)
builder.download_and_prepare(
    download_dir=FLAGS.download_dir,
    download_config=dl_config,
)
```

## Beam 데이터세트 구현하기

### 전제 조건

Apache Beam 데이터세트를 작성하려면, 다음 개념에 익숙해야 합니다.

- 대부분의 내용이 Beam 데이터세트에 적용되므로 [`tfds`데이터세트 작성 가이드](https://github.com/tensorflow/datasets/tree/master/docs/add_dataset.md)를 숙지하세요.
- Apache Beam에 대한 소개는 [Beam 프로그래밍 가이드](https://beam.apache.org/documentation/programming-guide/)를 참조하세요.
- Cloud Dataflow를 사용하여 데이터세트를 생성하려면, [Google 클라우드 설명서](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python) 및 [Apache Beam 종속성 가이드](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/)를 참조하세요.

### 지침

[데이터세트 생성 가이드](https://github.com/tensorflow/datasets/tree/master/docs/add_dataset.md)에 익숙하다면, Beam 데이터세트를 추가하기 위해 약간만 수정하면 됩니다.

- `DatasetBuilder`는 `tfds.core.GeneratorBasedBuilder` 대신 `tfds.core.BeamBasedBuilder`에서 상속합니다.
- Beam 데이터세트는 메서드 `_generate_examples(self, **kwargs)` 대신 추상 메서드 `_build_pcollection(self, **kwargs)`를 구현해야 합니다. `_build_pcollection`는 분할과 관련된 예제와 함께 `beam.PCollection`를 반환해야 합니다.
- Beam 데이터세트에 대한 단위 테스트의 작성은 다른 데이터세트에서와 같습니다.

몇 가지 추가 고려 사항:

- `tfds.core.lazy_imports`를 사용하여 Apache Beam을 가져옵니다. 지연 종속성(lazy dependency)을 사용하면 사용자는 Beam을 설치하지 않고도 생성된 데이터세트를 읽을 수 있습니다.
- Python closure에 주의하세요. 파이프라인을 실행할 때 `beam.Map` 및 `beam.DoFn` 함수는 `pickle`를 사용하여 직렬화되고 모든 작업자에게 전송됩니다. 이때 버그가 발생할 수 있습니다. 예를 들어, 함수 외부에서 선언된 변경 가능한 객체를 함수에서 사용하는 경우, `pickle` 오류 또는 예기치 않은 동작이 발생할 수 있습니다. 해결 방법은 일반적으로 닫힌 객체(closed-over objects)의 변경을 피하는 것입니다.
- Beam 파이프라인의 `DatasetBuilder`에서 메서드를 사용하는 것이 좋습니다. 그러나 피클 중 클래스가 직렬화되는 방식과 작성 중 특성에 대한 변경 사항은 무시됩니다.

### 예제

다음은 Beam 데이터세트의 예제입니다. 보다 복잡한 실제 예를 보려면, [`Wikipedia` 데이터세트](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/text/wikipedia.py)를 살펴보세요.

```python
class DummyBeamDataset(tfds.core.BeamBasedBuilder):

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
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(file_dir='path/to/train_data/'),
        ),
        splits_lib.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(file_dir='path/to/test_data/'),
        ),
    ]

  def _build_pcollection(self, pipeline, file_dir):
    """Generate examples as dicts."""
    beam = tfds.core.lazy_imports.apache_beam

    def _process_example(filename):
      # Use filename as key
      return filename, {
          'image': os.path.join(file_dir, filename),
          'label': filename.split('.')[1],  # Extract label: "0010102.dog.jpeg"
      }

    return (
        pipeline
        | beam.Create(tf.io.gfile.listdir(file_dir))
        | beam.Map(_process_example)
    )
```

### 파이프라인 실행하기

파이프라인을 실행하려면, 위 섹션을 살펴보세요.

**경고**: 다운로드를 등록하기 위해 데이터세트를 처음 실행할 때 레지스터 체크섬 `--register_checksums` 플래그를 `download_and_prepare` 스크립트에 추가하는 것을 잊지 마세요.

```sh
python -m tensorflow_datasets.scripts.download_and_prepare \
  --register_checksums \
  --datasets=my_new_dataset
```
