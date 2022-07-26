# Tuner TFX 파이프라인 구성 요소

Tuner 구성 요소는 모델의 하이퍼 매개변수를 조정합니다.

## Tuner 구성 요소 및 KerasTuner 라이브러리

Tuner 구성 요소는 하이퍼 매개변수를 조정하기 위해 Python [KerasTuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) API를 광범위하게 사용합니다.

참고: KerasTuner 라이브러리는 Keras 모델뿐만 아니라 모델링 API와 관계없이 하이퍼 매개변수 조정에 사용할 수 있습니다.

## 구성 요소

Tuner는 다음을 사용합니다.

- 훈련 및 평가에 사용되는 tf.Examples
- 모델 정의, 하이퍼 매개변수 검색 공간, 목표 등을 포함한 조정 로직을 정의하는 사용자 제공 모듈 파일(또는 모듈 fn)
- 훈련 인수 및 평가 인수의 [Protobuf](https://developers.google.com/protocol-buffers) 정의
- 조정 인수의 [Protobuf](https://developers.google.com/protocol-buffers) 정의(선택 사항)
- 업스트림 Transform 구성 요소에서 생성한 변환 그래프(선택 사항)
- SchemaGen 파이프라인 구성 요소로 생성되고 개발자가 선택적으로 변경하는 데이터 스키마(선택 사항)

주어진 데이터, 모델 및 목표를 사용하여 Tuner는 하이퍼 매개변수를 조정하고 최상의 결과를 내보냅니다.

## 지침

Tuner에는 다음 서명이 있는 사용자 모듈 함수 `tuner_fn`이 필요합니다.

```python
...
from keras_tuner.engine import base_tuner

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
  """Build the tuner using the KerasTuner API.
  Args:
    fn_args: Holds args as name/value pairs.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.
  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """
  ...
```

이 함수에서는 모델과 하이퍼 매개변수 검색 공간을 모두 정의하고 조정을 위한 목표와 알고리즘을 선택합니다. Tuner 구성 요소는 이 모듈 코드를 입력으로 사용하여 하이퍼 매개변수를 조정하고 최상의 결과를 내보냅니다.

Trainer는 Tuner의 출력 하이퍼 매개변수를 입력으로 가져와 사용자 모듈 코드에서 활용할 수 있습니다. 파이프라인 정의는 다음과 같습니다.

```python
...
tuner = Tuner(
    module_file=module_file,  # Contains `tuner_fn`.
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    train_args=trainer_pb2.TrainArgs(num_steps=20),
    eval_args=trainer_pb2.EvalArgs(num_steps=5))

trainer = Trainer(
    module_file=module_file,  # Contains `run_fn`.
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    # This will be passed to `run_fn`.
    hyperparameters=tuner.outputs['best_hyperparameters'],
    train_args=trainer_pb2.TrainArgs(num_steps=100),
    eval_args=trainer_pb2.EvalArgs(num_steps=5))
...
```

모델을 다시 훈련할 때마다 하이퍼 매개변수를 조정하지 않을 수 있습니다. Tuner를 사용하여 좋은 하이퍼 매개변수 세트를 결정한 후에는, 파이프라인에서 Tuner를 제거하고 `ImporterNode`를 사용하여 이전 훈련 실행에서 Tuner 아티팩트를 가져와 Trainer에 공급할 수 있습니다.

```python
hparams_importer = Importer(
    # This can be Tuner's output file or manually edited file. The file contains
    # text format of hyperparameters (keras_tuner.HyperParameters.get_config())
    source_uri='path/to/best_hyperparameters.txt',
    artifact_type=HyperParameters,
).with_id('import_hparams')

trainer = Trainer(
    ...
    # An alternative is directly use the tuned hyperparameters in Trainer's user
    # module code and set hyperparameters to None here.
    hyperparameters = hparams_importer.outputs['result'])
```

## Google Cloud Platform(GCP)에서 조정하기

Google Cloud Platform(GCP)에서 실행할 때 Tuner 구성 요소는 두 가지 서비스를 활용할 수 있습니다.

- [AI Platform Vizier](https://cloud.google.com/ai-platform/optimizer/docs/overview)(CloudTuner 구현을 통해)
- [AI Platform Training](https://cloud.google.com/ai-platform/training/docs)(분산 조정을 위한 무리 관리자로)

### 하이퍼 매개변수 조정의 백엔드 역할을 하는 AI Platform Vizier

[AI Platform Vizier](https://cloud.google.com/ai-platform/optimizer/docs/overview)는 [Google Vizier](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/bcb15507f4b52991a0783013df4222240e942381.pdf) 기술을 기반으로 블랙 박스 최적화를 수행하는 관리형 서비스입니다.

[CloudTuner](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/tuner/tuner.py)는 연구용 백엔드로 AI Platform Vizier 서비스와 통신하는 [KerasTuner](https://www.tensorflow.org/tutorials/keras/keras_tuner)를 구현한 것입니다. CloudTuner는 `kerastuner.Tuner`의 서브 클래스이므로 `tuner_fn` 모듈에서 드롭인 교체로 사용하고 TFX Tuner 구성 요소의 일부로 실행할 수 있습니다.

아래는 `CloudTuner`의 사용 방법을 보여주는 코드 조각입니다. `CloudTuner`에 대한 구성에는 `project_id` 및 `region`과 같은 GCP 관련 항목이 필요합니다.

```python
...
from tensorflow_cloud import CloudTuner

...
def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
  """An implementation of tuner_fn that instantiates CloudTuner."""

  ...
  tuner = CloudTuner(
      _build_model,
      hyperparameters=...,
      ...
      project_id=...,       # GCP Project ID
      region=...,           # GCP Region where Vizier service is run.
  )

  ...
  return TuneFnResult(
      tuner=tuner,
      fit_kwargs={...}
  )

```

### Cloud AI Platform 훈련 분산 작업자 무리의 병렬 조정

Tuner 구성 요소의 기본 구현인 KerasTuner 프레임워크에는 하이퍼 매개변수 검색을 병렬로 수행하는 기능이 있습니다. stock Tuner 구성 요소에는 둘 이상의 검색 작업자를 병렬로 실행할 수 있는 기능이 없지만, [Google Cloud AI Platform 확장 Tuner 구성 요소](https://github.com/tensorflow/tfx/blob/master/tfx/extensions/google_cloud_ai_platform/tuner/component.py)를 사용하면 AI Platform Training 작업을 분산 작업자 무리 관리자로 사용하여 병렬 조정을 실행할 수 있는 기능을 제공합니다. [TuneArgs](https://github.com/tensorflow/tfx/blob/master/tfx/proto/tuner.proto)는 이 구성 요소에 제공된 구성입니다. 이는 stock Tuner 구성 요소의 드롭인 교체입니다.

```python
tuner = google_cloud_ai_platform.Tuner(
    ...   # Same kwargs as the above stock Tuner component.
    tune_args=proto.TuneArgs(num_parallel_trials=3),  # 3-worker parallel
    custom_config={
        # Configures Cloud AI Platform-specific configs . For for details, see
        # https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#traininginput.
        TUNING_ARGS_KEY:
            {
                'project': ...,
                'region': ...,
                # Configuration of machines for each master/worker in the flock.
                'masterConfig': ...,
                'workerConfig': ...,
                ...
            }
    })
...

```

확장 Tuner 구성 요소의 동작과 출력은 여러 하이퍼 매개변수 검색이 서로 다른 작업자 머신에서 병렬로 실행된다는 점을 제외하면 stock Tuner 구성 요소와 같으며, 그 결과, `num_trials`가 더 빨리 완료됩니다. 이는 `RandomSearch`와 같이 검색 알고리즘이 과도하게 병렬화될 때 특히 효과적입니다. 그러나 AI Platform Optimizer에 구현된 Google Vizier 알고리즘과 같이, 검색 알고리즘이 이전 시도의 결과에서 나온 정보를 사용하면, 과도한 병렬 검색은 검색의 효율성에 부정적인 영향을 미칠 수 있습니다.

참고: 각 병렬 검색의 각 시도는 작업자 무리의 단일 시스템에서 수행됩니다. 즉, 각 시도는 다중 작업자 분산 교육을 활용하지 않습니다. 각 시도에 대해 다중 작업자 분산이 필요한 경우 `CloudTuner` 대신 [`DistributingCloudTuner`](https://github.com/tensorflow/cloud/blob/b9c8752f5c53f8722dfc0b5c7e05be52e62597a8/src/python/tensorflow_cloud/tuner/tuner.py#L384-L676)를 참조하세요.

참고: `CloudTuner`와 Google Cloud AI Platform 확장 Tuner 구성 요소를 함께 사용할 수 있으며, 이 경우 AI Platform Vizier의 하이퍼 매개변수 검색 알고리즘이 지원하는 분산 병렬 조정이 기능합니다. 그러나 이렇게 하려면 Cloud AI Platform 작업에 AI Platform Vizier 서비스에 대한 액세스 권한이 부여되어야 합니다. 사용자 지정 서비스 계정을 설정하려면 이 [가이드](https://cloud.google.com/ai-platform/training/docs/custom-service-account#custom)를 참조하세요. 그런 다음 파이프라인 코드에서 훈련 작업에 대한 사용자 지정 서비스 계정을 지정해야 합니다. 자세한 내용은 [GCP 예제의 E2E CloudTuner](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_kubeflow_gcp.py)를 참조하세요.

## 링크

[E2E 예제](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py)

[GCP 예제에서 E2E CloudTuner](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_kubeflow.py)

[KerasTuner 튜토리얼](https://www.tensorflow.org/tutorials/keras/keras_tuner)

[CloudTuner 튜토리얼](https://github.com/GoogleCloudPlatform/ai-platform-samples/blob/master/notebooks/samples/optimizer/ai_platform_vizier_tuner.ipynb)

[제안](https://github.com/tensorflow/community/blob/master/rfcs/20200420-tfx-tuner-component.md)

자세한 내용은 [Tuner API 참조](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Tuner)에서 확인할 수 있습니다.
