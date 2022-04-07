# Tuner TFX 파이프라인 구성 요소

Tuner 구성 요소는 모델의 하이퍼 매개변수를 조정합니다.

## Tuner 구성 요소 및 KerasTuner 라이브러리

Tuner 구성 요소는 하이퍼 매개변수를 조정하기 위해 Python [KerasTuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) API를 광범위하게 사용합니다.

Note: The KerasTuner library can be used for hyperparameter tuning regardless of the modeling API, not just for Keras models only.

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

- [AI Platform Vizier](https://cloud.google.com/ai-platform/optimizer/docs/overview) (via CloudTuner implementation)
- [AI Platform Training](https://cloud.google.com/ai-platform/training/docs) (as a flock manager for distributed tuning)

### AI Platform Vizier as the backend of hyperparameter tuning

[AI Platform Vizier](https://cloud.google.com/ai-platform/optimizer/docs/overview) is a managed service that performs black box optimization, based on the [Google Vizier](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/bcb15507f4b52991a0783013df4222240e942381.pdf) technology.

[CloudTuner](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/tuner/tuner.py) is an implementation of [KerasTuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) which talks to the AI Platform Vizier service as the study backend. Since CloudTuner is a subclass of `keras_tuner.Tuner`, it can be used as a drop-in replacement in the `tuner_fn` module, and execute as a part of the TFX Tuner component.

Below is a code snippet which shows how to use `CloudTuner`. Notice that configuration to `CloudTuner` requires items which are specific to GCP, such as the `project_id` and `region`.

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

The KerasTuner framework as the underlying implementation of the Tuner component has ability to conduct hyperparameter search in parallel. While the stock Tuner component does not have ability to execute more than one search worker in parallel, by using the [Google Cloud AI Platform extension Tuner component](https://github.com/tensorflow/tfx/blob/master/tfx/extensions/google_cloud_ai_platform/tuner/component.py), it provides the ability to run parallel tuning, using an AI Platform Training Job as a distributed worker flock manager. [TuneArgs](https://github.com/tensorflow/tfx/blob/master/tfx/proto/tuner.proto) is the configuration given to this component. This is a drop-in replacement of the stock Tuner component.

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

The behavior and the output of the extension Tuner component is the same as the stock Tuner component, except that multiple hyperparameter searches are executed in parallel on different worker machines, and as a result, the `num_trials` will be completed faster. This is particularly effective when the search algorithm is embarrassingly parallelizable, such as `RandomSearch`. However, if the search algorithm uses information from results of prior trials, such as Google Vizier algorithm implemented in the AI Platform Vizier does, an excessively parallel search would negatively affect the efficacy of the search.

Note: Each trial in each parallel search is conducted on a single machine in the worker flock, i.e., each trial does not take advantage of multi-worker distributed training. If multi-worker distribution is desired for each trial, refer to [`DistributingCloudTuner`](https://github.com/tensorflow/cloud/blob/b9c8752f5c53f8722dfc0b5c7e05be52e62597a8/src/python/tensorflow_cloud/tuner/tuner.py#L384-L676), instead of `CloudTuner`.

Note: Both `CloudTuner` and the Google Cloud AI Platform extensions Tuner component can be used together, in which case it allows distributed parallel tuning backed by the AI Platform Vizier's hyperparameter search algorithm. However, in order to do so, the Cloud AI Platform Job must be given access to the AI Platform Vizier service. See this [guide](https://cloud.google.com/ai-platform/training/docs/custom-service-account#custom) to set up a custom service account. After that, you should specify the custom service account for your training job in the pipeline code. More details see [E2E CloudTuner on GCP exmaple](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_kubeflow_gcp.py).

## 링크

[E2E Example](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py)

[E2E CloudTuner on GCP Example](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_kubeflow.py)

[KerasTuner tutorial](https://www.tensorflow.org/tutorials/keras/keras_tuner)

[CloudTuner tutorial](https://github.com/GoogleCloudPlatform/ai-platform-samples/blob/master/notebooks/samples/optimizer/ai_platform_vizier_tuner.ipynb)

[Proposal](https://github.com/tensorflow/community/blob/master/rfcs/20200420-tfx-tuner-component.md)

More details are available in the [Tuner API reference](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Tuner).
