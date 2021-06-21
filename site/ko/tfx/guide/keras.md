# TFX의 TensorFlow 2.x

[2019년에 발표된 TensorFlow 2.0](https://blog.tensorflow.org/2019/09/tensorflow-20-is-now-available.html)은 [Keras와의 긴밀한 통합](https://www.tensorflow.org/guide/keras/overview)을 지원하며, 기본으로 포함되는 [즉시 실행](https://www.tensorflow.org/guide/eager), 그리고 [Python 함수 실행](https://www.tensorflow.org/guide/function) 및 기타 [새로운 기능 및 개선 사항](https://www.tensorflow.org/guide/effective_tf2#a_brief_summary_of_major_changes)을 포함합니다.

이 가이드에서는 TFX의 TF 2.x에 대한 포괄적인 기술 개요를 제공합니다.

## 어떤 버전을 사용할까요?

TFX는 TensorFlow 2.x와 호환되며, TensorFlow 1.x(특히 Estimator)에 있던 고급 API는 계속 호환됩니다.

### TensorFlow 2.x에서 새 프로젝트를 시작합니다.

TensorFlow 2.x는 TensorFlow 1.x의 높은 수준의 기능을 유지하므로, 새 기능을 사용할 계획이 없더라도 새 프로젝트에서 이전 버전을 사용하여 얻는 이점이 없습니다.

따라서 새 TFX 프로젝트를 시작하려면 TensorFlow 2.x를 사용하는 것이 좋습니다. Keras 및 기타 새로운 기능에 대한 완전한 지원이 제공되면 나중에 코드를 업데이트하고 싶을 수 있으며, 향후 TensorFlow 1.x에서 업그레이드를 시도하는 대신 TensorFlow 2.x로 시작하면 변경 범위가 훨씬 더 제한됩니다.

### 기존 프로젝트를 TensorFlow 2.x로 변환하기

TensorFlow 1.x용으로 작성된 코드는 TensorFlow 2.x와 주로 호환되며 TFX에서 계속 동작합니다.

하지만 TF 2.x에서 제공되는 개선 사항과 새로운 기능을 활용하려면 [TF 2.x로 마이그레이션하려면 따라야 할 지침](https://www.tensorflow.org/guide/migrate)을 참고하세요.

## Estimator

Estimator API는 TensorFlow 2.x에서 유지되었지만 새로운 기능 및 개발의 초점은 아닙니다. Estimator를 사용하여 TensorFlow 1.x 또는 2.x로 작성된 코드는 TFX에서 예상대로 계속 동작합니다.

다음은 순수 Estimator를 사용한 엔드 투 엔드 TFX 예제입니다. [택시 예제(Estimator)](https://github.com/tensorflow/tfx/blob/r0.21/tfx/examples/chicago_taxi_pipeline/taxi_utils.py)

## `model_to_estimator`를 사용한 Keras

Keras 모델은 `tf.keras.estimator.model_to_estimator` 함수로 래핑할 수 있으며, 이를 통해 마치 Estimator인 것처럼 동작할 수 있습니다. 이를 사용하려면 다음을 따릅니다.

1. Keras 모델을 빌드합니다.
2. 컴파일된 모델을 `model_to_estimator`로 전달합니다.
3. 일반적으로 Estimator를 사용하는 방식으로 Trainer에서 `model_to_estimator`의 결과를 사용합니다.

```py
# Build a Keras model.
def _keras_model_builder():
  """Creates a Keras model."""
  ...

  model = tf.keras.Model(inputs=inputs, outputs=output)
  model.compile()

  return model


# Write a typical trainer function
def trainer_fn(trainer_fn_args, schema):
  """Build the estimator, using model_to_estimator."""
  ...

  # Model to estimator
  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=_keras_model_builder(), config=run_config)

  return {
      'estimator': estimator,
      ...
  }
```

Trainer의 사용자 모듈 파일을 제외하고 나머지 파이프라인은 변경되지 않습니다.

## 네이티브 Keras(예: `model_to_estimator`가 없는 Keras)

참고: Keras의 모든 기능에 대한 완전한 지원이 진행 중이며, 대부분의 경우 TFX의 Keras가 예상대로 동작합니다. FeatureColumns에 대한 희소 특성에서는 아직 동작하지 않습니다.

### 예제 및 Colab

다음은 기본 Keras를 사용한 몇 가지 예입니다.

- [Penguin](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py)([모듈 파일](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils.py)): 'Hello world' 엔드 투 엔드 예제
- [MNIST](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py)([모듈 파일](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_utils_native_keras.py)): 이미지 및 TFLite 엔드 투 엔드 예제
- [Taxi](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_native_keras.py)([모듈 파일](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_utils_native_keras.py)): 고급 Transform 사용법을 사용한 엔드 투 엔드 예제

또한 구성 요소별 [Keras Colab](https://www.tensorflow.org/tfx/tutorials/tfx/components_keras)이 있습니다.

### TFX 구성 요소

다음 섹션에서는 관련 TFX 구성 요소가 네이티브 Keras를 지원하는 방법을 설명합니다.

#### Transform

Transform은 현재 Keras 모델을 실험적으로 지원합니다.

Transform 구성 요소 자체를 변경 없이 네이티브 Keras에 사용할 수 있습니다. `preprocessing_fn` 정의는 [TensorFlow](https://www.tensorflow.org/api_docs/python/tf) 및 [tf.Transform](https://www.tensorflow.org/tfx/transform/api_docs/python/tft) 연산을 사용하여 동일하게 유지됩니다.

네이티브 Keras의 적용 함수 및 평가 함수가 변경되었습니다. 자세한 내용은 다음 Trainer 및 Evaluator 섹션에서 설명됩니다.

참고: `preprocessing_fn` 내의 이들 transform은 훈련 또는 평가용 레이블 특성에 적용할 수 없습니다.

#### Trainer

네이티브 Keras를 구성하려면 Trainer 구성 요소에 대해 `GenericExecutor`를 설정하여 기본 Estimator 기반 실행자를 대체해야 합니다. 자세한 내용은 [여기](trainer.md#configuring-the-trainer-component-to-use-the-genericexecutor)를 확인하세요.

##### Transform이 있는 Keras 모듈 파일

훈련 모듈 파일에는 `GenericExecutor`에서 호출하는 `run_fn`이 포함되어야 합니다. 일반적인 Keras `run_fn`은 다음과 같습니다.

```python
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  # Train and eval files contains transformed examples.
  # _input_fn read dataset based on transformed schema from tft.
  train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor,
                            tf_transform_output.transformed_metadata.schema)
  eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor,
                           tf_transform_output.transformed_metadata.schema)

  model = _build_keras_model()

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
```

위의 `run_fn`에서 훈련된 모델을 내보낼 때 모델이 예측을 위한 원시 예제를 사용할 수 있도록 적용 서명(serving signature)이 필요합니다. 일반적인 적용 함수는 다음과 같습니다.

```python
def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""

  # the layer is added as an attribute to the model in order to make sure that
  # the model assets are handled correctly when exporting.
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = model.tft_layer(parsed_features)

    return model(transformed_features)

  return serve_tf_examples_fn
```

위의 적용 함수에서 추론을 위해 [`tft.TransformFeaturesLayer`](https://github.com/tensorflow/transform/blob/master/docs/api_docs/python/tft/TransformFeaturesLayer.md) 레이어로 이들 tf.Transform 변환을 원시 데이터에 적용해야 합니다. Estimator에 필요했던 이전 `_serving_input_receiver_fn`은 더는 Keras에 필요하지 않습니다.

##### Transform이 없는 Keras 모듈 파일

위에 표시된 모듈 파일과 유사하지만 transform이 없습니다.

```python
def _get_serve_tf_examples_fn(model, schema):

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    feature_spec = _get_raw_feature_spec(schema)
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    return model(parsed_features)

  return serve_tf_examples_fn


def run_fn(fn_args: TrainerFnArgs):
  schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())

  # Train and eval files contains raw examples.
  # _input_fn reads the dataset based on raw data schema.
  train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, schema)
  eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, schema)

  model = _build_keras_model()

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model, schema).get_concrete_function(
              tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
```

#####

[tf.distribute.Strategy](https://www.tensorflow.org/guide/distributed_training)

현재 TFX는 단일 작업자 전략(예: [MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy), [OneDeviceStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/OneDeviceStrategy))만 지원합니다.

배포 전략을 사용하려면 적절한 tf.distribute.Strategy를 만들고 전략 범위 내에서 Keras 모델의 만들기 및 컴파일을 이동합니다.

예를 들어, 위의 `model = _build_keras_model()`을 다음으로 대체합니다.

```python
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model()

  # Rest of the code can be unchanged.
  model.fit(...)
```

`MirroredStrategy`에서 사용하는 기기(CPU/GPU)를 확인하려면 정보 수준의 tensorflow 로깅을 사용합니다.

```python
import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)
```

로그에서 `Using MirroredStrategy with devices (...)`를 볼 수 있습니다.

참고: GPU 메모리 부족 문제에 환경 변수 `TF_FORCE_GPU_ALLOW_GROWTH=true`가 필요할 수 있습니다. 자세한 내용은 [tensorflow GPU 가이드](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth)를 참조하세요.

#### Evaluator

TFMA v0.2x에서는 ModelValidator와 Evaluator가 하나의 [새로운 Evaluator 구성 요소](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-combining-model-validator-with-evaluator.md)로 결합되었습니다. 새로운 Evaluator 구성 요소는 단일 모델 평가를 모두 수행하고, 이전 모델과 비교하여 현재 모델을 검증할 수도 있습니다. 이 변경으로 Pusher 구성 요소는 이제 ModelValidator 대신 Evaluator의 축복(blessing) 결과를 사용합니다.

새로운 Evaluator는 Keras 모델과 Estimator 모델을 지원합니다. Evaluator는 이제 적용에 사용되는 것과 같은 `SavedModel`을 기반으로 하므로, 이전에 필요했던`_eval_input_receiver_fn` 및 저장된 평가 모델은 더는 Keras에 필요하지 않습니다.

[자세한 내용은 Evaluator를 참조하세요](evaluator.md).
