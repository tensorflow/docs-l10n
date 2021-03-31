# TFX용 TensorFlow 모델링 코드 설계하기

TFX용 TensorFlow 모델링 코드를 설계할 때 모델링 API의 선택을 포함하여 몇 가지 알아야 할 항목이 있습니다.

- 입력: [Transform](transform.md)의 SavedModel 및 [ExampleGen](examplegen.md)의 데이터
- 출력: SavedModel 형식의 훈련된 모델

<aside class="note" id="tf2-support"><b>참고:</b> TFX는 사소한 예외를 제외하고 거의 모든 TensorFlow 2.X를 지원합니다. TFX는 또한 TensorFlow 1.15를 전체적으로 지원합니다.</aside>

<ul>
  <li>새로운 TFX 파이프라인은 <a href="https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-generic-trainer.md">Generic Trainer</a>를 통해 Keras 모델이 있는 TensorFlow 2.x를 사용해야 합니다.</li>
  <li>tf.distribute에 대한 개선된 지원을 포함하여 TensorFlow 2.X에 대한 전체 지원이 향후 릴리스에서 점진적으로 추가될 것입니다.</li>
  <li>이전 TFX 파이프라인은 TensorFlow 1.15를 계속 사용할 수 있습니다. TensorFlow 2.X로 전환하려면 <a href="https://www.tensorflow.org/guide/migrate">TensorFlow 마이그레이션 가이드</a>를 참조하세요.</li>
</ul>

TFX 릴리스에 대한 최신 정보를 얻으려면 <a href="https://github.com/tensorflow/tfx/blob/master/ROADMAP.md">TFX OSS 로드맵</a>을 참조하고, <a href="https://blog.tensorflow.org/search?label=TFX&amp;max-results=20">TFX 블로그</a>를 읽고 <a href="https://services.google.com/fb/forms/tensorflow/">TensorFlow 뉴스레터</a>를 구독하세요.




모델의 입력 레이어는 [Transform](transform.md) 구성 요소로 만들어진 SavedModel에서 사용해야 하며, SavedModel 및 EvalSavedModel을 내보낼 때 [Transform](transform.md) 구성 요소로 만들어진 이들 변환이 포함되도록 Transform 모델의 레이어가 해당 모델에 포함되어야 합니다.

TFX용 일반적인 TensorFlow 모델 설계는 다음과 같습니다.

```python
def _build_estimator(tf_transform_dir,                      config,                      hidden_units=None,                      warm_start_from=None):   """Build an estimator for predicting the tipping behavior of taxi riders.    Args:     tf_transform_dir: directory in which the tf-transform model was written       during the preprocessing step.     config: tf.contrib.learn.RunConfig defining the runtime environment for the       estimator (including model_dir).     hidden_units: [int], the layer sizes of the DNN (input layer first)     warm_start_from: Optional directory to warm start from.    Returns:     Resulting DNNLinearCombinedClassifier.   """   metadata_dir = os.path.join(tf_transform_dir,                               transform_fn_io.TRANSFORMED_METADATA_DIR)   transformed_metadata = metadata_io.read_metadata(metadata_dir)   transformed_feature_spec = transformed_metadata.schema.as_feature_spec()    transformed_feature_spec.pop(_transformed_name(_LABEL_KEY))    real_valued_columns = [       tf.feature_column.numeric_column(key, shape=())       for key in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)   ]   categorical_columns = [       tf.feature_column.categorical_column_with_identity(           key, num_buckets=_VOCAB_SIZE + _OOV_SIZE, default_value=0)       for key in _transformed_names(_VOCAB_FEATURE_KEYS)   ]   categorical_columns += [       tf.feature_column.categorical_column_with_identity(           key, num_buckets=_FEATURE_BUCKET_COUNT, default_value=0)       for key in _transformed_names(_BUCKET_FEATURE_KEYS)   ]   categorical_columns += [       tf.feature_column.categorical_column_with_identity(           key, num_buckets=num_buckets, default_value=0)       for key, num_buckets in zip(           _transformed_names(_CATEGORICAL_FEATURE_KEYS),  #           _MAX_CATEGORICAL_FEATURE_VALUES)   ]   return tf.estimator.DNNLinearCombinedClassifier(       config=config,       linear_feature_columns=categorical_columns,       dnn_feature_columns=real_valued_columns,       dnn_hidden_units=hidden_units or [100, 70, 50, 25],       warm_start_from=warm_start_from)
```
