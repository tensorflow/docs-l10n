# Transform TFX 파이프라인 구성 요소

Transform TFX 파이프라인 구성 요소는 [SchemaGen](schemagen.md) 구성 요소에서 생성한 데이터 스키마를 사용하여 [ExampleGen](examplegen.md) 구성 요소에서 내보낸 tf.Examples에 대한 특성 엔지니어링을 수행하고 SavedModel과 변환 전 및 변환 후 데이터에 대한 통계를 모두 내보냅니다. 실행되면 SavedModel은 ExampleGen 구성 요소에서 내보낸 tf.Examples를 수락하고 변환된 특성 데이터를 내보냅니다.

- 입력: ExampleGen 구성 요소의 tf.Examples 및 SchemaGen 구성 요소의 데이터 스키마
- 출력: Trainer 구성 요소, 변환 전 및 변환 후 통계로 SavedModel을 내보냄

## Transform 구성 요소 구성하기

`preprocessing_fn`이 작성되면 Transform 구성 요소에 입력으로 제공되는 Python 모듈에서 정의되어야 합니다. 이 모듈은 변환에 의해 로드되고 `preprocessing_fn`이라는 함수가 발견되며 Transform에서 전처리 파이프라인을 구성하는 데 사용됩니다.

```
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(_taxi_transform_module_file))
```

또한 [TFDV](tfdv.md) 기반 변환 전 또는 변환 후 통계 계산에 옵션을 제공해야 할 수 있습니다. 그렇게 하려면 동일한 모듈 내에서 `stats_options_updater_fn`을 정의합니다.

## Transform 및 TensorFlow Transform

Transform은 데이터세트에서 특성 엔지니어링을 수행하기 위해 [TensorFlow Transform](tft.md)을 광범위하게 사용합니다. TensorFlow Transform은 특성 데이터가 모델로 이동하기 전에 훈련 프로세스의 일부로 특성 데이터를 변환할 수 있는 훌륭한 도구입니다. 일반적인 특성 변환은 다음과 같습니다.

- **임베딩**: 고차원 공간에서 저차원 공간으로의 의미 있는 매핑을 찾아 희소 특성(예: 어휘에서 생성되는 정수 ID)을 밀집 특성으로 변환합니다. 임베딩 시작하기는 [머신러닝 단기 집중 과정의 임베딩 단원](https://developers.google.com/machine-learning/crash-course/embedding)을 참조하세요.
- **어휘 생성**: 각 고유 값을 ID 번호에 매핑하는 어휘를 만들어 문자열 또는 숫자가 아닌 기타 특성을 정수로 변환합니다.
- **값 정규화**: 숫자 특성을 모두 유사한 범위에 속하도록 변환합니다.
- **버킷화** 이산 버킷에 값을 할당하여 범주 특성으로 연속 값 특성을 변환합니다.
- **텍스트 특성 강화**: 토큰, n-gram, 엔티티, 감상 등과 같은 원시 데이터에서 특성을 생성하여 특성 세트를 강화합니다.

TensorFlow Transform은 다음과 같은 다양한 유형의 변환을 지원합니다.

- 최신 데이터에서 어휘를 자동으로 생성합니다.

- 데이터를 모델로 보내기 전에 데이터에서 임의 변환을 수행합니다. TensorFlow Transform은 모델의 TensorFlow 그래프에 변환을 빌드하므로 훈련 및 추론 시간에 같은 변환이 수행됩니다. 모든 훈련 인스턴스에서 특성의 최대값과 같이 데이터의 전역 속성을 참조하는 변환을 정의할 수 있습니다.

TFX를 실행하기 전에 원하는 대로 데이터를 변환할 수 있습니다. 그러나 TensorFlow Transform 내에서 수행하면 변환이 TensorFlow 그래프의 일부가 됩니다. 이 접근 방식은 훈련/적용 편향을 방지하는 데 도움이 됩니다.

모델링 코드 내부의 변환은 FeatureColumns를 사용합니다. FeatureColumns를 사용하여 버킷화, 사전 정의된 어휘를 사용하는 정수화 또는 데이터를 보지 않고 정의할 수 있는 기타 변환을 정의할 수 있습니다.

반대로 TensorFlow Transform은 사전에 알려지지 않은 값을 계산하기 위해 데이터에 대한 전체 전달이 필요한 변환을 위해 설계되었습니다. 예를 들어, 어휘 생성에는 데이터에 대한 전체 전달이 필요합니다.

참고: 이러한 계산은 내부적으로 [Apache Beam](https://beam.apache.org/)에서 구현됩니다.

Apache Beam을 사용하여 값을 계산하는 것 외에도, TensorFlow Transform을 사용하면 사용자가 이러한 값을 TensorFlow 그래프로 삽입한 다음 훈련 그래프에 로드할 수 있습니다. 예를 들어, 특성을 정규화할 때 `tft.scale_to_z_score` 함수는 특성의 평균과 표준 편차를 계산하고, 평균을 빼고 표준 편차로 나누는 함수의 TensorFlow 그래프 표현도 계산합니다. 통계뿐만 아니라 TensorFlow 그래프를 내보냄으로써 TensorFlow Transform은 전처리 파이프라인의 작성 프로세스를 단순화합니다.

전처리가 그래프로 표현되기 때문에 서버에서 발생할 수 있으며 훈련과 적용 간에 일관성이 보장됩니다. 이러한 일관성은 훈련/적용 편향의 원인 하나를 제거합니다.

TensorFlow Transform을 통해 사용자는 TensorFlow 코드를 사용하여 전처리 파이프라인을 지정할 수 있습니다. 이는 파이프라인이 TensorFlow 그래프와 같은 방식으로 구성된다는 것을 의미합니다. 이 그래프에서 TensorFlow 연산만 사용된 경우, 파이프라인은 입력 배치를 받아 출력 배치를 반환하는 순수 맵이 됩니다. 이러한 파이프라인은 `tf.Estimator` API를 사용할 때 `input_fn` 내부에 이 그래프를 배치하는 것과 같습니다. 분위수 계산과 같은 전체 전달 연산을 지정하기 위해 TensorFlow Transform은 TensorFlow 연산처럼 보이는 `analyzers`라는 특수 함수를 제공하지만, 실제로는 Apache Beam에서 수행할 지연된 계산을 지정하고 출력은 상수로 그래프에 삽입됩니다. 일반 TensorFlow 연산은 단일 배치를 입력으로 사용하고 해당 배치에 대해 일부 계산을 수행하고 배치를 내보내지만, `analyzer`는 모든 배치에 대해 전역 감소(Apache Beam에서 구현됨)를 수행하고 결과를 반환합니다.

일반 TensorFlow 연산과 TensorFlow Transform 분석기를 결합하여 사용자는 복잡한 파이프라인을 만들어 데이터를 사전 처리할 수 있습니다. 예를 들어, `tft.scale_to_z_score` 함수는 입력 텐서를 사용하고 평균 `0` 및 분산 `1`을 갖도록 정규화된 해당 텐서를 반환합니다. 내부적으로 `mean` 및 `var` 분석기를 호출하여 이를 수행합니다. 그러면 입력 텐서의 평균 및 분산과 동일한 상수가 그래프 내에 효과적으로 생성됩니다. 그런 다음 TensorFlow 연산을 사용하여 평균을 빼고 표준 편차로 나눕니다.

## TensorFlow Transform `preprocessing_fn`

TFX Transform 구성 요소는 데이터 읽기 및 작성과 관련된 API 호출을 처리하고 출력 SavedModel을 디스크에 작성하여 Transform 사용을 단순화합니다. TFX 사용자는 `preprocessing_fn`이라는 단일 함수만 정의하면 됩니다. `preprocessing_fn`에서 텐서의 출력 사전을 생성하기 위해 텐서의 입력 사전을 조작하는 일련의 함수를 정의합니다. [TensorFlow Transform API](/tfx/transform/api_docs/python/tft)에서 scale_to_0_1 및 compute_and_apply_vocabulary와 같은 도우미 함수를 찾거나 아래와 같이 일반 TensorFlow 함수를 사용할 수 있습니다.

```python
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}
  for key in _DENSE_FLOAT_FEATURE_KEYS:
    # If sparse make it dense, setting nan's to 0 or '', and apply zscore.
    outputs[_transformed_name(key)] = transform.scale_to_z_score(
        _fill_in_missing(inputs[key]))

  for key in _VOCAB_FEATURE_KEYS:
    # Build a vocabulary for this feature.
    outputs[_transformed_name(
        key)] = transform.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]),
            top_k=_VOCAB_SIZE,
            num_oov_buckets=_OOV_SIZE)

  for key in _BUCKET_FEATURE_KEYS:
    outputs[_transformed_name(key)] = transform.bucketize(
        _fill_in_missing(inputs[key]), _FEATURE_BUCKET_COUNT)

  for key in _CATEGORICAL_FEATURE_KEYS:
    outputs[_transformed_name(key)] = _fill_in_missing(inputs[key])

  # Was this passenger a big tipper?
  taxi_fare = _fill_in_missing(inputs[_FARE_KEY])
  tips = _fill_in_missing(inputs[_LABEL_KEY])
  outputs[_transformed_name(_LABEL_KEY)] = tf.where(
      tf.is_nan(taxi_fare),
      tf.cast(tf.zeros_like(taxi_fare), tf.int64),
      # Test if the tip was > 20% of the fare.
      tf.cast(
          tf.greater(tips, tf.multiply(taxi_fare, tf.constant(0.2))), tf.int64))

  return outputs
```

### preprocessing_fn에 대한 입력 이해하기

`preprocessing_fn`은 텐서(즉, `Tensor` 또는 `SparseTensor`)에 대한 일련의 연산을 설명하므로 `preprocessing_fn`을 올바르게 작성하기 위해 데이터가 텐서로 표현되는 방식을 이해해야 합니다. `preprocessing_fn`에 대한 입력은 스키마로 결정됩니다. `Schema` proto에는 `Feature`의 목록이 포함되어 있으며, Transform은 특성 목록을 '특성 사양'('구문 분석 사양'이라고도 함)으로 변환합니다. 이때 특성 사양은 키가 특성 이름이고 값이 `FixedLenFeature` 또는 `VarLenFeature`(또는 TensorFlow Transform에서 사용되지 않는 기타 옵션) 중 하나인 사전입니다.

`Schema`에서 특성 사양을 추론하는 규칙은 다음과 같습니다.

- `shape`가 설정된 각 `feature`의 결과는 형상 및 `default_value=None`이 있는 `tf.FixedLenFeature`입니다.`presence.min_fraction`은 `1`이어야 합니다. 그렇지 않으면 오류가 발생합니다. 기본값이 없으면 `tf.FixedLenFeature`는 특성이 항상 필요하기 때문입니다.
- `shape`가 설정되지 않은 각 `feature`의 결과는 `VarLenFeature`입니다.
- 각 `sparse_feature`의 결과는 `size` 및 `is_sorted`가 `fixed_shape` 및 `SparseFeature` 메시지의 `is_sorted` 필드로 결정되는 `tf.SparseFeature`입니다.
- `sparse_feature`의 `index_feature` 또는 `value_feature`로 사용되는 특성에는 특성 사양에서 생성된 자체 항목이 없습니다.
- `feature`의 `type` 필드(또는 `sparse_feature` proto의 값 특성)와 특성 사양의 `dtype` 간의 대응은 다음 표에 나와 있습니다.

`type` | `dtype`
--- | ---
`schema_pb2.INT` | `tf.int64`
`schema_pb2.FLOAT` | `tf.float32`
`schema_pb2.BYTES` | `tf.string`

## TensorFlow Transform을 사용하여 문자열 레이블 처리하기

일반적으로 TensorFlow Transform을 사용하여 어휘를 생성하고 해당 어휘를 적용하여 문자열을 정수로 변환하려고 합니다. 이 워크플로를 따를 때 모델에 생성된 `input_fn`은 정수화된 문자열을 출력합니다. 그렇지만 레이블은 예외입니다. 모델이 출력(정수) 레이블을 다시 문자열로 매핑할 수 있으려면 모델이 레이블의 가능한 값 목록과 함께 문자열 레이블을 출력하기 위해 `input_fn`이 필요하기 때문입니다. 예를 들어, 레이블이 `cat`과 `dog`이면 `input_fn`의 출력은 이러한 원시 문자열이어야 하며, 이들 키 `["cat", "dog"]`는 매개변수로 estimator에 전달되어야 합니다(아래 세부 사항 참조).

문자열 레이블을 정수로 매핑하려면 TensorFlow Transform을 사용하여 어휘를 생성해야 합니다. 아래 코드 조각에서 이를 보여줍니다.

```python
def _preprocessing_fn(inputs):
  """Preprocess input features into transformed features."""

  ...


  education = inputs[features.RAW_LABEL_KEY]
  _ = tft.vocabulary(education, vocab_filename=features.RAW_LABEL_KEY)

  ...
```

위의 전처리 함수는 원시 입력 특성(전처리 함수의 출력의 일부로도 반환됨)을 사용하고 여기에서 `tft.vocabulary`를 호출합니다. 그 결과 모델에서 액세스할 수 있는 `education`에 대한 어휘가 생성됩니다.

이 예제에서는 또한 레이블을 변환한 다음 변환된 레이블에 대한 어휘를 생성하는 방법을 보여줍니다. 특히 원시 레이블 `education`을 사용하며 레이블을 정수로 변환하지 않고 상위 5개 레이블(빈도별)을 제외한 모든 레이블을 `UNKNOWN`으로 변환합니다.

모델 코드에서 분류자에는 `tft.vocabulary`로 생성된 어휘를 `label_vocabulary` 인수로 제공해야 합니다. 먼저 도우미 함수를 사용하여 이 어휘를 목록으로 읽으면 됩니다. 이는 아래 코드 조각에 나와 있습니다. 예제 코드는 위에서 설명한 변환된 레이블을 사용하지만, 여기에서는 원시 레이블을 사용하는 코드를 보여줍니다.

```python
def create_estimator(pipeline_inputs, hparams):

  ...

  tf_transform_output = trainer_util.TFTransformOutput(
      pipeline_inputs.transform_dir)

  # vocabulary_by_name() returns a Python list.
  label_vocabulary = tf_transform_output.vocabulary_by_name(
      features.RAW_LABEL_KEY)

  return tf.contrib.learn.DNNLinearCombinedClassifier(
      ...
      n_classes=len(label_vocab),
      label_vocabulary=label_vocab,
      ...)
```

## 변환 전 및 변환 후 통계 구성하기

위에서 언급했듯이 Transform 구성 요소는 TFDV를 호출하여 변환 전 및 변환 후 통계를 모두 계산합니다. TFDV는 선택적 [StatsOptions](https://github.com/tensorflow/datavalidation/blob/master/tensorflow_data_validation/statistics/stats_options.py) 객체를 입력으로 사용합니다. 사용자는 특정한 추가 통계(예: NLP 통계)를 활성화하거나 검증된 임계값(예: 최소/최대 토큰 빈도)을 설정하기 위해 이 객체를 구성해야 할 수 있습니다. 그렇게 하려면 모듈 파일에 `stats_options_updater_fn`을 정의합니다.

```python
def stats_options_updater_fn(stats_type, stats_options):
  ...
  if stats_type == stats_options_util.StatsType.PRE_TRANSFORM:
    # Update stats_options to modify pre-transform statistics computation.
    # Most constraints are specified in the schema which can be accessed
    # via stats_options.schema.
  if stats_type == stats_options_util.StatsType.POST_TRANSFORM
    # Update stats_options to modify post-transform statistics computation.
    # Most constraints are specified in the schema which can be accessed
    # via stats_options.schema.
  return stats_options
```

Post-transform statistics often benefit from knowledge of the vocabulary being used for preprocessing a feature. The vocabulary name to path mapping is provided to StatsOptions (and hence TFDV) for every TFT-generated vocabulary. Additionally, mappings for externally-created vocabularies can be added by either (i) directly modifying the `vocab_paths` dictionary within StatsOptions or by (ii) using `tft.annotate_asset`.
