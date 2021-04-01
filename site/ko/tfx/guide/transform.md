# Transform TFX 파이프라인 구성 요소

Transform TFX 파이프라인 구성 요소는 [SchemaGen](examplegen.md) 구성 요소로 만들어진 데이터 스키마를 사용하여 [ExampleGen](schemagen.md) 구성 요소에서 내보낸 tf.Examples에 대한 특성 엔지니어링(feature engineering)을 수행하고 SavedModel을 내보냅니다. 실행 시 SavedModel은 ExampleGen 구성 요소에서 내보낸 tf.Examples를 허용하고 변환된 특성 데이터를 내보냅니다.

- 입력: ExampleGen 구성 요소의 tf.Examples 및 SchemaGen 구성 요소의 데이터 스키마
- 출력: SavedModel을 Trainer 구성 요소로

## Transform 구성 요소 구성하기

`preprocessing_fn`이 작성되면 Transform 구성 요소에 입력으로 제공되는 Python 모듈에서 정의되어야 합니다. 이 모듈은 변환에 의해 로드되고 `preprocessing_fn`이라는 함수가 발견되며 Transform에서 전처리 파이프라인을 구성하는 데 사용됩니다.

```
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(_taxi_transform_module_file))
```

## Transform 및 TensorFlow Transform

Transform은 데이터세트에서 특성 엔지니어링을 수행하기 위해 [TensorFlow Transform](tft.md)을 광범위하게 사용합니다. TensorFlow Transform은 특성 데이터가 모델로 이동하기 전에 훈련 프로세스의 일부로 특성 데이터를 변환할 수 있는 훌륭한 도구입니다. 일반적인 특성 변환은 다음과 같습니다.

- **Embedding**: converting sparse features (like the integer IDs produced by a vocabulary) into dense features by finding a meaningful mapping from high- dimensional space to low dimensional space. See the [Embeddings unit in the Machine-learning Crash Course](https://developers.google.com/machine-learning/crash-course/embedding) for an introduction to embeddings.
- **어휘 생성**: 각 고유 값을 ID 번호에 매핑하는 어휘를 만들어 문자열 또는 숫자가 아닌 기타 특성을 정수로 변환합니다.
- **값 정규화**: 숫자 특성을 모두 유사한 범위에 속하도록 변환합니다.
- **버킷화** 이산 버킷에 값을 할당하여 범주 특성으로 연속 값 특성을 변환합니다.
- **Enriching text features**: producing features from raw data like tokens, n-grams, entities, sentiment, etc., to enrich the feature set.

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

By combining ordinary TensorFlow ops and TensorFlow Transform analyzers, users can create complex pipelines to preprocess their data. For example the `tft.scale_to_z_score` function takes an input tensor and returns that tensor normalized to have mean `0` and variance `1`. It does this by calling the `mean` and `var` analyzers under the hood, which will effectively generate constants in the graph equal to the mean and variance of the input tensor. It will then use TensorFlow ops to subtract the mean and divide by the standard deviation.

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
    # Preserve this feature as a dense float, setting nan's to the mean.
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
- The correspondence between `type` field of the `feature` (or the values feature of a `sparse_feature` proto) and the `dtype` of the feature spec is given by the following table:

`type` | `dtype`
--- | ---
`schema_pb2.INT` | `tf.int64`
`schema_pb2.FLOAT` | `tf.float32`
`schema_pb2.BYTES` | `tf.string`

## TensorFlow Transform을 사용하여 문자열 레이블 처리하기

Usually one wants to use TensorFlow Transform to both generate a vocabulary and apply that vocabulary to convert strings to integers. When following this workflow, the `input_fn` constructed in the model will output the integerized string. However labels are an exception, because in order for the model to be able to map the output (integer) labels back to strings, the model needs the `input_fn` to output a string label, together with a list of possible values of the label. E.g. if the labels are `cat` and `dog` then the output of the `input_fn` should be these raw strings, and the keys `["cat", "dog"]` need to be passed into the estimator as a parameter (see details below).

문자열 레이블을 정수로 매핑하려면 TensorFlow Transform을 사용하여 어휘를 생성해야 합니다. 아래 코드 조각에서 이를 보여줍니다.

```python
def _preprocessing_fn(inputs):
  """Preprocess input features into transformed features."""

  ...


  education = inputs[features.RAW_LABEL_KEY]
  _ = tft.uniques(education, vocab_filename=features.RAW_LABEL_KEY)

  ...
```

위의 전처리 함수는 원시 입력 특성(전처리 함수의 출력의 일부로 반환됨)을 사용하고 `tft.uniques`를 호출합니다. 그 결과 모델에서 액세스할 수 있는 `education`을 위해 어휘가 생성됩니다.

The example also shows how to transform a label and then generate a vocabulary for the transformed label. In particular it takes the raw label `education` and converts all but the top 5 labels (by frequency) to `UNKNOWN`, without converting the label to an integer.

모델 코드에서 분류자에는 `tft.uniques`로 생성된 어휘를 `label_vocabulary` 인수로 제공해야 합니다. 먼저 도우미 함수를 사용하여 이 어휘를 목록으로 읽으면 됩니다. 이는 아래 코드 조각에 나와 있습니다. 예제 코드는 위에서 설명한 변환된 레이블을 사용하지만, 여기에서는 원시 레이블을 사용하는 코드를 보여줍니다.

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
