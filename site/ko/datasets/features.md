# 피쳐 커넥터 (특성 커넥터)

`tfds.features.FeatureConnector`의 API:

- `tf.data.Dataset`의 최종 구조, 형상, dtype을 정의합니다.
- to/from 디스크의 직렬화를 추상화합니다.
- 추가 메타 데이터를 드러냅니다. (예: 라벨 이름, 오디오 샘플 속도, ...)

## 개요

`tfds.features.FeatureConnector`는 (`tfds.core.DatasetInfo`)안의 데이터세트 특성 구조를 정의합니다:

```python
tfds.core.DatasetInfo(
    features=tfds.features.FeaturesDict({
        'image': tfds.features.Image(shape=(28, 28, 1), doc='Grayscale image'),
        'label': tfds.features.ClassLabel(
            names=['no', 'yes'],
            doc=tfds.features.Documentation(
                desc='Whether this is a picture of a cat',
                value_range='yes or no'
            ),
        ),
        'metadata': {
            'id': tf.int64,
            'timestamp': tfds.features.Scalar(
                tf.int64,
                doc='Timestamp when this picture was taken as seconds since epoch'),
            'language': tf.string,
        },
    }),
)
```

텍스트 설명(`doc='description'`)만 사용하거나 직접 `tfds.features.Documentation`을 사용하여 보다 자세한 기능 설명을 제공함으로써 특성을 문서화할 수 있습니다.

특성은 다음과 같을 수 있습니다.

- 스칼라 값: `tf.bool`, `tf.string`, `tf.float32`,... 특성을 문서화하려는 경우 `tfds.features.Scalar(tf.int64, doc='description')`도 사용할 수 있습니다.
- `tfds.features.Audio`, `tfds.features.Video`,... (사용할 수 있는 특성 [목록](https://www.tensorflow.org/datasets/api_docs/python/tfds/features?version=nightly) 참조)
- 중첩된 특성의 `dict`: `{'metadata': {'image': Image(), 'description': tf.string}}`,...
- 중첩된 `tfds.features.Sequence`: `Sequence({'image': ..., 'id': ...})`, `Sequence(Sequence(tf.int64))`,...

생성하는 동안 예제는 ` FeatureConnector.encode_example `에 의해 디스크에 적합한 형식 (현재 <code> tf.train.Example </code> 프로토콜 버퍼)으로 자동 직렬화됩니다.

```python
yield {
    'image': '/path/to/img0.png',  # `np.array`, file bytes,... also accepted
    'label': 'yes',  # int (0-num_classes) also accepted
    'metadata': {
        'id': 43,
        'language': 'en',
    },
}
```

데이터세트를 읽을 때 (예를들면 `tfds.load`), 데이터는 자동으로`FeatureConnector.decode_example`사용하여 디코딩 됩니다. 반환된 `tf.data.Dataset` 는`tfds.core.DatasetInfo`에 정의된 `dict`와 일치합니다:

```python
ds = tfds.load(...)
ds.element_spec == {
    'image': tf.TensorSpec(shape=(28, 28, 1), tf.uint8),
    'label': tf.TensorSpec(shape=(), tf.int64),
    'metadata': {
        'id': tf.TensorSpec(shape=(), tf.int64),
        'language': tf.TensorSpec(shape=(), tf.string),
    },
}
```

## 프로토의 직렬화/역직렬화

TFDS는 예시를 `tf.train.Example` 프로토에 직렬화/역직렬화하기 위해 로우 레벨 API를 노출합니다.

`dict[np.ndarray | Path | str | ...]`를 프로토 `bytes`에 직렬화하려면 `features.serialize_example`를 사용합니다.

```python
with tf.io.TFRecordWriter('path/to/file.tfrecord') as writer:
  for ex in all_exs:
    ex_bytes = features.serialize_example(data)
    f.write(ex_bytes)
```

프로토 `bytes`를 `tf.Tensor`에 역직렬화하려면 `features.deserialize_example`를 사용합니다.

```python
ds = tf.data.TFRecordDataset('path/to/file.tfrecord')
ds = ds.map(features.deserialize_example)
```

## 메타 데이터에 접근

메타 데이터(라벨 이름, 형상, dtype,...)의 특성에 접근하고 싶다면 [introduction doc](https://www.tensorflow.org/datasets/overview#access_the_dataset_metadata)를 참조하세요. 예:

```python
ds, info = tfds.load(..., with_info=True)

info.features['label'].names  # ['cat', 'dog', ...]
info.features['label'].str2int('cat')  # 0
```

## 자신만의 `FeatureConnector` 작성하기

[available features](https://www.tensorflow.org/datasets/api_docs/python/tfds/features#classes)에서 누락된 특성이 있다고 생각한다면,  [new issue](https://github.com/tensorflow/datasets/issues)를 열어보세요.

자신만의 피쳐 커넥터(특성 커넥터)를 만들기 위해서는 `tfds.features.FeatureConnector`로 부터 상속 받아 추상 메소드를 구현해야 합니다.

- 특성이 단일 텐서 값인 경우, ` tfds.feature.Tensor {/ code0}에서 상속하고 필요한 경우 <code> super () `을 사용하는 것이 가장 좋습니다. 예는 ` tfds.features.BBoxFeature ` 소스 코드를 참조하세요.
- 특성이 여러 텐서 값을 갖고있는 컨테이너일 경우,  `tfds.features.FeaturesDict`로 부터 상속받고 `super()` 메소드를 사용하여 하위 커넥터를 자동으로 인코딩하는 것이 가장 좋습니다.

`tfds.features.FeatureConnector` 객체는 특성이 사용자에게 제공되는 방식에서 디스크에 인코딩되는 방식을 추상화합니다. 아래는 데이터세트의 추상화 레이어와 원시 데이터세트 파일에서 `tf.data.Dataset` 객체로의 변환을 보여주는 다이어그램입니다.

<p align="center">   <img src="dataset_layers.png" alt="DatasetBuilder abstraction layers" width="700"></p>

자신만의 피쳐 커넥터(특성 커넥터)를 만들기 위해서는`tfds.features.FeatureConnector`를 하위 클래스로 만들고 추상 메소드를 구현하세요:

- `encode_example(input_data)`: 생성기 `_generate_examples()`에서 주어진 데이터를 `tf.train.Example` 호환 데이터로 인코딩하는 방법을 정의합니다. 단일 값 또는 값의 `dict`를 반환할 수 있습니다.
- `decode_example`: `tf.train.Example`에서 읽은 텐서의 데이터를 `tf.data.Dataset`에서 반환된  사용자 텐서로 디코딩하는 방법을 정의합니다.
- `get_tensor_info()`: `tf.data.Dataset`에서 반환된 텐서의 형상/dtype을 나타냅니다. `tfds.features`에서 상속받은 경우 선택사항일 수 있습니다.
- (선택 사항) `get_serialized_info()`: `get_tensor_info()`에서 반환된 정보가 실제로 디스크에 데이터가 기록되는 방식과 다른 경우, `tf.train.Example`의 사양과 일치하도록 `get_serialized_info()`를 덮어써야 합니다.
- `to_json_content`/`from_json_content`: 이 코드는 원본 소스 코드 없이 데이터 세트를 로드하는 데 필요합니다. 예시는 [오디오 특성](https://github.com/tensorflow/datasets/blob/65a76cb53c8ff7f327a3749175bc4f8c12ff465e/tensorflow_datasets/core/features/audio_feature.py#L121)을 참조하세요.

참고: 피쳐 커넥터(특성 커넥터)는 `self.assertFeature` 및 `tfds.testing.FeatureExpectationItem`와 함께 테스트 되어야 합니다. [테스트 예시](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/features/image_feature_test.py)를 살펴보세요.

더 많은 정보를 원한다면,`tfds.features.FeatureConnector` 문서를 살펴보세요. [real examples](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/features)를 살펴보는 것 또한 좋은 방법입니다.
