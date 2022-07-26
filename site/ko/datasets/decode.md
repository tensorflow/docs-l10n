# 특성 디코딩 사용자 정의하기

`tfds.decode` API를 사용하면 기본 특성 디코딩을 재정의할 수 있습니다. 주요 사용 사례는 더 나은 성능을 위해 이미지 디코딩을 건너뛰는 것입니다.

참고: 이 API를 사용하면 디스크의 하위 레벨 `tf.train.Example` 형식에 액세스할 수 있습니다(`FeatureConnector`에 의해 정의됨). 이 API는 이미지로 더 나은 읽기 성능을 원하는 고급 사용자를 대상으로 합니다.

## 사용 예

### 이미지 디코딩 건너뛰기

디코딩 파이프라인을 완전히 제어하거나 (성능 향상을 위해) 이미지가 디코딩되기 전에 필터를 적용하기 위해 이미지 디코딩을 완전히 건너뛸 수 있습니다. 이것은 `tfds.features.Image` 및 `tfds.features.Video` 모두에서 동작합니다.

```python
ds = tfds.load('imagenet2012', split='train', decoders={
    'image': tfds.decode.SkipDecoding(),
})

for example in ds.take(1):
  assert example['image'].dtype == tf.string  # Images are not decoded
```

### 이미지가 디코딩되기 전에 데이터세트 필터링/셔플링하기

이전 예제와 마찬가지로, `tfds.decode.SkipDecoding()`을 사용하여 이미지를 디코딩하기 전에 추가 `tf.data` 파이프라인 사용자 정의을 삽입할 수 있습니다. 이렇게 하면 필터링된 이미지가 디코딩되지 않으며 더 큰 셔플 버퍼를 사용할 수 있습니다.

```python
# Load the base dataset without decoding
ds, ds_info = tfds.load(
    'imagenet2012',
    split='train',
    decoders={
        'image': tfds.decode.SkipDecoding(),  # Image won't be decoded here
    },
    as_supervised=True,
    with_info=True,
)
# Apply filter and shuffle
ds = ds.filter(lambda image, label: label != 10)
ds = ds.shuffle(10000)
# Then decode with ds_info.features['image']
ds = ds.map(
    lambda image, label: ds_info.features['image'].decode_example(image), label)
```

### 동시에 자르기 및 디코딩하기

기본 `tf.io.decode_image` 연산을 재정의하기 위해 `tfds.decode.make_decoder()` 데코레이터를 사용하여 새 `tfds.decode.Decoder` 객체를 작성할 수 있습니다.

```python
@tfds.decode.make_decoder()
def decode_example(serialized_image, feature):
  crop_y, crop_x, crop_height, crop_width = 10, 10, 64, 64
  return tf.image.decode_and_crop_jpeg(
      serialized_image,
      [crop_y, crop_x, crop_height, crop_width],
      channels=feature.feature.shape[-1],
  )

ds = tfds.load('imagenet2012', split='train', decoders={
    # With video, decoders are applied to individual frames
    'image': decode_example(),
})
```

다음과 같습니다.

```python
def decode_example(serialized_image, feature):
  crop_y, crop_x, crop_height, crop_width = 10, 10, 64, 64
  return tf.image.decode_and_crop_jpeg(
      serialized_image,
      [crop_y, crop_x, crop_height, crop_width],
      channels=feature.shape[-1],
  )

ds, ds_info = tfds.load(
    'imagenet2012',
    split='train',
    with_info=True,
    decoders={
        'image': tfds.decode.SkipDecoding(),  # Skip frame decoding
    },
)
ds = ds.map(functools.partial(decode_example, feature=ds_info.features['image']))
```

### 비디오 디코딩 사용자 정의하기

비디오는 `Sequence(Image())`입니다. 사용자 정의 디코더를 적용할 때 개별 프레임에 적용됩니다. 이는 이미지용 디코더가 비디오와 자동으로 호환됨을 의미합니다.

```python
@tfds.decode.make_decoder()
def decode_example(serialized_image, feature):
  crop_y, crop_x, crop_height, crop_width = 10, 10, 64, 64
  return tf.image.decode_and_crop_jpeg(
      serialized_image,
      [crop_y, crop_x, crop_height, crop_width],
      channels=feature.feature.shape[-1],
  )

ds = tfds.load('ucf101', split='train', decoders={
    # With video, decoders are applied to individual frames
    'video': decode_example(),
})
```

다음과 같습니다.

```python
def decode_frame(serialized_image):
  """Decodes a single frame."""
  crop_y, crop_x, crop_height, crop_width = 10, 10, 64, 64
  return tf.image.decode_and_crop_jpeg(
      serialized_image,
      [crop_y, crop_x, crop_height, crop_width],
      channels=ds_info.features['video'].shape[-1],
  )


def decode_video(example):
  """Decodes all individual frames of the video."""
  video = example['video']
  video = tf.map_fn(
      decode_frame,
      video,
      dtype=ds_info.features['video'].dtype,
      parallel_iterations=10,
  )
  example['video'] = video
  return example


ds, ds_info = tfds.load('ucf101', split='train', with_info=True, decoders={
    'video': tfds.decode.SkipDecoding(),  # Skip frame decoding
})
ds = ds.map(decode_video)  # Decode the video
```

### 요소의 하위 집합만 디코딩합니다.

필요한 요소만 지정하여 일부 요소를 완전히 건너뛸 수도 있습니다. 다른 모든 요소는 무시/건너뜁니다.

```python
builder = tfds.builder('my_dataset')
builder.as_dataset(split='train', decoders=tfds.decode.PartialDecoding({
    'image': True,
    'metadata': {'num_objects', 'scene_name'},
    'objects': {'label'},
})
```

TFDS는 주어진 `tfds.decode.PartialDecoding` 구조와 일치하는 `builder.info.features`의 하위 집합을 선택합니다.

위의 코드에서 `builder.info.features`와 일치하도록 요소가 암시적으로 추출됩니다. 요소를 명시적으로 정의하는 것도 가능합니다. 위의 코드는 다음과 같습니다.

```python
builder = tfds.builder('my_dataset')
builder.as_dataset(split='train', decoders=tfds.decode.PartialDecoding({
    'image': tfds.features.Image(),
    'metadata': {
        'num_objects': tf.int64,
        'scene_name': tfds.features.Text(),
    },
    'objects': tfds.features.Sequence({
        'label': tfds.features.ClassLabel(names=[]),
    }),
})
```

원본 메타데이터(레이블 이름, 이미지 모양 등)는 자동으로 재사용되므로 제공할 필요가 없습니다.

`tfds.decode.SkipDecoding`은 `PartialDecoding(..., decoders={})` kwargs를 통해 `tfds.decode.PartialDecoding`으로 전달할 수 있습니다.
