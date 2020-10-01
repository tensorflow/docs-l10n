# 특성 디코딩 사용자 정의하기

- 사용 예
    - [이미지 디코딩 건너뛰기](#skipping-the-image-decoding)
    - [이미지가 디코딩되기 전에 데이터세트 필터링/셔플링하기](#filtershuffle-dataset-before-images-get-decoded)
    - [ 동시에 자르기 및 디코딩하기](#cropping-and-decoding-at-the-same-time)
    - [비디오 디코딩 사용자 정의하기](#customizing-video-decoding)

`tfds.decode` API를 사용하면 기본 특성 디코딩을 재정의할 수 있습니다. 주요 사용 사례는 더 나은 성능을 위해 이미지 디코딩을 건너뛰는 것입니다.

경고: 이 API를 사용하면 디스크에서 하위 레벨 `tf.train.Example` 형식에 액세스할 수 있습니다(`FeatureConnector`에서 정의된 대로). 이 API는 이미지에서 더 나은 읽기 성능을 원하는 고급 사용자를 대상으로 합니다.

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
      back_prop=False,
  )
  example['video'] = video
  return example


ds, ds_info = tfds.load('ucf101', split='train', with_info=True, decoders={
    'video': tfds.decode.SkipDecoding(),  # Skip frame decoding
})
ds = ds.map(decode_video)  # Decode the video
```
