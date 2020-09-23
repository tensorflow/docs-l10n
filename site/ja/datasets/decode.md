# 特徴量のデコードをカスタマイズする

- [使用例](#usage-examples)
    - [画像のデコードをスキップする](#skipping-the-image-decoding)
    - [画像をデコードする前にデータセットをフィルタ/シャッフルする](#filtershuffle-dataset-before-images-get-decoded)
    - [クロップとデコードを同時に実行する](#cropping-and-decoding-at-the-same-time)
    - [動画のデコードをカスタマイズする](#customizing-video-decoding)

`tfds.decode` API を使うと、デフォルトの特徴量のデコードをオーバーライドすることができます。主なユースケースは、パフォーマンスを改善するために、画像のデコードをスキップすることです。

警告: この API では、ディスク上の低レベルの `tf.train.Example` 形式にアクセスできます（`FeatureConnector` で定義されています）。この API は、画像の読み取り性能の改善を求める高度ユーザーを対象としています。

## 使用例

### 画像のデコードをスキップする

デコードパイプラインの完全な制御を維持するため、または画像がデコードされる前にフィルタを適用するため（パフォーマンスの改善）に、画像のデコードを完全にスキップすることができます。これは、`tfds.features.Image` と `tfds.features.Video` の両方で機能します。

```python
ds = tfds.load('imagenet2012', split='train', decoders={
    'image': tfds.decode.SkipDecoding(),
})

for example in ds.take(1):
  assert example['image'].dtype == tf.string  # Images are not decoded
```

### 画像をデコードする前にデータセットをフィルタ/シャッフルする

前の例と同様に、`tfds.decode.SkipDecoding()` を使用して、画像をデコードする前に `tf.data` パイプラインのカスタマイズを追加することができます。こうすることで、フィルタされた画像がデコードされなくなるため、より大きなシャッフルバッファを使用することができます。

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

### クロップとデコードを同時に実行する

デフォルトの `tf.io.decode_image` 演算をオーバーライドするには、`tfds.decode.make_decoder()` デコレータを使用して、新しい `tfds.decode.Decoder` オブジェクトを作成することができます。

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

これは次のコードと同等です。

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

### 動画のデコードをカスタマイズする

動画は `Sequence(Image())` です。カスタムデコーダを適用すると、個別のフレームに適用されます。つまり、画像のデコーダには、自動的に動画との互換性があります。

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

これは次のコードと同等です。

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
