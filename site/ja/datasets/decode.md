# 特徴量のデコードをカスタマイズする

`tfds.decode` API を使うと、デフォルトの特徴量のデコードをオーバーライドすることができます。主なユースケースは、パフォーマンスを改善するために、画像のデコードをスキップすることです。

注意: この API では、ディスク上の低レベルの `tf.train.Example` 形式にアクセスできます（`FeatureConnector` で定義されています）。この API は、画像の読み取り性能の改善を求める高度ユーザーを対象としています。

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
  )
  example['video'] = video
  return example


ds, ds_info = tfds.load('ucf101', split='train', with_info=True, decoders={
    'video': tfds.decode.SkipDecoding(),  # Skip frame decoding
})
ds = ds.map(decode_video)  # Decode the video
```

### 特徴量の一部のみをデコードする

必要な特徴量のみを指定することで、一部の特徴量を完全にスキップすることも可能です。指定されていないすべての特徴量は無視またはスキップされます

```python
builder = tfds.builder('my_dataset')
builder.as_dataset(split='train', decoders=tfds.decode.PartialDecoding({
    'image': True,
    'metadata': {'num_objects', 'scene_name'},
    'objects': {'label'},
})
```

TFDS は、特定の `tfds.decode.PartialDecoding` 構造に一致する `builder.info.features` のサブセットを選択します。

上記のコードでは、特徴量は `builder.info.features` に一致するように暗黙的に抽出されます。また、特徴量を明示的に定義することも可能です。上記のコードは次のコードと同等です。

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

元のメタデータ（ラベル名、画像の形状など）は自動的に再利用されるため、それらを提供する必要はありません。

`tfds.decode.SkipDecoding` は、`PartialDecoding(..., decoders={})` kwargs を通じて `tfds.decode.PartialDecoding` に渡すことができます。
