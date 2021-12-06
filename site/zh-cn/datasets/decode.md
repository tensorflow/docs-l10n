# 自定义特征解码

使用 `tfds.decode` API，您可以重写默认特征解码。主要用例是跳过图像解码以获得更高的性能。

注：此 API 支持访问磁盘上的低级别 `tf.train.Example` 格式（由 `FeatureConnector` 定义）。此 API 面向希望在图像方面获得更高读取性能的高级用户。

## 用法示例

### 跳过图像解码

为了完全控制解码流水线，或者在对图像进行解码之前应用筛选器（以获得更高的性能），您可以完全跳过图像解码。这适用于 `tfds.features.Image` 和 `tfds.features.Video`。

```python
ds = tfds.load('imagenet2012', split='train', decoders={
    'image': tfds.decode.SkipDecoding(),
})

for example in ds.take(1):
  assert example['image'].dtype == tf.string  # Images are not decoded
```

### 在解码图像之前筛选数据集/打乱数据集顺序

与上一个示例类似，您可以在解码图像之前使用 `tfds.decode.SkipDecoding()` 以插入其他 `tf.data` 流水线自定义。这样，筛选的图像将不会被解码，您可以使用更大的随机缓冲区。

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

### 同时裁剪和解码

要重写默认的 `tf.io.decode_image` 运算，您可以使用 `tfds.decode.make_decoder()` 装饰器创建新的 `tfds.decode.Decoder` 对象。

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

等效于：

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

### 自定义视频解码

视频为 `Sequence(Image())`。当应用自定义解码器时，它们将应用于单独的帧。这意味着图像的解码器会自动与视频兼容。

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

等效于：

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

### 仅解码特征的子集。

也可以通过仅指定您需要的特征来完全跳过某些特征。此时将忽略/跳过所有其他特征。

```python
builder = tfds.builder('my_dataset')
builder.as_dataset(split='train', decoders=tfds.decode.PartialDecoding({
    'image': True,
    'metadata': {'num_objects', 'scene_name'},
    'objects': {'label'},
})
```

TFDS 将选择与给定 `tfds.decode.PartialDecoding` 结构匹配的 `builder.info.features` 的子集。

在上面的代码中，会隐式提取特征以匹配 `builder.info.features`。此外，也可以显式定义特征。上面的代码等价于：

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

原始元数据（标签名称、图像形状…）会自动重用，因此不需要提供它们。

`tfds.decode.SkipDecoding` 可以通过 `PartialDecoding(..., decoders={})` kwarg 传递给 `tfds.decode.PartialDecoding`。
