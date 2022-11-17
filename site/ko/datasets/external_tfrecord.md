# TFDS로 외부 tfrecord 로드하기

tfds API로 직접 로드하기 원하는 타사 도구로 생성한 `tf.train.Example` 프로토(내부에 `.tfrecord`, `.riegeli`...)가 있는 경우 이 페이지의 내용을 확인하세요.

`.tfrecord` 파일을 로드하려는 경우 다음만 수행하면 됩니다.

- TFDS 명명 규칙을 따릅니다.
- tfrecord 파일과 함께 메타데이터 파일(`dataset_info.json`, `features.json`)을 추가합니다.

제한 사항:

- `tf.train.SequenceExample`는 지원되지 않습니다. `tf.train.Example`만 지원됩니다.
- `tfds.features`로 `tf.train.Example`을 표현할 수 있어야 합니다(아래 섹션 참조).

## 파일 명명 규칙

TFDS supports defining a template for file names, which provides flexibility to use different file naming schemes. The template is represented by a `tfds.core.ShardedFileTemplate` and supports the following variables: `{DATASET}`, `{SPLIT}`, `{FILEFORMAT}`, `{SHARD_INDEX}`, `{NUM_SHARDS}`, and `{SHARD_X_OF_Y}`. For example, the default file naming scheme of TFDS is: `{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}`. For MNIST, this means that [file names](https://console.cloud.google.com/storage/browser/tfds-data/datasets/mnist/3.0.1) look as follows:

- `mnist-test.tfrecord-00000-of-00001`
- `mnist-train.tfrecord-00000-of-00001`

## 메타데이터 추가하기

### 특성 구조 제공하기

TFDS가 `tf.train.Example` 프로토를 디코딩할 수 있게 하려면 사양과 일치하는 `tfds.features` 구조를 제공해야 합니다. 예시:

```python
features = tfds.features.FeaturesDict({
    'image':
        tfds.features.Image(
            shape=(256, 256, 3),
            doc='Picture taken by smartphone, downscaled.'),
    'label':
        tfds.features.ClassLabel(names=['dog', 'cat']),
    'objects':
        tfds.features.Sequence({
            'camera/K': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
        }),
})
```

이는 다음 `tf.train.Example` 사양에 해당합니다.

```python
{
    'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
    'label': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
    'objects/camera/K': tf.io.FixedLenSequenceFeature(shape=(3,), dtype=tf.int64),
}
```

Specifying the features allow TFDS to automatically decode images, video,... Like any other TFDS datasets, features metadata (e.g. label names,...) will be exposed to the user (e.g. `info.features['label'].names`).

#### 직접 생성 파이프라인을 제어하는 경우

TFDS 외부에서 데이터세트를 생성하지만 여전히 직접 생성 파이프라인을 제어하는 경우 `tfds.features.FeatureConnector.serialize_example`을 사용하여 `dict[np.ndarray]`에서 `tf.train.Example` 프로토 `bytes`으로 데이터를 인코딩할 수 있습니다.

```python
with tf.io.TFRecordWriter('path/to/file.tfrecord') as writer:
  for ex in all_exs:
    ex_bytes = features.serialize_example(data)
    writer.write(ex_bytes)
```

이 경우 TFDS와의 특성 호환성이 보장됩니다.

마찬가지로 `feature.deserialize_example`는 프로토([example](https://www.tensorflow.org/datasets/features#serializedeserialize_to_proto))를 디코딩하기 위해 존재합니다.

#### 직접 생성 파이프라인을 제어하지 않는 경우

`tfds.features`가 `tf.train.Example`에서 표시되는 방식은 colab에서 확인할 수 있습니다.

- 사람이 읽을 수 있는 `tf.train.Example`의 구조로 `tfds.features`를 변환하기 위해 `features.get_serialized_info()`를 호출할 수 있습니다.
- `tf.io.parse_single_example`로 전달된 정확한 `FixedLenFeature`,... 사양을 가져오기 위해 `spec = features.tf_example_spec`을 사용할 수 있습니다.

Note: If you're using custom feature connector, make sure to implement `to_json_content`/`from_json_content` and test with `self.assertFeature` (see [feature connector guide](https://www.tensorflow.org/datasets/features#create_your_own_tfdsfeaturesfeatureconnector))

### 분할에 대한 통계 가져오기

TFDS는 각 샤드 내의 정확한 예시 수량을 요구합니다. 이는 `len(ds)` 혹은 [하위 분할 API](https://www.tensorflow.org/datasets/splits): `split='train[75%:]'`와 같은 기능에 필요합니다.

- 이 정보가 있는 경우 명시적으로 `tfds.core.SplitInfo` 목록을 만들고 다음 섹션으로 넘어갈 수 있습니다.

    ```python
    split_infos = [
        tfds.core.SplitInfo(
            name='train',
            shard_lengths=[1024, ...],  # Num of examples in shard0, shard1,...
            num_bytes=0,  # Total size of your dataset (if unknown, set to 0)
        ),
        tfds.core.SplitInfo(name='test', ...),
    ]
    ```

- If you do not know this information, you can compute it using the `compute_split_info.py` script (or in your own script with `tfds.folder_dataset.compute_split_info`). It will launch a beam pipeline which will read all shards on the given directory and compute the info.

### 메타데이터 파일 추가하기

To automatically add the proper metadata files along your dataset, use `tfds.folder_dataset.write_metadata`:

```python
tfds.folder_dataset.write_metadata(
    data_dir='/path/to/my/dataset/1.0.0/',
    features=features,
    # Pass the `out_dir` argument of compute_split_info (see section above)
    # You can also explicitly pass a list of `tfds.core.SplitInfo`.
    split_infos='/path/to/my/dataset/1.0.0/',
    # Pass a custom file name template or use None for the default TFDS
    # file name template.
    filename_template='{SPLIT}-{SHARD_X_OF_Y}.{FILEFORMAT}',

    # Optionally, additional DatasetInfo metadata can be provided
    # See:
    # https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetInfo
    description="""Multi-line description."""
    homepage='http://my-project.org',
    supervised_keys=('image', 'label'),
    citation="""BibTex citation.""",
)
```

Once the function has been called once on your dataset directory, metadata files ( `dataset_info.json`,...) have been added and your datasets are ready to be loaded with TFDS (see next section).

## Load dataset with TFDS

### 폴더에서 직접

Once the metadata have been generated, datasets can be loaded using `tfds.builder_from_directory` which returns a `tfds.core.DatasetBuilder` with the standard TFDS API (like `tfds.builder`):

```python
builder = tfds.builder_from_directory('~/path/to/my_dataset/3.0.0/')

# Metadata are avalailable as usual
builder.info.splits['train'].num_examples

# Construct the tf.data.Dataset pipeline
ds = builder.as_dataset(split='train[75%:]')
for ex in ds:
  ...
```

### 여러 폴더에서 직접

It is also possible to load data from multiple folders. This can happen, for example, in reinforcement learning when multiple agents are each generating a separate dataset and you want to load all of them together. Other use cases are when a new dataset is produced on a regular basis, e.g. a new dataset per day, and you want to load data from a date range.

To load data from multiple folders, use `tfds.builder_from_directories`, which returns a `tfds.core.DatasetBuilder` with the standard TFDS API (like `tfds.builder`):

```python
builder = tfds.builder_from_directories(builder_dirs=[
    '~/path/my_dataset/agent1/1.0.0/',
    '~/path/my_dataset/agent2/1.0.0/',
    '~/path/my_dataset/agent3/1.0.0/',
])

# Metadata are avalailable as usual
builder.info.splits['train'].num_examples

# Construct the tf.data.Dataset pipeline
ds = builder.as_dataset(split='train[75%:]')
for ex in ds:
  ...
```

참고: 각 폴더에는 분할에 대한 정보가 포함된 자체 메타데이터가 있어야 합니다.

### 폴더 구조(선택 사항)

TFDS와의 더 나은 호환성을 위해 데이터를 `<data_dir>/<dataset_name>[/<dataset_config>]/<dataset_version>`와 같이 구성할 수 있습니다. 예시:

```
data_dir/
    dataset0/
        1.0.0/
        1.0.1/
    dataset1/
        config0/
            2.0.0/
        config1/
            2.0.0/
```

This will make your datasets compatible with the `tfds.load` / `tfds.builder` API, simply by providing `data_dir/`:

```python
ds0 = tfds.load('dataset0', data_dir='data_dir/')
ds1 = tfds.load('dataset1/config0', data_dir='data_dir/')
```
