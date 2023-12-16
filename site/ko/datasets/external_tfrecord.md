# TFDS로 외부 tfrecord 로드하기

tfds API로 직접 로드하기 원하는 타사 도구로 생성한 `tf.train.Example` 프로토(내부에 `.tfrecord`, `.riegeli`...)가 있는 경우 이 페이지의 내용을 확인하세요.

`.tfrecord` 파일을 로드하려는 경우 다음만 수행하면 됩니다.

- TFDS 명명 규칙을 따릅니다.
- tfrecord 파일과 함께 메타데이터 파일(`dataset_info.json`, `features.json`)을 추가합니다.

제한 사항:

- `tf.train.SequenceExample`는 지원되지 않습니다. `tf.train.Example`만 지원됩니다.
- `tfds.features`로 `tf.train.Example`을 표현할 수 있어야 합니다(아래 섹션 참조).

## 파일 명명 규칙

TFDS는 파일 이름에 대한 템플릿 정의를 지원하므로 다양한 파일 명명 체계를 유연하게 사용할 수 있습니다. 템플릿은 `tfds.core.ShardedFileTemplate`로 표시되며 `{DATASET}`, `{SPLIT}`, `{FILEFORMAT}`, `{SHARD_INDEX}`, `{NUM_SHARDS}`, 및 `{SHARD_X_OF_Y}`과 같은 변수를 지원합니다. 예를 들어, TFDS의 기본 파일 명명 체계는 `{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}`입니다. MNIST의 경우, 이는 [파일 이름](https://console.cloud.google.com/storage/browser/tfds-data/datasets/mnist/3.0.1)이 다음과 같이 표시된다는 것을 의미합니다.

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

특성을 지정하면 TFDS가 이미지, 동영상 등을 자동으로 디코딩할 수 있습니다. 다른 TFDS 데이터세트와 마찬가지로 특성 메타데이터(예: 레이블 이름 등)가 사용자에게 노출됩니다(예: `info.features['label'].names`).

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

참고: 사용자 정의 특성 커넥터를 사용하는 경우 `to_json_content`/`from_json_content`를 구현한 후 `self.assertFeature`를 사용하여 테스트합니다([특성 커넥터 가이드](https://www.tensorflow.org/datasets/features#create_your_own_tfdsfeaturesfeatureconnector) 참조).

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

- 이 정보를 모르는 경우 `compute_split_info.py` 스크립트(또는 `tfds.folder_dataset.compute_split_info`를 사용하는 자체 스크립트)를 사용하여 계산할 수 있습니다. 이러한 스크립트는 제공된 디렉터리에 있는 모든 샤드를 읽고 정보를 계산하는 빔 파이프라인을 시작합니다.

### 메타데이터 파일 추가하기

데이터세트에 따른 적절한 메타데이터 파일을 자동으로 추가하려면 다음과 같이 `tfds.folder_dataset.write_metadata`를 사용합니다.

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

데이터세트 디렉터리에서 함수를 호출하면 메타데이터 파일(`dataset_info.json`,...)이 추가되고 TFDS를 사용하여 데이터세트를 로드할 준비가 완료됩니다(다음 섹션 참조).

## TFDS를 사용하여 데이터세트 로드하기

### 폴더에서 직접

메타데이터가 생성되면 표준 TFDS API(예: `tfds.builder`)와 함께 `tfds.core.DatasetBuilder`를 반환하는 `tfds.builder_from_directory`를 사용하여 데이터세트를 로드할 수 있습니다.

```python
builder = tfds.builder_from_directory('~/path/to/my_dataset/3.0.0/')

# Metadata are available as usual
builder.info.splits['train'].num_examples

# Construct the tf.data.Dataset pipeline
ds = builder.as_dataset(split='train[75%:]')
for ex in ds:
  ...
```

### 여러 폴더에서 직접

여러 폴더에서 데이터를 로드할 수도 있습니다. 예를 들어 강화 훈 수행 시 다수의 에이전트가 각각 별도의 데이터세트를 생성한 후 모든 데이터세트를 함께 로드하려는 경우가 이 기능이 필요합니다. 또한 매일 데이터세트를 생성하고 특정 날짜 범위의 데이터를 로드하는 경우와 같이 정기적으로 데이터세트가 생성되는 경우에도 이 기능이 필요합니다.

여러 폴더에서 데이터세트를 로드하려면 표준 TFDS API(예: `tfds.builder`)와 함께 `tfds.core.DatasetBuilder`를 반환하는 `tfds.builder_from_directory`를 사용합니다.

```python
builder = tfds.builder_from_directories(builder_dirs=[
    '~/path/my_dataset/agent1/1.0.0/',
    '~/path/my_dataset/agent2/1.0.0/',
    '~/path/my_dataset/agent3/1.0.0/',
])

# Metadata are available as usual
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

이 경우 `data_dir/`만 제공해도 데이터세트가 `tfds.load` / `tfds.builder` API와 호환됩니다.

```python
ds0 = tfds.load('dataset0', data_dir='data_dir/')
ds1 = tfds.load('dataset1/config0', data_dir='data_dir/')
```
