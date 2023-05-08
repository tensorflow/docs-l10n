# 형식별 데이터세트 빌더

[목차]

이 가이드는 TFDS에서 현재 사용할 수 있는 모든 형식별 데이터세트 빌더를 기록합니다.

형식별 데이터세트 빌더는 특정 데이터 형식에 대한 대부분의 데이터 처리를 처리하는 [`tfds.core.GeneratorBasedBuilder`](https://www.tensorflow.org/datasets/api_docs/python/tfds/core/GeneratorBasedBuilder)의 하위 클래스입니다.

## `tf.data.Dataset`를 기반으로 하는 데이터세트

`tf.data.Dataset`([참조](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)) 형식의 데이터세트에서 TFDS 데이터세트를 생성하길 원한다면 `tfds.dataset_builders.TfDataBuilder`( [API 문서](https://www.tensorflow.org/datasets/api_docs/python/tfds/dataset_builders/TfDataBuilder) 참조)를 사용할 수 있습니다.

이 클래스의 일반적인 용도로 다음 두 가지를 구상합니다.

- 노트북과 같은 환경에서 실험적인 데이터세트 생성
- 코드로 데이터세트 빌더 정의

### 노트북에서 새 데이터세트 생성

노트북에서 작업 중이며 여러 변환(맵, 필터 등)을 적용한 몇몇 데이터를 `tf.data.Dataset`로 로드하고, 이제 이 데이터를 팀원과 쉽게 공유하거나 다른 노트북에 로드하길 원한다고 가정해 봅시다. 데이터세트를 TFDS 데이터세트로 저장하기 위해 새 데이터세트 빌더 클래스를 정의하는 대신 `tfds.dataset_builders.TfDataBuilder`를 인스턴스화하고 `download_and_prepare`를 호출할 수 있습니다.

TFDS 데이터세트이므로 이를 버저닝하고 구성을 사용하고 다르게 분할하고 추후에 더 쉽게 사용하도록 문서화할 수 있습니다. 이는 또한 데이터세트에 어떤 기능이 있는지 TFDS에 알려야 한다는 것을 의미합니다.

다음은 사용 방법에 대한 더미 예제입니다.

```python
import tensorflow as tf
import tensorflow_datasets as tfds

my_ds_train = tf.data.Dataset.from_tensor_slices({"number": [1, 2, 3]})
my_ds_test = tf.data.Dataset.from_tensor_slices({"number": [4, 5]})

# Optionally define a custom `data_dir`.
# If None, then the default data dir is used.
custom_data_dir = "/my/folder"

# Define the builder.
single_number_builder = tfds.dataset_builders.TfDataBuilder(
    name="my_dataset",
    config="single_number",
    version="1.0.0",
    data_dir=custom_data_dir,
    split_datasets={
        "train": my_ds_train,
        "test": my_ds_test,
    },
    features=tfds.features.FeaturesDict({
        "number": tfds.features.Scalar(dtype=tf.int64),
    }),
    description="My dataset with a single number.",
    release_notes={
        "1.0.0": "Initial release with numbers up to 5!",
    }
)

# Make the builder store the data as a TFDS dataset.
single_number_builder.download_and_prepare()
```

`download_and_prepare` 메서드는 입력 `tf.data.Dataset`를 되풀이하고 해당 TFDS 데이터세트를 훈련 및 테스트 분할을 모두 포함하는 `/my/folder/my_dataset/single_number/1.0.0`에 저장합니다.

`config` 인수는 선택 사항이며 동일한 데이터세트에 다른 구성을 저장하고자 한다면 유용할 수 있습니다.

`data_dir` 인수는 이를테면 다른 이들과 (아직) 이를 공유하지 않고 싶다면 나만의 샌드박스와 같은 다른 폴더에 생성된 TFDS 데이터세트를 저장하는 데 사용될 수 있습니다. 이를 수행할 때, `data_dir`를 `tfds.load`로 전달하기도 해야 한다는 점을 주의하세요. `data_dir` 인수가 지정되지 않은 경우 기본 TFDS 데이터 디렉터리를 사용합니다.

#### 데이터세트 로드

TFDS 데이터세트가 저장되면 데이터에 대한 액세스 권한이 있는 다른 스크립트나 팀원이 로드할 수 있습니다.

```python
# If no custom data dir was specified:
ds_test = tfds.load("my_dataset/single_number", split="test")

# When there are multiple versions, you can also specify the version.
ds_test = tfds.load("my_dataset/single_number:1.0.0", split="test")

# If the TFDS was stored in a custom folder, then it can be loaded as follows:
custom_data_dir = "/my/folder"
ds_test = tfds.load("my_dataset/single_number:1.0.0", split="test", data_dir=custom_data_dir)
```

#### 새 버전 또는 구성 추가

데이터세트에서 더 반복한 후, 소스 데이터의 변경에 몇 가지를 추가하거나 변경할 수 있습니다. 이 데이터세트를 저장하고 공유하려면 이를 쉽게 새로운 버전으로 저장할 수 있습니다.

```python
def add_one(example):
  example["number"] = example["number"] + 1
  return example

my_ds_train_v2 = my_ds_train.map(add_one)
my_ds_test_v2 = my_ds_test.map(add_one)

single_number_builder_v2 = tfds.dataset_builders.TfDataBuilder(
    name="my_dataset",
    config="single_number",
    version="1.1.0",
    data_dir=custom_data_dir,
    split_datasets={
        "train": my_ds_train_v2,
        "test": my_ds_test_v2,
    },
    features=tfds.features.FeaturesDict({
        "number": tfds.features.Scalar(dtype=tf.int64, doc="Some number"),
    }),
    description="My dataset with a single number.",
    release_notes={
        "1.1.0": "Initial release with numbers up to 6!",
        "1.0.0": "Initial release with numbers up to 5!",
    }
)

# Make the builder store the data as a TFDS dataset.
single_number_builder_v2.download_and_prepare()
```

### 새로운 데이터세트 빌더 클래스 정의

이 클래스를 바탕으로 새로운 `DatasetBuilder`를 정의할 수도 있습니다.

```python
import tensorflow as tf
import tensorflow_datasets as tfds

class MyDatasetBuilder(tfds.dataset_builders.TfDataBuilder):
  def __init__(self):
    ds_train = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    ds_test = tf.data.Dataset.from_tensor_slices([4, 5])
    super().__init__(
      name="my_dataset",
      version="1.0.0",
      split_datasets={
          "train": ds_train,
          "test": ds_test,
      },
      features=tfds.features.FeaturesDict({
          "number": tfds.features.Scalar(dtype=tf.int64),
      }),
      config="single_number",
      description="My dataset with a single number.",
      release_notes={
          "1.0.0": "Initial release with numbers up to 5!",
      }
    )
```

## CoNLL

### 형식

[CoNLL](https://aclanthology.org/W03-0419.pdf)은 주석이 달린 텍스트 데이터를 나타내는 데 사용되는 인기 있는 형식입니다.

CoNLL 형식 데이터는 일반적으로 라인당 언어적인 주석을 포함한 하나의 토큰을 포함합니다. 동일한 라인에, 주석은 일반적으로 공백 또는 탭으로 분리됩니다. 빈 라인은 문장 경계를 나타냅니다.

CoNLL 주석 형식을 따르는 [conll2003](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/conll2003/conll2003.py) 데이터세트의 다음 문장을 예로 들어 봅시다.

```markdown
U.N. NNP I-NP I-ORG official
NN I-NP O
Ekeus NNP I-NP I-PER
heads VBZ I-VP O
for IN I-PP O
Baghdad NNP I-NP
I-LOC . . O O
```

### `ConllDatasetBuilder`

새 CoNLL 기반 데이터세트를 TFDS에 추가하려면 `tfds.dataset_builders.ConllDatasetBuilder`에서 데이터세트 빌더 클래스를 기반으로 할 수 있습니다. 이 기반 클래스는 공통 코드를 포함하여 CoNLL 데이터세트의 특수성(열 기반 형식, 기능 및 태그의 사전 컴파일된 목록 등)을 다룹니다.

`tfds.dataset_builders.ConllDatasetBuilder`는 CoNLL별 `GeneratorBasedBuilder`를 구현합니다. 다음 클래스를 CoNLL 데이터세트 빌더의 최소 예시로 참조하세요.

```python
from tensorflow_datasets.core.dataset_builders.conll import conll_dataset_builder_utils as conll_lib
import tensorflow_datasets.public_api as tfds

class MyCoNNLDataset(tfds.dataset_builders.ConllDatasetBuilder):
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'Initial release.'}

  # conllu_lib contains a set of ready-to-use CONLL-specific configs.
  BUILDER_CONFIGS = [conll_lib.CONLL_2003_CONFIG]

  def _info(self) -> tfds.core.DatasetInfo:
    return self.create_dataset_info(
      # ...
    )

  def _split_generators(self, dl_manager):
    path = dl_manager.download_and_extract('https://data-url')

    return {'train': self._generate_examples(path=path / 'train.txt'),
            'test': self._generate_examples(path=path / 'train.txt'),
    }
```

표준 데이터세트 빌더에 대해 말하자면, 이것은 클래스 메서드 `_info` 및 `_split_generators`를 덮어쓰는 데 필요합니다. 데이터세트에 따라, 데이터세트에 기능과 태그별 목록을 포함하기 위해 [conll_dataset_builder_utils.py](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_builders/conll/conll_dataset_builder_utils.py)를 업데이트해야 할 수도 있습니다.

`_generate_examples` 메서드는 데이터세트가 특정 구현을 필요로 하지 않는 한 추가적인 덮어쓰기가 필요하지 않습니다.

### 예시

[conll2003](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/conll2003/conll2003.py)을 CoNLL별 데이터세트 빌더를 사용하는 구현된 데이터세트 예시로 들어봅시다.

### CLI

새로운 CoNLL 기반 데이터세트를 작성하는 가장 쉬운 방법은 [TFDS CLI](https://www.tensorflow.org/datasets/cli)를 사용하는 것입니다.

```sh
cd path/to/my/project/datasets/
tfds new my_dataset --format=conll   # Create `my_dataset/my_dataset.py` CoNLL-specific template files
```

## CoNLL-U

### 형식

[CoNLL-U](https://universaldependencies.org/format.html)는 주석이 달린 텍스트 데이터를 나타내는 데 사용되는 인기 있는 형식입니다.

CoNLL-U는 [다중 토큰 단어](https://universaldependencies.org/u/overview/tokenization.html)에 대한 지원과 같은 많은 기능을 추가하여 CoNLL 형식을 강화합니다. CoNLL-U 형식 데이터는 일반적으로 라인당 언어적인 주석을 포함한 하나의 토큰을 포함합니다. 동일한 라인에, 주석은 단일 탭 문자로 분리됩니다. 빈 라인은 문장 경계를 나타냅니다.

일반적으로, 각 CoNLL-U 주석이 달린 단어 라인은 [공식 설명서](https://universaldependencies.org/format.html)에서 보고된 대로 다음 필드를 포함합니다.

- ID: 단어 색인, 새로운 각 문장에 대해 1로 시작하는 정수, 다중단어 토큰에 대한 범위, 빈 노드에 대한 10진수일 수 있습니다(십진수는 1보다 작을 수 있지만 0보다 커야 합니다).
- FORM: 어형 또는 구두점 기호.
- LEMMA: 단어의 기본형 또는 어형의 어간.
- UPOS: 범용 품사 태그.
- XPOS: 언어별 품사 태그, 사용할 수 없다면 밑줄을 긋습니다.
- FEATS: 보편적인 특징 목록 또는 정의된 언어별 확장의 형태론적 특성의 목록. 사용할 수 없다면 밑줄을 긋습니다.
- HEAD: ID 값 또는 0인 현재 단어의 어두.
- DEPREL: HEAD(root iff HEAD = 0)에 대한 범용 종속성 관계 또는 정의된 언어별 하위 유형 중 하나.
- DEPS: head-deprel 쌍의 목록 형식으로 강화된 종속성 그래프.
- MISC: 기타 다른 주석.

[공식 설명서](https://universaldependencies.org/format.html)의 다음 CoNLL-U 주석 문장을 예로 들어봅시다.

```markdown
1-2    vámonos   _
1      vamos     ir
2      nos       nosotros
3-4    al        _
3      a         a
4      el        el
5      mar       mar
```

### `ConllUDatasetBuilder`

새 CoNLL-U 기반 데이터세트를 TFDS에 추가하려면 `tfds.dataset_builders.ConllDatasetBuilder`에서 데이터세트 빌더 클래스를 기반으로 할 수 있습니다. 이 기반 클래스는 공통 코드를 포함하여 CoNLL 데이터세트의 특수성(열 기반 형식, 기능 및 태그의 사전 컴파일된 목록 등)을 다룹니다.

`tfds.dataset_builders.ConllDatasetBuilder`는 CoNLL별 `GeneratorBasedBuilder`를 구현합니다. 다음 클래스를 CoNLL-U 데이터세트 빌더의 최소 예시로 참조하세요.

```python
from tensorflow_datasets.core.dataset_builders.conll import conllu_dataset_builder_utils as conllu_lib
import tensorflow_datasets.public_api as tfds

class MyCoNNLUDataset(tfds.dataset_builders.ConllUDatasetBuilder):
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'Initial release.'}

  # conllu_lib contains a set of ready-to-use features.
  BUILDER_CONFIGS = [
      conllu_lib.get_universal_morphology_config(
          language='en',
          features=conllu_lib.UNIVERSAL_DEPENDENCIES_FEATURES,
      )
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    return self.create_dataset_info(
        # ...
    )

  def _split_generators(self, dl_manager):
    path = dl_manager.download_and_extract('https://data-url')

    return {
        'train':
            self._generate_examples(
                path=path / 'train.txt',
                # If necessary, add optional custom processing (see conllu_lib
                # for examples).
                # process_example_fn=...,
            )
    }
```

표준 데이터세트 빌더에 대해 말하자면, 이것은 클래스 메서드 `_info` 및 `_split_generators`를 덮어쓰는 데 필요합니다. 데이터세트에 따라, 데이터세트에 기능과 태그별 목록을 포함하기 위해 [conll_dataset_builder_utils.py](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_builders/conll/conllu_dataset_builder_utils.py)를 업데이트해야 할 수도 있습니다.

`_generate_examples` 메서드는 데이터세트가 특정 구현을 필요로 하지 않는 한 추가적인 덮어쓰기가 필요하지 않습니다. 데이터세트가 특정 전처리, 예를 들면 비고전적인 [범용 종속성 기능](https://universaldependencies.org/guidelines.html)을 고려하는 경우 [`generate_examples`](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_builders/conll/conllu_dataset_builder.py#L192) 함수의 `process_example_fn` 속성을 업데이트해야 할 수 있습니다([xtreme_pos](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/universal_dependencies/xtreme_pos.py) 데이터세트를 예시로 참조하세요).

### 예시

CoNNL-U 특정 데이터세트 빌더를 사용하는 다음 데이터세트를 예시로 들어봅시다.

- [universal_dependencies](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/universal_dependencies/universal_dependencies.py)
- [xtreme_pos](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/universal_dependencies/xtreme_pos.py)

### CLI

데이터세트를 기반으로 새로운 CoNLL-U를 작성하는 가장 쉬운 방법은 [TFDS CLI](https://www.tensorflow.org/datasets/cli)를 사용하는 것입니다.

```sh
cd path/to/my/project/datasets/
tfds new my_dataset --format=conllu   # Create `my_dataset/my_dataset.py` CoNLL-U specific template files
```
