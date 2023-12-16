# 새 데이터세트 모음 추가

이 가이드에 따라 새 데이터세트 모음을 생성하세요(TFDS 또는 자체 리포지토리 이용).

## 개요

TFDS에 새 데이터세트 모음 `my_collection`을 추가하려면 사용자는 다음 파일을 포함하는 `my_collection` 폴더를 생성해야 합니다.

```sh
my_collection/
  __init__.py
  my_collection.py # Dataset collection definition
  my_collection_test.py # (Optional) test
  description.md # (Optional) collection description (if not included in my_collection.py)
  citations.md # (Optional) collection citations (if not included in my_collection.py)
```

관례대로, 새 데이터세트 모음은 TFDS 리포지토리의 `tensorflow_datasets/dataset_collections/` 폴더에 추가되어야 합니다.

## 데이터세트 모음 작성하기

모든 데이터세트 모음은 [`tfds.core.dataset_collection_builder.DatasetCollection`](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_collection_builder.py)의 하위 클래스로 구현됩니다.

다음은 파일 `my_collection.py`에 정의된 데이터세트 모음 빌더의 최소한의 예시입니다.

```python
import collections
from typing import Mapping
from tensorflow_datasets.core import dataset_collection_builder
from tensorflow_datasets.core import naming

class MyCollection(dataset_collection_builder.DatasetCollection):
  """Dataset collection builder my_dataset_collection."""

  @property
  def info(self) -> dataset_collection_builder.DatasetCollectionInfo:
    return dataset_collection_builder.DatasetCollectionInfo.from_cls(
        dataset_collection_class=self.__class__,
        description="my_dataset_collection description.",
        release_notes={
            "1.0.0": "Initial release",
        },
    )

  @property
  def datasets(
      self,
  ) -> Mapping[str, Mapping[str, naming.DatasetReference]]:
    return collections.OrderedDict({
        "1.0.0":
            naming.references_for({
                "dataset_1": "natural_questions/default:0.0.2",
                "dataset_2": "media_sum:1.0.0",
            }),
        "1.1.0":
            naming.references_for({
                "dataset_1": "natural_questions/longt5:0.1.0",
                "dataset_2": "media_sum:1.0.0",
                "dataset_3": "squad:3.0.0"
            })
    })
```

다음 섹션에서는 덮어쓰기를 위한 2가지 추상 메서드에 대해 설명합니다.

### `info`: 데이터세트 모음 메타데이터

`info` 메서드는 모음의 메타데이터를 포함하는 [`dataset_collection_builder.DatasetCollectionInfo`](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/core/dataset_collection_builder.py#L66)를 반환합니다.

데이터세트 모음 정보는 다음과 같은 네 필드를 포함합니다.

- 이름: 데이터세트 모음의 이름.
- 설명: 데이터세트 모음의 마크다운 형식 설명. 데이터세트 모음 설명을 정의하는 데 두 가지 방법이 있습니다. (1) TFDS 데이터세트에 대해 이미 수행한 것과 유사하게 모음의 `my_collection.py` 파일에서 직접 (여러 줄) 문자열로 설명. (2) 데이터세트 모음 폴더에 배치해야 하는 `description.md` 파일에서 설명.
- release_notes: 데이터세트 모음의 버전에서 해당 릴리즈 노트에 매핑.
- 인용: 데이터세트 모음에 대한 옵션 BibTeX 인용 (목록). 데이터세트 모음 인용을 정의하는 데 두 가지 방법이 있습니다. (1) TFDS 데이터세트에 대해 이미 수행한 것과 유사하게 모음의 `my_collection.py` 파일에서 직접 (여러 줄) 문자열로 설명. (2) 데이터세트 모음 폴더에 배치해야 하는 `citations.bib` 파일에서 설명

### `datasets`: 모음에서 데이터세트 정의

`datasets` 메서드는 모음에서 TFDS 데이터세트를 반환합니다.

데이터세트 모음의 진화를 설명하는 버전의 사전으로 정의됩니다.

각 버전의 경우, 포함된 TFDS 데이터세트는 데이터세트 이름부터 [`naming.DatasetReference`](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/core/naming.py#L187)까지 사전으로 저장됩니다. 예를 들면 다음과 같습니다.

```python
class MyCollection(dataset_collection_builder.DatasetCollection):
  ...
  @property
  def datasets(self):
    return {
        "1.0.0": {
            "yes_no":
                naming.DatasetReference(
                    dataset_name="yes_no", version="1.0.0"),
            "sst2":
                naming.DatasetReference(
                    dataset_name="glue", config="sst2", version="2.0.0"),
            "assin2":
                naming.DatasetReference(
                    dataset_name="assin2", version="1.0.0"),
        },
        ...
    }
```

[`naming.references_for`](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/core/naming.py#L257) 메서드는 위와 동일한 대로 표현하기 위한 보다 간편한 방식을 제공합니다.

```python
class MyCollection(dataset_collection_builder.DatasetCollection):
  ...
  @property
  def datasets(self):
    return {
        "1.0.0":
            naming.references_for({
                "yes_no": "yes_no:1.0.0",
                "sst2": "glue/sst:2.0.0",
                "assin2": "assin2:1.0.0",
            }),
        ...
    }
```

## 데이터세트 모음 단위 테스트

[DatasetCollectionTestBase](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/testing/dataset_collection_builder_testing.py#L28)는 데이터세트 모음에 대한 기본 테스트 클래스입니다. 데이터세트 모음이 올바르게 등록되고 데이터세트가 TFDS에 존재하는지 확인하기 위한 많은 단순한 검사를 제공합니다.

설정할 유일한 클래스 속성은 `DATASET_COLLECTION_CLASS`로, 이는 테스트 할 데이터세트 모음의 클래스 객체를 지정합니다.

또한, 사용자는 다음 클래스 속성을 설정할 수 있습니다.

- `VERSION`: 테스트를 실행하는 데 사용되는 데이터세트 모음의 버전(최신 버전이 기본값)
- `DATASETS_TO_TEST`: TFDS에 존재하는지 테스트할 데이터세트를 포함하는 리스트(모음의 모든 데이터세트가 기본값)
- `CHECK_DATASETS_VERSION`: 데이터세트 모음에 버저닝된 데이터세트가 있는지 또는 기본 버전(기본값은 true)을 확인할지 여부.

데이터세트 모음에 대한 가장 단순한 유효 테스트는 다음과 같습니다.

```python
from tensorflow_datasets.testing.dataset_collection_builder_testing import DatasetCollectionTestBase
from . import my_collection

class TestMyCollection(DatasetCollectionTestBase):
  DATASET_COLLECTION_CLASS = my_collection.MyCollection
```

다음 명령을 실행하여 데이터세트 모음을 테스트합니다.

```sh
python my_dataset_test.py
```

## 피드백

지속해서 데이터세트 생성 워크플로를 개선하려고 시도하고 있지만, 문제에 대해 알고 있는 경우에만 그렇게 할 수 있습니다. 데이터세트 모음을 생성하는 동안 겪은 문제나 오류는 무엇입니까? 헷갈리는 부분이 있었나요, 혹은 처음에 작동하지 않은 부분이 있었나요?

[GitHub](https://github.com/tensorflow/datasets/issues)에 피드백을 공유해 주시길 바랍니다.
