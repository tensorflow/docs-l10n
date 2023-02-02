# 添加新的数据集集合

按照本指南创建新的数据集集合（可在 TFDS 或您自己的仓库中创建）。

## 概述

要将新的数据集集合 `my_collection` 添加到 TFDS，用户需要生成一个包含以下文件的 `my_collection` 文件夹：

```sh
my_collection/
  __init__.py
  my_collection.py # Dataset collection definition
  my_collection_test.py # (Optional) test
  description.md # (Optional) collection description (if not included in my_collection.py)
  citations.md # (Optional) collection citations (if not included in my_collection.py)
```

按照惯例，应将新的数据集集合添加到 TFDS 仓库的 `tensorflow_datasets/dataset_collections/` 文件夹中。

## 编写数据集集合

所有数据集集合都是 [`tfds.core.dataset_collection_builder.DatasetCollection`](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_collection_builder.py) 的实现子类。

下面是一个数据集集合构建工具的最小示例，在文件 `my_collection.py` 中定义：

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

接下来的部分介绍了要覆盖的 2 个抽象方法。

### `info`：数据集集合元数据

`info` 方法返回包含集合元数据的 [`dataset_collection_builder.DatasetCollectionInfo`](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/core/dataset_collection_builder.py#L66)。

数据集集合信息包含四个字段：

- name：数据集集合的名称。
- description：数据集集合的 markdown 格式描述。可以通过两种方式定义数据集集合的描述：(1) 直接在集合的 `my_collection.py` 文件中作为（多行）字符串 – 与已对 TFDS 数据集执行的操作类似；(2) 在一个`description.md` 文件中，此文件必须置于数据集集合文件夹中。
- release_notes：从数据集集合的版本到相应版本说明的映射。
- citation：数据集集合的可选 BibTeX 引文（列表）。可以通过两种方式定义数据集集合的引文：(1) 直接在集合的 `my_collection.py` 文件中作为（多行）字符串 – 与已对 TFDS 数据集执行的操作类似；(2) 在一个`citations.bib` 文件中，此文件必须置于数据集集合文件夹中。

### `datasets`：定义集合中的数据集

`datasets` 方法返回集合中的 TFDS 数据集。

它被定义为一个版本字典，描述了数据集集合的演变。

对于每个版本，包含的 TFDS 数据集都会存储为从数据集名称到 [`naming.DatasetReference`](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/core/naming.py#L187) 的字典。例如：

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

[`naming.references_for`](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/core/naming.py#L257) 方法提供了一种更简便的方式来表达与上面相同的内容：

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

## 对数据集集合执行单元测试

[DatasetCollectionTestBase](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/testing/dataset_collection_builder_testing.py#L28) 是数据集集合的基础测试类。它提供了许多简单的检查来保证数据集集合已正确注册，并且它的数据集存在于 TFDS 中。

唯一要设置的类特性是 `DATASET_COLLECTION_CLASS`，它指定要测试的数据集集合的类对象。

此外，用户可以设置以下类特性：

- `VERSION`：用于运行测试的数据集集合的版本（默认为最新版本）。
- `DATASETS_TO_TEST`：包含测试在 TFDS 中的存在性的数据集的列表（默认为集合中的所有数据集）。
- `CHECK_DATASETS_VERSION`：是否检查数据集集合中版本化数据集的存在性，或者它们的默认版本（默认为 true）。

对数据集集合最简单的有效测试如下：

```python
from tensorflow_datasets.testing.dataset_collection_builder_testing import DatasetCollectionTestBase
from . import my_collection

class TestMyCollection(DatasetCollectionTestBase):
  DATASET_COLLECTION_CLASS = my_collection.MyCollection
```

运行以下命令以测试数据集集合。

```sh
python my_dataset_test.py
```

## 反馈

我们一直在努力改进数据集创建工作流，但只有在我们意识到这些问题的情况下才能这样做。您在创建数据集集合时遇到了哪些问题或错误？是否有部分令人困惑，或者第一次没有运行？

请在 [GitHub](https://github.com/tensorflow/datasets/issues) 上分享您的反馈。
