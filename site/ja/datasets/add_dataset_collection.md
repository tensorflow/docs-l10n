# 新しいデータセットコレクションを追加する

新しいデータセットコレクションを作成する（TFDS または独自のリポジトリ）には、このガイドに従います。

## 概要

新しいデータセットコレクション `my_collection` を TFDS に追加するには、以下のファイルを含む `my_collection` フォルダを生成する必要があります。

```sh
my_collection/
  __init__.py
  my_collection.py # Dataset collection definition
  my_collection_test.py # (Optional) test
  description.md # (Optional) collection description (if not included in my_collection.py)
  citations.md # (Optional) collection citations (if not included in my_collection.py)
```

原則として、新しいデータセットコレクションは、TFDS リポジトリの `tensorflow_datasets/dataset_collections/` フォルダに追加する必要があります。

## データセットコレクションを書く

すべてのデータセットコレクションは、[`tfds.core.dataset_collection_builder.DatasetCollection`](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_collection_builder.py) のサブクラスに実装されます。

以下は、データセットコレクションビルダーの最低限の例です。`my_collection.py` ファイルに定義されています。

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

次のセクションでは、上書きする 2 つの抽象メソッドを説明します。

### `info`: データセットコレクションのメタデータ

`info` メソッドは、コレクションのメタデータを含む [`dataset_collection_builder.DatasetCollectionInfo`](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/core/dataset_collection_builder.py#L66) を返します。

データセットコレクションの info には、以下の 4 つのフィールドが含まれます。

- name: データセットコレクションの名前。
- description: マークダウンでフォーマットされたデータセットコレクションの説明。データセットの description を定義するには、次の 2 つの方法があります。（1）コレクションの `my_collection.py` ファイルに（複数行の）文字列を直接書き込みます。TFDS データセットで行う方法に似ています。（2）`description.md` ファイルに書き込み、データセットコレクションのフォルダに配置します。
- release_notes: データセットコレクションのバージョンから対応するリリースノートへのマッピング。
- citation: データセットコレクションに関するオプションの BibTeX 引用（リスト）。データセットコレクションの citation を定義するには、次の 2 つの方法があります。（1）コレクションの `my_collection.py` ファイルに（複数行の）文字列を直接書き込みます。TFDS データセットで行う方法に似ています。（2）`citations.bib` ファイルに書き込み、データセットコレクションのフォルダに配置します。

### `datasets`: コレクションのデータセットを定義する

`datasets` メソッドは、コレクションの TFDS データセットを返します。

バージョンのディクショナリとして定義されており、データセットコレクションの進化が記述されます。

バージョンごとに、含まれている TFDS データセットはデータセットから [`naming.DatasetReference`](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/core/naming.py#L187) にディクショナリとして保存されます。以下に例を示します。

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

[`naming.references_for`](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/core/naming.py#L257) メソッドは、上記と同じことをよりコンパクトに表現します。

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

## データセットコレクションをユニットテストする

[DatasetCollectionTestBase](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/testing/dataset_collection_builder_testing.py#L28) はデータセットコレクションの基底テストクラスです。データセットコレクションが正しく登録されていること、またそのデータセットが TFDS に存在することを保証するための多数の単純なチェックを提供します。

設定する必要のある唯一のクラス属性は `DATASET_COLLECTION_CLASS` です。これは、テストするデータセットコレクションのクラスオブジェクトを指定します。

また、以下のクラス属性も設定可能です。

- `VERSION`: テストの実行に使用されるデータセットコレクションのバージョン（デフォルトは最新のバージョンです）。
- `DATASETS_TO_TEST`: TFDS での存在をテストするデータセットを含むリスト（デフォルトはコレクション内のすべてのデータセットです）。
- `CHECK_DATASETS_VERSION`: データセットコレクションのバージョン管理されたデータセットの存在をチェックするか、またはデフォルトバージョンをチェックするか（デフォルトは true です）。

データセットコレクションの最も単純で有効なテストは、以下のようになります。

```python
from tensorflow_datasets.testing.dataset_collection_builder_testing import DatasetCollectionTestBase
from . import my_collection

class TestMyCollection(DatasetCollectionTestBase):
  DATASET_COLLECTION_CLASS = my_collection.MyCollection
```

以下のコマンドを実行し、データセットコレクションをテストします。

```sh
python my_dataset_test.py
```

## フィードバック

データセット作成ワークフローは継続的な改善が進められていますが、問題を認識していなければ、改善することはできません。データセットコレクションの作成中にどのような問題またはエラーが発生しましたか？混乱したり、初めて使用したときに機能しなかった部分はありませんでしたか？

フィードバックを [GitHub](https://github.com/tensorflow/datasets/issues) にお送りください。
