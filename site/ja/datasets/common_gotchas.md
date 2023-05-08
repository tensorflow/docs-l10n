# 共通する実装の落とし穴

このページでは、新しいデータセットを実装する際に共通する実装の落とし穴を説明します。

## レガシーの `SplitGenerator` は回避すること

古い `tfds.core.SplitGenerator` API は使用廃止となっています。

```python
def _split_generator(...):
  return [
      tfds.core.SplitGenerator(name='train', gen_kwargs={'path': train_path}),
      tfds.core.SplitGenerator(name='test', gen_kwargs={'path': test_path}),
  ]
```

上記は、以下のように置き換えることをお勧めします。

```python
def _split_generator(...):
  return {
      'train': self._generate_examples(path=train_path),
      'test': self._generate_examples(path=test_path),
  }
```

**理由**: 新しい API は以前ほど詳細でなくなり、より明示的になっています。古い API は今後のバージョンで削除される予定です。

## 新しいデータセットはフォルダ内で自己完結型であること

`tensorflow_datasets/` リポジトリ内にデータセットを追加する際は、フォルダとしてのデータセットの構造に必ず従ってください（チェックサム、ダミーデータ、実装コードがフォルダ内にすべて収まっていること）。

- 古いデータベース（悪い例）: `<category>/<ds_name>.py`
- 新しいデータベース（良い例）: `<category>/<ds_name>/<ds_name>.py`

[TFDS CLI](https://www.tensorflow.org/datasets/cli#tfds_new_implementing_a_new_dataset)（`tfds new`、または Google 開発者の場合は `gtfds new`）を使用して、テンプレートを生成します。

**理由**: 古い構造には、チェックサム、フェイクデータの絶対パスが必要であり、データセットが複数の場所に分散していました。そのため、TFDS リポジトリ外部でデータセットを実装するのがより困難になっていました。現在では、一貫性を得るために、新しい構造をあらゆる箇所に使用することが推奨されています。

## 説明リストはマークダウンでフォーマットされていること

`DatasetInfo.description` `str` はマークダウンとしてフォーマットされています。マークダウンリストには、最初の項目の前に空の行が必要です。

```python
_DESCRIPTION = """
Some text.
                      # << Empty line here !!!
1. Item 1
2. Item 1
3. Item 1
                      # << Empty line here !!!
Some other text.
"""
```

**理由**: 不正にフォーマットされた説明によって、カタログドキュメントに視覚的なアーティファクトが生成されます。空の行が挿入されていない場合、上記のテキストは以下のようにレンダリングされてしまいます。

Some text. 1. Item 1 2. Item 1 3. Item 1 Some other text

## ClassLabel 名忘れ

`tfds.features.ClassLabel` を使用する際に、人間が読み取れるラベル `str` に `names=` または `names_file=`（`num_classes=10` ではなく）を指定するようにしてください。

```python
features = {
    'label': tfds.features.ClassLabel(names=['dog', 'cat', ...]),
}
```

**理由**: 人間が読み取れるラベルは、多くの場所で使用されています。

- `_generate_examples` で直接 `str` を生成できるようにする: `yield {'label': 'dog'}`
- `info.features['label'].names` のようにユーザーに公開されている（変換メソッド `.str2int('dog')` なども利用可能）
- [可視化ユーティリティ](https://www.tensorflow.org/datasets/overview#tfdsas_dataframe) の `tfds.show_examples`、`tfds.as_dataframe` で使用されている

## 画像の形状忘れ

`tfds.features.Image` や `tfds.features.Video` を使用する際に、形状が静的である場合は、それを明示的に指定する必要があります。

```python
features = {
    'image': tfds.features.Image(shape=(256, 256, 3)),
}
```

**理由**: 静的形状推論（`ds.element_spec['image'].shape` など）が可能になるためです。これは、バッチ処理に必要となります（形状が不明な画像をバッチ処理するには、先にサイズを変更する必要があります）。

## `tfds.features.Tensor` の代わりにより具体的な型を推奨

可能な場合は、汎用の `tfds.features.Tensor` ではなく、`tfds.features.ClassLabel` や `tfds.features.BBoxFeatures` などの具体的な型を使用するようにしてください。

**理由**: 意味的により正しいからというだけでなく、特徴量に具体性を指定することで、ユーザーに追加メタデータを提供し、ツールが検出できるようになるためです。

## グローバル空間での遅延インポート

グローバル空間では、遅延インポートを呼び出してはいけません。たとえば、以下は誤りです。

```python
tfds.lazy_imports.apache_beam # << Error: Import beam in the global scope

def f() -> beam.Map:
  ...
```

**理由**: グローバル範囲で遅延インポートを使用すると、すべての tfds ユーザーのモジュールがインポートされてしまい、遅延インポートの意味が無くなってしまいます。

## train/test Split の動的計算

データセットに正式な Split が指定されていない場合、TFDS にも指定されていてはいけません。以下のようにしないでください。

```python
_TRAIN_TEST_RATIO = 0.7

def _split_generator():
  ids = list(range(num_examples))
  np.random.RandomState(seed).shuffle(ids)

  # Split train/test
  train_ids = ids[_TRAIN_TEST_RATIO * num_examples:]
  test_ids = ids[:_TRAIN_TEST_RATIO * num_examples]
  return {
      'train': self._generate_examples(train_ids),
      'test': self._generate_examples(test_ids),
  }
```

**理由**: TFDS は、できるだけ元のデータに近いデータセットを提供しようとします。ユーザーが必要とする subsplit を動的に作成できるようにするには、代わりに [sub-split API](https://www.tensorflow.org/datasets/splits) を使用してください。

```python
ds_train, ds_test = tfds.load(..., split=['train[:80%]', 'train[80%:]'])
```

## Python スタイルガイド

### pathlib API の使用を推奨

`tf.io.gfile` API の代わりに、[pathlib API](https://docs.python.org/3/library/pathlib.html) の使用が推奨されます。すべての `dl_manager` メソッドは GCS や S3 などと互換性のある pathlib のようなオブジェクトを返します。

```python
path = dl_manager.download_and_extract('http://some-website/my_data.zip')

json_path = path / 'data/file.json'

json.loads(json_path.read_text())
```

**理由**: pathlib API は、ボイラープレートを取り除く最新のオブジェクト指向ファイル API です。`.read_text()` / `.read_bytes()` を使用すると、ファイルが正しく閉じられたことも保証されます。

### メソッドが `self` を使用していない場合、そのメソッドは関数である

クラスメソッドに `self` が使用されていない場合、そのメソッドは単なる（クラス外部で定義された）関数です。

**理由**: 関数に副作用や隠れた入出力がないことがユーザーに明白になります。

```python
x = f(y)  # Clear inputs/outputs

x = self.f(y)  # Does f depend on additional hidden variables ? Is it stateful ?
```

## Python での遅延インポート

TensorFlow などの大規模なモジュールは遅延インポートします。遅延インポートは、モジュールが初めて使用されるまで実際のインポートを遅らせる手法です。そのため、その大規模なモジュールを必要としないユーザーがそれをインポートすることはありません。

```python
from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf
# After this statement, TensorFlow is not imported yet

...

features = tfds.features.Image(dtype=tf.uint8)
# After using it (`tf.uint8`), TensorFlow is now imported
```

内部では、[`LazyModule` class](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/utils/lazy_imports_utils.py) がファクトリとして機能し、属性へのアクセスがあったときにのみ（`__getattr__`）モジュールが実際にインポートされるようになっています。

これをコンテキストマネージャーと利用することもできます。

```python
from tensorflow_datasets.core.utils.lazy_imports_utils import lazy_imports

with lazy_imports(error_callback=..., success_callback=...):
  import some_big_module
```
