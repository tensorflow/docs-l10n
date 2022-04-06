# カスタムデータセットを書く

新しいデータセットを作成するには（TFDS または独自のリポジトリ）、このガイドに従ってください。

必要なデータセットがすでに存在するかどうか、[データセットのリスト](catalog/overview.md)を確認してください。

## 要約

新しいデータセットを書く場合の最も簡単な方法は、[TFDS CLI](https://www.tensorflow.org/datasets/cli) を使うことです。

```sh
cd path/to/my/project/datasets/
tfds new my_dataset  # Create `my_dataset/my_dataset.py` template files
# [...] Manually modify `my_dataset/my_dataset.py` to implement your dataset.
cd my_dataset/
tfds build  # Download and prepare the dataset to `~/tensorflow_datasets/`
```

`tfds.load('my_dataset')` で新しいデータセットを使用すると、次のようになります。

- `tfds.load` は、`~/tensorflow_datasets/my_dataset/` に生成されたデータセット（`tfds build` などによって）を自動的に検出して読み取ります。
- または、次のように明示的に `import my.project.datasets.my_dataset` を実行して、データセットを登録することもできます。

```python
import my.project.datasets.my_dataset  # Register `my_dataset`

ds = tfds.load('my_dataset')  # `my_dataset` registered
```

## 概要

データセットは、あらゆる種類の形式であらゆる場所に分散されており、必ずしも機械学習パイプラインにフィードできる形式で保存されているわけではありません。TFDS に入ります。

TFDS は、それらのデータセットを標準形式に処理し（外部データ -&gt; シリアル化ファイル）、機械学習パイプラインとして読み取れるようにします（シリアル化ファイル -&gt; `tf.data.Dataset`）。シリアル化は一度しか行われません。以降のアクセスでは、前処理済みのファイルから直接読み取られます。

ほとんどの前処理は自動的に行われます。各データセットは `tfds.core.DatasetBuilder` のサブクラスを実装し、次の項目を指定します。

- データの送信元（URL）
- データセットはどのように見えるか（特徴量）
- データはどのように分割されるか（`TRAIN` と `TEST` など）
- データセット内の個別の Example

## データセットを書く

### デフォルトテンプレート: `tfds new`

[TFDS CLI](https://www.tensorflow.org/datasets/cli) を使用して、必要なテンプレート Python ファイルを生成します。

```sh
cd path/to/project/datasets/  # Or use `--dir=path/to/project/datasets/` below
tfds new my_dataset
```

このコマンドによって、次の構造を持つ新しい `my_dataset/` フォルダが生成されます。

```sh
my_dataset/
    __init__.py
    my_dataset.py # Dataset definition
    my_dataset_test.py # (optional) Test
    dummy_data/ # (optional) Fake data (used for testing)
    checksum.tsv # (optional) URL checksums (see `checksums` section).
```

ここで `TODO(my_dataset)` を探して、適宜、変更してください。

### データセットの Example

すべてのデータセットは、`tfds.core.GeneratorBasedBuilder`、つまり ほとんどのボイラープレートを処理する `tfds.core.DatasetBuilder` のサブクラスとして実装されます。これは次の項目をサポートします。

- 単一のマシンで生成できる小または中規模のデータセット（このチュートリアル）。
- 分散型の生成が必要な非常に大規模なデータセット（[Apache Beam](https://beam.apache.org/) を使用。[大規模なデータセットのガイド](https://www.tensorflow.org/datasets/beam_datasets#implementing_a_beam_dataset)をご覧ください）。

以下は、データセットクラスの最小限の Example です。

```python
class MyDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata (homepage, citation,...)."""
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(256, 256, 3)),
            'label': tfds.features.ClassLabel(
                names=['no', 'yes'],
                doc='Whether this is a picture of a cat'),
        }),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    extracted_path = dl_manager.download_and_extract('http://data.org/data.zip')
    # dl_manager returns pathlib-like objects with `path.read_text()`,
    # `path.iterdir()`,...
    return {
        'train': self._generate_examples(path=extracted_path / 'train_images'),
        'test': self._generate_examples(path=extracted_path / 'test_images'),
    }

  def _generate_examples(self, path) -> Iterator[Tuple[Key, Example]]:
    """Generator of examples for each split."""
    for img_path in path.glob('*.jpeg'):
      # Yields (key, example)
      yield img_path.name, {
          'image': img_path,
          'label': 'yes' if img_path.name.startswith('yes_') else 'no',
      }
```

では、上書きする 3 つの抽象メソッドを詳しく見てみましょう。

### `_info`: データセットのメタデータ

`_info` は `tfds.core.DatasetInfo` を返し、[データセットのメタデータ](https://www.tensorflow.org/datasets/overview#access_the_dataset_metadata) をそれに含めます。

```python
def _info(self):
  return tfds.core.DatasetInfo(
      builder=self,
      # Description and homepage used for documentation
      description="""
      Markdown description of the dataset. The text will be automatically
      stripped and dedent.
      """,
      homepage='https://dataset-homepage.org',
      features=tfds.features.FeaturesDict({
          'image_description': tfds.features.Text(),
          'image': tfds.features.Image(),
          # Here, 'label' can be 0-4.
          'label': tfds.features.ClassLabel(num_classes=5),
      }),
      # If there's a common `(input, target)` tuple from the features,
      # specify them here. They'll be used if as_supervised=True in
      # builder.as_dataset.
      supervised_keys=('image', 'label'),
      # Specify whether to disable shuffling on the examples. Set to False by default.
      disable_shuffling=False,
      # Bibtex citation for the dataset
      citation=r"""
      @article{my-awesome-dataset-2020,
               author = {Smith, John},}
      """,
  )
```

ほとんどのフィールドには説明はいりませんが、以下にいくつか補足します。

- `features`: これは、データセットの構造や形状などを指定します。複雑なデータタイプ（音声、動画、ネストされたシーケンスなど）をサポートしています。詳細は、[利用可能な特徴量](https://www.tensorflow.org/datasets/api_docs/python/tfds/features#classes)または[特徴量コネクタガイド](https://www.tensorflow.org/datasets/features)をご覧ください。
- `disable_shuffling`: 「[データセットの順序を維持する](#maintain-dataset-order)」をご覧ください。
- `citation`: 次のようにして `BibText` の引用を見つけます。
    - データセットのウェブサイトで引用方法を検索します（BibTex 形式で使用します）。
    - [arXiv](https://arxiv.org/) の論文: 論文を見つけ、右側の `BibText` リンクをクリックします。
    - [Google Scholar](https://scholar.google.com) で論文を検索し、題名の下にある二重引用符をクリックし、ポップアップ表示に示される `BibTeX` をクリックします。
    - 関連する論文がない場合（ウェブサイトのみなど）、[BibTeX Online Editor](https://truben.no/latex/bibtex/) を使用して、カスタム BibTeX エントリを作成します（`Online` エントリタイプ）。

#### データセットの順序を維持する

データセットのレコードは保存されるとデータセット内のクラスの分散を均一にするために、シャッフルされます。これは、同じクラスに属するレコードは連続していることがほとんであるためです。データセットを `_generate_examples` が提供する生成されたキーで並べ替えるように指定するには、`disable_shuffling` フィールドを `True` に設定する必要があります。デフォルトでは、`False` に設定されています。

```python
def _info(self):
  return tfds.core.DatasetInfo(
    # [...]
    disable_shuffling=True,
    # [...]
  )
```

シャッフルを無効にすると、シャードが並行して読み取れなくなるため、パフォーマンスに悪影響が及ぶことに注意してください。

### `_split_generators`: データのダウンロードと分割

#### ソースデータをダウンロードして抽出する

ほとんどのデータセットはウェブからダウンロードする必要があります。これは、`_split_generators` の `tfds.download.DownloadManager` 入力引数を使用して行います。`dl_manager` には、次のメソッドがあります。

- `download`: `http(s)://`、`ftp(s)://` がサポートされています。
- `extract`: 現在サポートされているのは `.zip`、`.gz`、および `.tar` ファイルです。
- `download_and_extract`: `dl_manager.extract(dl_manager.download(urls))` と同じです。

これらのすべてのメソッドは `tfds.core.Path`（[`epath.Path`](https://github.com/google/etils) のエイリアス）を戻し、これらは [pathlib.Path のような](https://docs.python.org/3/library/pathlib.html)オブジェクトです。

これらのメソッドは次のように、任意のネストされた構造（`list`、`dict` など）をサポートしています。

```python
extracted_paths = dl_manager.download_and_extract({
    'foo': 'https://example.com/foo.zip',
    'bar': 'https://example.com/bar.zip',
})
# This returns:
assert extracted_paths == {
    'foo': Path('/path/to/extracted_foo/'),
    'bar': Path('/path/extracted_bar/'),
}
```

#### 手動によるダウンロードと抽出

一部のデータは自動的にダウンロードされません（ログインが必要です）。この場合、ユーザーはソースデータを手動でダウンロードして、`manual_dir/` に配置する必要があります（デフォルトは `~/tensorflow_datasets/downloads/manual/` です）。

すると、`dl_manager.manual_dir` を介してファイルにアクセスできるようになります。

```python
class MyDataset(tfds.core.GeneratorBasedBuilder):

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Register into https://example.org/login to get the data. Place the `data.zip`
  file in the `manual_dir/`.
  """

  def _split_generators(self, dl_manager):
    # data_path is a pathlib-like `Path('<manual_dir>/data.zip')`
    archive_path = dl_manager.manual_dir / 'data.zip'
    # Extract the manually downloaded `data.zip`
    extracted_path = dl_manager.extract(archive_path)
    ...
```

`manual_dir` の場所は、`tfds build --manual_dir=` または `tfds.download.DownloadConfig` を使ってカスタマイズ可能です。

#### アーカイブを直接読み取る

`dl_manager.iter_archive` は、アーカイブを抽出せずに順に読み取ります。このため、ストレージの領域を節約し、一部のファイルシステムのパフォーマンスを改善できます。

```python
for filename, fobj in dl_manager.iter_archive('path/to/archive.zip'):
  ...
```

`fobj` には、`with open('rb') as fobj:` と同じメソッドがあります（`fobj.read()`  など）。

#### データセットの分割を指定する

データセットに事前定義済みの Split が含まれる場合（たとえば `MNIST` には `train` と `test` の Split が含まれます）、それらを維持してください。含まれない場合は、単一の `tfds.Split.TRAIN` Split のみを指定します。ユーザーは[サブスプリット API](https://www.tensorflow.org/datasets/splits) で独自のサブスプリットを動的に作成できます（`split='train[80%:]'` など）。

```python
def _split_generators(self, dl_manager):
  # Download source data
  extracted_path = dl_manager.download_and_extract(...)

  # Specify the splits
  return {
      'train': self._generate_examples(
          images_path=extracted_path / 'train_imgs',
          label_path=extracted_path / 'train_labels.csv',
      ),
      'test': self._generate_examples(
          images_path=extracted_path / 'test_imgs',
          label_path=extracted_path / 'test_labels.csv',
      ),
  }
```

### `_generate_examples`: Example ジェネレータ

`_generate_examples` はソースデータから各 Split の Example を生成します。

このメソッドは通常、ソースデータセットのアーティファクト（CSV ファイルなど）を読み取り、`(key, feature_dict)` タプルを生成します。

- `key`: Example の識別子。`hash(key)` を使って Example を決定的にシャッフルするか、シャッフルが無効である場合に key で並べ替えるために使用されます（「[データセットの順序を維持する](#maintain-dataset-order)」をご覧ください）。次のようである必要があります。
    - **一意であること**: 2 つの Example が同じ key を使用している場合、例がが発生します。
    - **決定的であること**: `download_dir`、 `os.path.listdir` の順などに依存してはいけません。データを 2 回生成すると、同じ kye が生成されてしまいます。
    - **比較可能であること**: シャッフルが無効である場合、key はデータセットの並べ替えに使用されます。
- `feature_dict`: Example の値を含む `dict` です。
    - 構造は `tfds.core.DatasetInfo` で定義されている `features=` 構造に一致する必要があります。
    - 複雑なデータタイプ（画像、動画、音声など）は自動的に暗号化されます。
    - 各特徴量は通常、複数の入力タイプを受け入れます（たとえば、動画は `/path/to/vid.mp4`、`np.array(shape=(l, h, w, c))`、`List[paths]`、`List[np.array(shape=(h, w, c)]`、`List[img_bytes]` などを受け入れます）。
    - 詳細は、[特徴量コネクタのガイド](https://www.tensorflow.org/datasets/features)をご覧ください。

```python
def _generate_examples(self, images_path, label_path):
  # Read the input data out of the source files
  with label_path.open() as f:
    for row in csv.DictReader(f):
      image_id = row['image_id']
      # And yield (key, feature_dict)
      yield image_id, {
          'image_description': row['description'],
          'image': images_path / f'{image_id}.jpeg',
          'label': row['label'],
      }
```

#### ファイルのアクセスと `tf.io.gfile`

クラウドストレージシステムをサポートするために、Python のビルトイン I/O 演算子の使用は避けれください。

代わりに、`dl_manager` は Google Cloud Storage に対応する [pathlib のような](https://docs.python.org/3/library/pathlib.html)オブジェクトを直接返します。

```python
path = dl_manager.download_and_extract('http://some-website/my_data.zip')

json_path = path / 'data/file.json'

json.loads(json_path.read_text())
```

または、ファイル操作を行う場合は、ビルトインの代わりに `tf.io.gfile` API を使用してください。

- `open` -&gt; `tf.io.gfile.GFile`
- `os.rename` -&gt; `tf.io.gfile.rename`
- など

Pathlib は `tf.io.gfile` よりも優先される必要があります（[rational](https://www.tensorflow.org/datasets/common_gotchas#prefer_to_use_pathlib_api) の項目をご覧ください）。

#### 追加の依存関係

一部のデータセットでは、生成中のみに追加の Python 依存関係が必要となります。たとえば、SVHN データセットでは、データの読み込みに `scipy` を使用しています。

データセットを TFDS リポジトリに追加する場合は、`tfds.core.lazy_imports` を使用して `tensorflow-datasets` パッケージを小さく維持してください。ユーザーは必要な場合にのみ追加の依存関係をインストールします。

`lazy_imports` を使用するには、次を実行します。

- データセットのエントリを [`setup.py`](https://github.com/tensorflow/datasets/tree/master/setup.py) の`DATASET_EXTRAS` に追加します。これにより、たとえば `pip install 'tensorflow-datasets[svhn]'` を使用して、追加の依存関係をインストールできるようになります。
- インポートのエントリを [`LazyImporter`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/lazy_imports_lib.py) と [`LazyImportsTest`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/lazy_imports_lib_test.py) に追加します。
- `tfds.core.lazy_imports` を使用して、`DatasetBuilder` の依存関係（`tfds.core.lazy_imports.scipy` など）にアクセスします。

#### 破損データ

一部のデータセットは完全にはクリーンでなく、破損データ（画像は JPEG ファイルですが無効な JPEG であるなど）が含まれているものもあります。これらのサンプルをスキップし、いくつのサンプルが削除され、なぜ削除されたのかのメモを残す必要があります。

### データセットの構成/バリアント（tfds.core.BuilderConfig）

一部のデータセットには、複数のバリアントや、データの処理方法とディスクへの書き込み方法のオプションがあります。たとえば、[cycle_gan](https://www.tensorflow.org/datasets/catalog/cycle_gan) の構成は、オブジェクトペア（`cycle_gan/horse2zebra`、`cycle_gan/monet2photo` など）ごとに 1 つです。

これは、`tfds.core.BuilderConfig` を使って以下のように行います。

1. 構成オブジェクトを `tfds.core.BuilderConfig` のサブクラスとして、`MyDatasetConfig` というように定義します。

    ```python
    @dataclasses.dataclass
    class MyDatasetConfig(tfds.core.BuilderConfig):
      img_size: Tuple[int, int] = (0, 0)
    ```

    注意: https://bugs.python.org/issue33129 のバグにより、デフォルト値が必要です。

2. データセットが公開する `MyDatasetConfig` をリストする `BUILDER_CONFIGS = []` クラスメンバーを `MyDataset` に定義します。

    ```python
    class MyDataset(tfds.core.GeneratorBasedBuilder):
      VERSION = tfds.core.Version('1.0.0')
      # pytype: disable=wrong-keyword-args
      BUILDER_CONFIGS = [
          # `name` (and optionally `description`) are required for each config
          MyDatasetConfig(name='small', description='Small ...', img_size=(8, 8)),
          MyDatasetConfig(name='big', description='Big ...', img_size=(32, 32)),
      ]
      # pytype: enable=wrong-keyword-args
    ```

    注意: `# pytype: disable=wrong-keyword-args` は必要です。データクラスの継承に関する [Pytype バグ](https://github.com/google/pytype/issues/628)があります。

3. `self.builder_config` を `MyDataset` に使用して、データ生成を構成します（`shape=self.builder_config.img_size` など）。これには、`_info()` にさまざまな値の設定や、ダウンロードデータへのアクセスの変更が含まれる場合があります。

注意:

- 構成にはそれぞれに一意の名前があります。構成の完全修飾名は `dataset_name/config_name` です（例: `coco/2017`）。
- 指定されていない場合は、`BUILDER_CONFIGS` の最初の構成が使用されます（たとえば、`tfds.load('c4')` はデフォルトで `c4/en` となります）。

[`anli`](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/anli.py#L69) で、`BuilderConfig` を使用するデータセットの例をご覧ください。

### バージョン

バージョンには、2 つの意味があります。

- 「外部」の元のデータバージョン: COCO v2019、v2017、など。
- 「内部」の TFDS コードバージョン: `tfds.features.FeaturesDict` の特徴量の名前の変更、`_generate_examples` のバグの修正など。

データセットを更新するには、次のように行います。

- 「外部」データの更新: 不空数のユーザーが特定の年/バージョンに同時にアクセスすることがあります。これは、バージョン当たり 1 つの `tfds.core.BuilderConfig`（`coco/2017`、`coco/2019` など）またはバージョン当たり 1 つのクラス（`Voc2007`、`Voc2012` など）を使って行われます。
- 「内部」コードの更新: ユーザーは最も最近のバージョンのみをダウンロードします。コードが更新されると、`VERSION` クラス属性が増加（`1.0.0` から `VERSION = tfds.core.Version('2.0.0')` など）します。これは[セマンティックバージョン管理](https://www.tensorflow.org/datasets/datasets_versioning#semantic)に従います。

### import を追加して登録する

プロジェクトの `__init__` に忘れずにデータセットモジュールをインポートして、`tfds.load`、`tfds.builder` に自動的に登録されるようにします。

```python
import my_project.datasets.my_dataset  # Register MyDataset

ds = tfds.load('my_dataset')  # MyDataset available
```

たとえば、`tensorflow/datasets` に貢献する場合は、モジュールのインポートをサブディレクトリの `__init__.py`（[`image/__init__.py`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/image/__init__.py) など）に追加します。

### 共通する実装の落とし穴を確認する

[共通する実装の落とし穴](https://www.tensorflow.org/datasets/common_gotchas)をご覧ください。

## データセットをテストする

### ダウンロードと準備: `tfds build`

データセットを生成するには、`my_dataset/` ディレクトリから `tfds build` を実行します。

```sh
cd path/to/datasets/my_dataset/
tfds build --register_checksums
```

開発に使用できる便利なフラグには、次のようなものがあります。

- `--pdb`: 例外が発生すると、デバッグモードに入ります。
- `--overwrite`: データセットがすでに生成されている場合、既存のファイルを削除します。
- `--max_examples_per_split`: データセットすべてではなく、最初の X 個の Example （デフォルトは 1 個）のみを生成します。
- `--register_checksums`: ダウンロード URL のチェックサムを記録します。開発中のみに使用します。

全フラグのリストは、[CLI ドキュメント](https://www.tensorflow.org/datasets/cli#tfds_build_download_and_prepare_a_dataset)をご覧ください。

### チェックサム

決定性の確保や文書化などのために、データセットのチェックサムを記録することをお勧めします。これは、`--register_checksums` を使ってデータセットを生成して行います（前のセクションをご覧ください）。

PyPI を介してデータセットをリリースする場合、忘れずに （`setup.py` の `package_data` などの）`checksums.tsv` ファイルをエクスポートしてください。

### データセットのユニットテストを行う

`tfds.testing.DatasetBuilderTestCase` は、データセットを完全に演習するためのベースの `TestCase` です。テストデータとして、ソースデータセットの構造を模倣した「ダミーデータ」が使用されます。

- テストデータは、`my_dataset/dummy_data/` ディレクトリに配置し、ソースデータセットアーチファクトをダウンロードおよび抽出されたとおりに模倣しています。作成は手動のほか、スクリプト（[サンプルスクリプト](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/image/bccd/dummy_data_generation.py)）を使用して自動的に作成することもできます。
- データセット分割が重なっている場合、テストは失敗してしまうため、テストデータ分割に異なるデータが使用されるようにしてください。
- **テストデータには、著作権で保護された資料を含めてはいけません**。不明な場合は、元のデータセットの資料を使ってデータを作成しないでください。

```python
import tensorflow_datasets as tfds
from . import my_dataset


class MyDatasetTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for my_dataset dataset."""
  DATASET_CLASS = my_dataset.MyDataset
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  DL_EXTRACT_RESULT = {
      'name1': 'path/to/file1',  # Relative to dummy_data/my_dataset dir.
      'name2': 'file2',
  }


if __name__ == '__main__':
  tfds.testing.test_main()
```

次のコマンドを使用して、データセットをテストします。

```sh
python my_dataset_test.py
```

## フィードバックをお送りください

データセットの作成ワークフローの改善に絶えず努めていますが、問題がわからなければ、改善することはできません。データセットの作成中にどのような問題やエラーが発生しましたか？わかりにくい部分やボイラープレートがありますか？または最初、機能していませんでしたか？[GitHub にフィードバック](https://github.com/tensorflow/datasets/issues)をお送りください。
