# データセットのバージョン管理

## 定義

バージョン管理には、異なる意味があります。

- TFDS API バージョン (pip バージョン): `tfds.__version__`
- TFDS のバージョンから独立した別のバージョン管理方式 ([Voc2007](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)、Voc2012 など)。TFDS では、各パブリックデータセットバージョンを独立したデータセットとして実装する必要があります。
    - [builder configs](https://www.tensorflow.org/datasets/add_dataset#dataset_configurationvariants_tfdscorebuilderconfig) を介して実装する: `voc/2007`、`voc/2012` など
    - 2つの独立したデータセットとして実装する: `wmt13_translate`、`wmt14_translate` など
- TFDS のデータセット生成コードバージョン(`my_dataset:1.0.0`): 例えば、 `voc/2007` の TFDS 実装にバグが見つかった場合、`voc.py` 生成コードが更新されます。 (`voc/2007:1.0.0` -&gt; `voc/2007:2.0.0`)。

以下では、最後の定義 (TFDS リポジトリーのデータセットコードバージョン) のみを見ていきます。

## サポートされているバージョン

一般的なルール:

- 最後の最新のバージョンのみを生成できます。
- 以前に生成されたすべてのデータセットを読み取ることができます（注: これには、TFDS 4 以降で生成されたデータセットが必要です）。

```python
builder = tfds.builder('my_dataset')
builder.info.version  # Current version is: '2.0.0'

# download and load the last available version (2.0.0)
ds = tfds.load('my_dataset')

# Explicitly load a previous version (only works if
# `~/tensorflow_datasets/my_dataset/1.0.0/` already exists)
ds = tfds.load('my_dataset:1.0.0')
```

## セマンティック

TFDS で定義されているすべての`DatasetBuilder`にはバージョンがあります。例を示します。

```python
class MNIST(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('2.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release',
      '2.0.0': 'Update dead download url',
  }
```

バージョンは [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html): `MAJOR.MINOR.PATCH`に従います。バージョンの目的は、再現性を保証できるようにすることにあります。つまり、特定のデータセットを固定のバージョンで読み込むと、同じデータが生成されます。より具体的に示します。

- `PATCH`のバージョンをインクリメントする場合、クライアントが読み込むデータは同じですが、ディスク上でデータのシリアライズが異なっていたり、メタデータが変更されている可能性があります。与えられたスライスに対して、スライス API は同じレコードのセットを返します。
- `MINOR`のバージョンをインクリメントする場合、クライアントが読み込む既存のデータは同じですが、追加データ（各レコードの特徴）があります。任意のスライスに対して、スライス API は同じレコードのセットを返します。
- `MAJOR`のバージョンをインクリメントする場合、既存のデータは変更されているか、および/または任意のスライスに対してスライス API は同じレコードのセットを返すとは限りません。

TFDS ライブラリのコード変更が行われ、そのコード変更がデータセットのシリアライズやクライアントの読み取り方に影響を与える場合は、対応するビルダーバージョンを上記のガイドラインに沿ってインクリメントします。

上記のセマンティックはベストエフォートですが、バージョンのインクリメント前にはデータセットに影響を及ぼす、気づかないバグが存在する可能性があるので注意してください。そのようなバグは最終的には修正されますが、バージョニングに大きく依存する場合は、（`HEAD`とは対照的に）リリースされているバージョンの TFDS の使用をお勧めします。

また、データセットによっては、TFDS のバージョンから独立した別のバージョン管理方式があることにも注意してください。例えば、Open Images データセットにはいくつかのバージョンがあり、TFDS では、対応するビルダーは`open_images_v4`、`open_images_v5`などです。

## 特定のバージョンを読み込む

データセットや`DatasetBuilder`を読み込む際に、使用するバージョンを指定できます。例を示します。

```python
tfds.load('imagenet2012:2.0.1')
tfds.builder('imagenet2012:2.0.1')

tfds.load('imagenet2012:2.0.0')  # Error: unsupported version.

# Resolves to 3.0.0 for now, but would resolve to 3.1.1 if when added.
tfds.load('imagenet2012:3.*.*')
```

パブリケーションに TFDS を使用する場合には、以下をお勧めします。

- バージョンの`MAJOR`コンポーネントのみを修正する。
- **結果の中に、使用したデータセットのバージョン番号を公表する。**

そうすることによって、将来の自分自身、読者、レビュアーが結果を再現しやすくなります。

## BUILDER_CONFIGS とバージョン

データセットによっては、複数の`BUILDER_CONFIGS`を定義しているものがあります。その場合、`version`と`supported_versions`は、それらの構成オブジェクト上で定義します。それ以外は、セマンティックと使用方法は同じです。例を示します。

```python
class OpenImagesV4(tfds.core.GeneratorBasedBuilder):

  BUILDER_CONFIGS = [
      OpenImagesV4Config(
          name='original',
          version=tfds.core.Version('0.2.0'),
          supported_versions=[
            tfds.core.Version('1.0.0', "Major change in data"),
          ],
          description='Images at their original resolution and quality.'),
      ...
  ]

tfds.load('open_images_v4/original:1.*.*')
```

## 実験バージョン

注意: 以下は悪い習慣であり、エラーが発生しやすくなるため、避けることをお勧めします。

2 つのバージョン (1つ のデフォルトバージョンと 1 つの実験バージョン) を同時に生成できるようにすることができます。以下に例を示します。

```python
class MNIST(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version("1.0.0")  # Default version
  SUPPORTED_VERSIONS = [
      tfds.core.Version("2.0.0"),  # Experimental version
  ]


# Download and load default version 1.0.0
builder = tfds.builder('mnist')

#  Download and load experimental version 2.0.0
builder = tfds.builder('mnist', version='experimental_latest')
```

コードでは、2つのバージョンをサポートしていることを確認する必要があります。

```python
class MNIST(tfds.core.GeneratorBasedBuilder):

  ...

  def _generate_examples(self, path):
    if self.info.version >= '2.0.0':
      ...
    else:
      ...
```
