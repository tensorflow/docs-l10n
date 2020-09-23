# データセットのバージョン管理

- [セマンティック](#semantic)
- [サポートされているバージョン](#supported-versions)
- [特定のバージョンを読み込む](#loading-a-specific-version)
- [実験](#experiments)
- [BUILDER_CONFIGS とバージョン](#builder-configs-and-versions)

## セマンティック

TFDS で定義されているすべての`DatasetBuilder`にはバージョンがあります。例を示します。

```py
class MNIST(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version("1.0.0")
```

バージョンは [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html): `MAJOR.MINOR.PATCH`に従います。バージョンの目的は、再現性を保証できるようにすることにあります。つまり、特定のデータセットを固定のバージョンで読み込むと、同じデータが生成されます。より具体的に示します。

- `PATCH`のバージョンをインクリメントする場合、クライアントが読み込むデータは同じですが、ディスク上でデータのシリアライズが異なっていたり、メタデータが変更されている可能性があります。与えられたスライスに対して、スライス API は同じレコードのセットを返します。
- `MINOR`のバージョンをインクリメントする場合、クライアントが読み込む既存のデータは同じですが、追加データ（各レコードの特徴）があります。任意のスライスに対して、スライス API は同じレコードのセットを返します。
- `MAJOR`のバージョンをインクリメントする場合、既存のデータは変更されているか、および/または任意のスライスに対してスライス API は同じレコードのセットを返すとは限りません。

TFDS ライブラリのコード変更が行われ、そのコード変更がデータセットのシリアライズやクライアントの読み取り方に影響を与える場合は、対応するビルダーバージョンを上記のガイドラインに沿ってインクリメントします。

上記のセマンティックはベストエフォートですが、バージョンのインクリメント前にはデータセットに影響を及ぼす、気づかないバグが存在する可能性があるので注意してください。そのようなバグは最終的には修正されますが、バージョニングに大きく依存する場合は、（`HEAD`とは対照的に）リリースされているバージョンの TFDS の使用をお勧めします。

また、データセットによっては、TFDS のバージョンから独立した別のバージョン管理方式があることにも注意してください。例えば、Open Images データセットにはいくつかのバージョンがあり、TFDS では、対応するビルダーは`open_images_v4`、`open_images_v5`などです。

## サポートされているバージョン

`DatasetBuilder`は、複数のバージョンをサポートできますが、そのバージョンが正規バージョンより高い場合もあれば低い場合もあります。例を示します。

```py
class Imagenet2012(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('2.0.1', 'Encoding fix. No changes from user POV')
  SUPPORTED_VERSIONS = [
      tfds.core.Version('3.0.0', 'S3: tensorflow.org/datasets/splits'),
      tfds.core.Version('1.0.0'),
      tfds.core.Version('0.0.9', tfds_version_to_prepare="v1.0.0"),
  ]
```

古いバージョンのサポートを継続するかどうかについての選択は、主にデータセットとバージョンの人気度に基づき、ケースバイケースで行われます。最終的にはデータセットごとにごく少数のバージョン、理想的には単一のバージョンのみのサポートを目指しています。上記の例では、バージョン`2.0.0`は読者の観点から見た場合`2.0.1`と同じなので、既にサポートされていないことが分かります。

サポートされているバージョンの中で正規のバージョン番号よりも高い番号を持つバージョンは、実験的なものとみなしており、破損している場合があります。しかしそれらは最終的には正規化されます。

バージョンは`tfds_version_to_prepare`を指定することができます。これは、このデータセットのバージョンが既に古いバージョンのコードで準備されている場合、現在のバージョンの TFDS コードと使用はできますが、準備はできないことを意味します。`tfds_version_to_prepare`の値には、このバージョンでデータセットをダウンロードして準備するために使用が可能な、TFDS の最新の既知のバージョンを指定します。

## 特定のバージョンを読み込む

データセットや`DatasetBuilder`を読み込む際に、使用するバージョンを指定できます。例を示します。

```py
tfds.load('imagenet2012:2.0.1')
tfds.builder('imagenet2012:2.0.1')

tfds.load('imagenet2012:2.0.0')  # Error: unsupported version.

# Resolves to 3.0.0 for now, but would resolve to 3.1.1 if when added.
tfds.load('imagenet2012:3.*.*')
```

パブリケーションに TFDS を使用する場合には、以下をお勧めします。

- **バージョンの`MAJOR`コンポーネントのみを修正する。**
- **結果の中に、使用したデータセットのバージョン番号を公表する。**

そうすることによって、将来の自分自身、読者、レビュアーが結果を再現しやすくなります。

## 実験

多くのデータセットビルダーに影響を与える TFDS の変更を段階的に展開するために、実験という概念を導入しました。実験は、最初の導入ではデフォルトで無効になっていますが、特定のデータセットのバージョンを有効にすることができます。これは通常、最初に「将来の」（まだ正規化されていない）バージョンで行います。例を示します。

```py
class MNIST(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version("1.0.0")
  SUPPORTED_VERSIONS = [
      tfds.core.Version("2.0.0", "EXP1: Opt-in for experiment 1",
                        experiments={tfds.core.Experiment.EXP1: True}),
  ]
```

実験が期待通りに動作することを確認した後、それをすべての、または大部分のデータセットに拡張し、その時点でデフォルトで有効化することができます。上記の定義は以下のようになります。

```py
class MNIST(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version("1.0.0",
                              experiments={tfds.core.Experiment.EXP1: False})
  SUPPORTED_VERSIONS = [
      tfds.core.Version("2.0.0", "EXP1: Opt-in for experiment 1"),
  ]
```

すべてのデータセットバージョンに渡って実験を使用すると（`{experiment: False}`を指定したデータセットのバージョンが残っていない場合は）、その実験を削除することができます。

実験とその説明は`core/utils/version.py`で定義されています。

## BUILDER_CONFIGS とバージョン

データセットによっては、複数の`BUILDER_CONFIGS`を定義しているものがあります。その場合、`version`と`supported_versions`は、それらの構成オブジェクト上で定義します。それ以外は、セマンティックと使用方法は同じです。例を示します。

```py
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
