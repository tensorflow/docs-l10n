# 形式固有のデータセットビルダー

[目次]

このガイドには、TFDS で現在提供されている形式固有のすべてのデータセットビルダーを記載しています。

形式固有のデータセットビルダーは、特定のデータ形式について、ほとんどのデータ処理を請け負う [`tfds.core.GeneratorBasedBuilder`](https://www.tensorflow.org/datasets/api_docs/python/tfds/core/GeneratorBasedBuilder) のサブクラスです。

## `tf.data.Dataset` に基づくデータセット

`tf.data.Dataset`（[リファレンス](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)）形式でないデータセットから TFDS データセットを作成するには、`tfds.dataset_builders.TfDataBuilder` を使用できます。（[API ドキュメントを参照](https://www.tensorflow.org/datasets/api_docs/python/tfds/dataset_builders/TfDataBuilder)）

このクラスには、一般的な用途として以下の 2 つが想定されています。

- ノートブックのような環境で、実験的データセットを作成する
- コードでデータセットビルダーを定義する

### ノートブックから新しいデータセットを作成する

ノートブックで作業をしているとします。`tf.data.Dataset` としてデータが読み込まれており、様々な変換（マップ、フィルタなど）が適用されている状態です。このデータを保存し、チームメイトと共有したり、別のノートブックに読み込みたいと考えています。この場合、新しいデータセットビルダークラスを定義する代わりに、`tfds.dataset_builders.TfDataBuilder` をインスタンス化して、`download_and_prepare` を呼び出してデータセットを TFDS データセットとして保存することができます。

これは TFDS データセットであるため、バージョン管理、構成の使用、異なる分割の設定、後で使いやすくするための文書化を行えます。つまり、TFDS にデータセットの特徴量を知らせることも可能と言うことになります。

以下は、その使用方法を示すダミーの例です。

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

`download_and_prepare` メソッドは、入力 `tf.data.Dataset` をイテレートして対応する TFDS データセットを `/my/folder/my_dataset/single_number/1.0.0` に保存します。これには、train と test のいずれの分割も含まれます。

`config` 引数はオプションであり、同じデータセットで異なる構成を保存する場合に便利です。

`data_dir` 引数は、生成された TFDS データセットを異なるフォルダに保存するために使用できます。たとえば、他の人と（まだ）共有したくない場合は、独自のサンドボックスに保存できます。これを行う際は、`data_dir` を `tfds.load` にも渡す必要があることに注意してください。`data_dir` 引数が指定されていない場合は、デフォルトの TFDS データディレクトリが使用されます。

#### データセットを読み込む

TFDS データセットが保存されたら、他のスクリプトから、またはデータにアクセスできるチームメイトがそれを読み込むことができます。

```python
# If no custom data dir was specified:
ds_test = tfds.load("my_dataset/single_number", split="test")

# When there are multiple versions, you can also specify the version.
ds_test = tfds.load("my_dataset/single_number:1.0.0", split="test")

# If the TFDS was stored in a custom folder, then it can be loaded as follows:
custom_data_dir = "/my/folder"
ds_test = tfds.load("my_dataset/single_number:1.0.0", split="test", data_dir=custom_data_dir)
```

#### 新しいバージョンまたは構成を追加する

データセットをさらにイテレートすると、ソースデータの一部の変換がさらに追加されるか、変更されます。このデータセットを保存して共有するために、これを簡単に新しいバージョンとして保存することができます。

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

### 新しいデータセットビルダークラスを定義する

このクラスに基づく新しい `DatasetBuilder` を定義することもできます。

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

### 形式

[CoNLL](https://aclanthology.org/W03-0419.pdf) は、アノテーション付きのテキストデータを表現する際に、一般的に使用される形式です。

CoNLL でフォーマットされたデータには通常、行ごとに 1 つのトークンと言語的なアノテーションが含まれます。同じ行内のアノテーションはスペースまたはタブで区切られているのが通例です。空の行は、文の境界を表します。

例として、以下の [conll2003](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/conll2003/conll2003.py) データセットの文を考察しましょう。これは、CoNLL アノテーション形式を使用しています。

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

新しい CoNLL ベースのデータセットを TFDS に追加するには、データセットビルダークラスのベースを `tfds.dataset_builders.ConllDatasetBuilder` にすることができます。この基底クラスには、CoNLL データセットの特異性に対処するための共通コードが含まれています（列ベースの形式、特徴量やタグのコンパイル済みのリストなどをイテレート）。

`tfds.dataset_builders.ConllDatasetBuilder` は CoNLL 固有の `GeneratorBasedBuilder` を実装します。CoNLL データセットビルダーの最低限の例として、以下のクラスを参照してください。

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

標準的なデータセットビルダーについては、クラスメソッド `_info` と `_split_generators` を上書きする必要があります。データセットによっては、[conll_dataset_builder_utils.py](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_builders/conll/conll_dataset_builder_utils.py) も更新して、データセット固有の特徴量とタグを含める必要がある可能性があります。

使用するデータセットに特有の実装が必要でない限り、`_generate_examples` メソッドでは他に上書きする必要はありません。

### 例

[conll2003](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/conll2003/conll2003.py) を、CoNLL 固有のデータセットビルダーを使用して実装されたデータセットの例として検討してみましょう。

### CLI

新しい CoNLL ベースのデータセットを書く際の最も簡単な方法は、[TFDS CLI](https://www.tensorflow.org/datasets/cli) を使用することです。

```sh
cd path/to/my/project/datasets/
tfds new my_dataset --format=conll   # Create `my_dataset/my_dataset.py` CoNLL-specific template files
```

## CoNLL-U

### 形式

[CoNLL-U](https://universaldependencies.org/format.html) は、アノテーション付きのテキストデータを表現する際に、一般的に使用される形式です。

CoNLL-U は、[マルチトークン単語](https://universaldependencies.org/u/overview/tokenization.html)のサポートなど、多数の特徴を追加することで、CoNLL 形式を強化しています。CoNLL-U でフォーマットされたデータには通常、行ごとに 1 つのトークンと言語的なアノテーションが含まれます。同じ行内のアノテーションは 1 つのタブ文字で区切られているのが通例です。空の行は、文の境界を表します。

一般に、それぞれの CoNLL-U アノテーション付き単語行には、[公式ドキュメント](https://universaldependencies.org/format.html)に記載のとおり、以下のフィールドが含まれています。

- ID: 単語のインデックス。新しい文につき、1 から開始する整数の ID です。マルチワードトークンの範囲である場合があります。空のノードの場合は 10 進数にすることができます（10 進数は 1 未満にすることもできますが、0 より大きくする必要があります）。
- FORM: 語形または句読点記号。
- LEMMA: 語形のレンマまたは語幹。
- UPOS: ユニバーサル PoS（品詞）タグ。
- XPOS: 言語固有の PoS タグ。利用できない場合はアンダースコア。
- FEATS: ユニバーサル特徴量インベントリまたは定義された言語固有の拡張からの形態学的特徴量のリスト。利用できない場合はアンダースコア。
- HEAD: 現在の単語の先頭。ID の値またはゼロ（0）のいずれか。
- DEPREL: HEAD（HEAD = 0 の場合のルート）または定義済みの言語固有のサブタイプ 1 に対するユニバーサル依存関係。
- DEPS: head-deprel ペアのリスト形式による拡張依存関係グラフ。
- MISC: その他のアノテーション。

例として、[公式ドキュメント](https://universaldependencies.org/format.html)にある、以下の CoNLL-U アノテーション付き文を見てみましょう。

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

新しい CoNLL ベースのデータセットを TFDS に追加するには、データセットビルダークラスのベースを `tfds.dataset_builders.ConllUDatasetBuilder` にすることができます。この基底クラスには、CoNLL-U データセットの特異性に対処するための共通コードが含まれています（列ベースの形式、特徴量やタグのコンパイル済みのリストなどをイテレート）。

`tfds.dataset_builders.ConllUDatasetBuilder` は CoNLL 固有の `GeneratorBasedBuilder` を実装します。CoNLL-U データセットビルダーの最低限の例として、以下のクラスを参照してください。

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

標準的なデータセットビルダーについては、クラスメソッド `_info` と `_split_generators` を上書きする必要があります。データセットによっては、[conll_dataset_builder_utils.py](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_builders/conll/conllu_dataset_builder_utils.py) も更新して、データセット固有の特徴量とタグを含める必要がある可能性があります。

使用するデータセットに特有の実装が必要でない限り、`_generate_examples` メソッドでは他に上書きする必要はありません。データセットに特定の前処理が必要な場合、たとえば、古典的でない[ユニバーサル依存関係](https://universaldependencies.org/guidelines.html)を検討する場合などは、[`generate_examples`](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_builders/conll/conllu_dataset_builder.py#L192) 関数の `process_example_fn` 属性を更新する必要がある場合があります（例として、[xtreme_pos](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/universal_dependencies/xtreme_pos.py) データセットをご覧ください）。

### 例

例として、CoNLL-U 固有のデータセットビルダーを使用する以下のデータセットを検討してください。

- [universal_dependencies](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/universal_dependencies/universal_dependencies.py)
- [xtreme_pos](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/universal_dependencies/xtreme_pos.py)

### CLI

新しい CoNLL-U ベースのデータセットを書く際の最も簡単な方法は、[TFDS CLI](https://www.tensorflow.org/datasets/cli) を使用することです。

```sh
cd path/to/my/project/datasets/
tfds new my_dataset --format=conllu   # Create `my_dataset/my_dataset.py` CoNLL-U specific template files
```
