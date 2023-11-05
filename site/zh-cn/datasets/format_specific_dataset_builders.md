# 格式特定的数据集构建器

[TOC]

本指南介绍了 TFDS 中当前可用的所有格式特定的数据集构建器。

格式特定的数据集构建器是 [`tfds.core.GeneratorBasedBuilder`](https://www.tensorflow.org/datasets/api_docs/python/tfds/core/GeneratorBasedBuilder) 的子类，后者负责特定数据格式的大部分数据处理。

## 基于 `tf.data.Dataset` 的数据集

要从 `tf.data.Dataset`（[参考](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)）格式的数据集创建 TFDS 数据集，您可以使用 `tfds.dataset_builders.TfDataBuilder`（请参阅 [API 文档](https://www.tensorflow.org/datasets/api_docs/python/tfds/dataset_builders/TfDataBuilder)）。

我们设想此类的两个典型用途：

- 在类似笔记本的环境中创建实验数据集
- 在代码中定义数据集构建器

### 从笔记本创建新数据集

假设您在笔记本中工作，将一些数据加载为 `tf.data.Dataset`，应用了各种转换（映射、过滤等），现在您想要存储这些数据并轻松共享给队友或将其加载到其他笔记本中。无需定义新的数据集构建器类，即可实例化 `tfds.dataset_builders.TfDataBuilder` 并调用 `download_and_prepare` 将数据集存储为 TFDS 数据集。

由于它是一个 TFDS 数据集，您可以将其版本化、使用配置、进行不同的拆分，并将其记录下来以供后续轻松使用。这意味着您还必须告知 TFDS 数据集中的特征是什么。

下面是一个它的使用方法的虚拟示例。

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

`download_and_prepare` 方法将遍历输入 `tf.data.Dataset` 并将相应的 TFDS 数据集存储在 `/my/folder/my_dataset/single_number/1.0.0` 中，其中同时包含训练拆分和测试拆分。

`config` 是可选参数，如果您想在同一个数据集下存储不同的配置，它会十分有用。

`data_dir` 参数可用于将生成的 TFDS 数据集存储到不同的文件夹中，例如，如果您（还）不想共享给其他人，则可以将其存储在您自己的沙盒中。请注意，执行此操作时，您还需要将 `data_dir` 传递给 `tfds.load`。如果未指定 `data_dir` 参数，则会使用默认的 TFDS 数据目录。

#### 加载数据集

TFDS 数据集存储后，可以从其他脚本或由队友加载（如果他们有权访问数据）：

```python
# If no custom data dir was specified:
ds_test = tfds.load("my_dataset/single_number", split="test")

# When there are multiple versions, you can also specify the version.
ds_test = tfds.load("my_dataset/single_number:1.0.0", split="test")

# If the TFDS was stored in a custom folder, then it can be loaded as follows:
custom_data_dir = "/my/folder"
ds_test = tfds.load("my_dataset/single_number:1.0.0", split="test", data_dir=custom_data_dir)
```

#### 添加新版本或配置

在您的数据集上进一步迭代后，您可能已经添加或更改了源数据的一些转换。要存储和共享此数据集，您可以将其轻松存储为新版本。

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

### 定义新的数据集构建器类

您还可以基于此类定义新的 `DatasetBuilder`。

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

### 格式

[CoNLL](https://aclanthology.org/W03-0419.pdf) 是一种用于表示带注解的文本数据的常见格式。

CoNLL 格式化的数据通常包含一个词例，每行带有其语言注解；在同一行中，注解通常用空格或制表符分隔。空行代表句子边界。

以 [conll2003](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/conll2003/conll2003.py) 数据集中的以下句子为例，该数据集遵循 CoNLL 注解格式：

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

要将基于 CoNLL 的新数据集添加到 TFDS，您可以基于 `tfds.dataset_builders.ConllDatasetBuilder` 定义数据集构建器类。此基本类包含处理 CoNLL 数据集特殊性的通用代码（迭代基于列的格式、预编译的特征和标签列表...）。

`tfds.dataset_builders.ConllDatasetBuilder` 实现了一个 CoNLL 特定的 `GeneratorBasedBuilder`。请参阅以下类作为 CoNLL 数据集构建器的最小示例：

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

对于标准数据集构建器，需要覆盖类方法 `_info` 和 `_split_generators`。根据数据集，您可能还需要更新 [conll_dataset_builder_utils.py](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_builders/conll/conll_dataset_builder_utils.py) 以包含数据集特定的特征和标签列表。

`_generate_examples` 方法不需要进一步覆盖，除非您的数据集需要特定实现。

### 示例

将 [conll2003](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/conll2003/conll2003.py) 视为使用 CoNLL 特定的数据集构建器实现的数据集示例。

### CLI

编写基于 CoNLL 的新数据集的最简单方式是使用 [TFDS CLI](https://www.tensorflow.org/datasets/cli)：

```sh
cd path/to/my/project/datasets/
tfds new my_dataset --format=conll   # Create `my_dataset/my_dataset.py` CoNLL-specific template files
```

## CoNLL-U

### 格式

[CoNLL-U](https://universaldependencies.org/format.html) 是一种用于表示带注解的文本数据的常见格式。

CoNLL-U 通过添加许多功能来增强 CoNLL 格式，例如对[多词例词](https://universaldependencies.org/u/overview/tokenization.html)的支持。CoNLL-U 格式化的数据通常包含一个词例，每行带有其语言注解；在同一行中，注解通常用单个制表符分隔。空行代表句子边界。

通常，每个 CoNLL-U 注解的词行都包含以下字段，如[官方文档](https://universaldependencies.org/format.html)中所述：

- ID：词索引，每个新句子从 1 开始的整数；可能是多词词例的范围；可以是空节点的十进制数（十进制数可以小于 1，但必须大于 0）。
- FORM：词形式或标点符号。
- LEMMA：词形式的词元或词干。
- UPOS：通用词性标签。
- XPOS：语言特定的词性标签；如果不可用，则添加下划线。
- FEATS：来自通用特征清单或定义的语言特定扩展的形态特征列表；如果不可用，则添加下划线。
- HEAD：当前词的 head，可以是 ID 值，也可以是零 (0)。
- DEPREL：与 HEAD 关联的通用依赖关系（如果 HEAD = 0，则为根）或定义的语言特定子类型之一。
- DEPS：head-deprel 对列表形式的增强依赖关系图。
- MISC：任何其他注解。

以[官方文档](https://universaldependencies.org/format.html)中的以下 CoNLL-U 注解句子为例：

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

要将基于 CoNLL-U 的新数据集添加到 TFDS，您可以基于 `tfds.dataset_builders.ConllUDatasetBuilder` 定义数据集构建器类。此基本类包含处理 CoNLL-U 数据集特殊性的通用代码（迭代基于列的格式、预编译的特征和标签列表...）。

`tfds.dataset_builders.ConllUDatasetBuilder` 实现了一个 CoNLL-U 特定的 `GeneratorBasedBuilder`。请参阅以下类作为 CoNLL-U 数据集构建器的最小示例：

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

对于标准数据集构建器，需要覆盖类方法 `_info` 和 `_split_generators`。根据数据集，您可能还需要更新 [conll_dataset_builder_utils.py](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_builders/conll/conllu_dataset_builder_utils.py) 以包含数据集特定的特征和标签列表。

`_generate_examples` 方法不需要进一步覆盖，除非您的数据集需要特定实现。请注意，如果您的数据集需要特定的预处理 - 例如，如果它考虑非经典[通用依赖关系特征](https://universaldependencies.org/guidelines.html) - 您可能需要更新 [`generate_examples`](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_builders/conll/conllu_dataset_builder.py#L192) 函数的 `process_example_fn` 特性（请参阅 [xtreme_pos](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/universal_dependencies/xtreme_pos.py) 数据集作为示例）。

### 示例

以下面使用 CoNNL-U 特定数据集构建器的数据集为例：

- [universal_dependencies](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/universal_dependencies/universal_dependencies.py)
- [xtreme_pos](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/universal_dependencies/xtreme_pos.py)

### CLI

编写基于 CoNLL-U 的新数据集的最简单方式是使用 [TFDS CLI](https://www.tensorflow.org/datasets/cli)：

```sh
cd path/to/my/project/datasets/
tfds new my_dataset --format=conllu   # Create `my_dataset/my_dataset.py` CoNLL-U specific template files
```
