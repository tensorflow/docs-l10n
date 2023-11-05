# 编写自定义数据集

按照本指南创建新的数据集（可在 TFDS 或您自己的仓库中创建）。

请查看我们的[数据集列表](catalog/overview.md)，了解您希望使用的数据集是否已存在。

## 精彩速览

编写新数据集的最简单方式是使用 [TFDS CLI](https://www.tensorflow.org/datasets/cli)：

```sh
cd path/to/my/project/datasets/
tfds new my_dataset  # Create `my_dataset/my_dataset.py` template files
# [...] Manually modify `my_dataset/my_dataset_dataset_builder.py` to implement your dataset.
cd my_dataset/
tfds build  # Download and prepare the dataset to `~/tensorflow_datasets/`
```

要将新数据集与 `tfds.load('my_dataset')` 搭配使用：

- `tfds.load` 将自动检测并加载在 `~/tensorflow_datasets/my_dataset/` 中生成的数据集（例如，由 `tfds build` 生成）。
- 或者，您也可以显式 `import my.project.datasets.my_dataset` 以注册您的数据集。

```python
import my.project.datasets.my_dataset  # Register `my_dataset`

ds = tfds.load('my_dataset')  # `my_dataset` registered
```

## 概述

数据集以各种格式分布于各个角落，它们并不总是以可以立即馈入机器学习流水线的格式进行存储。

TFDS 将这些数据集处理成标准格式（外部数据 -&gt; 序列化文件），并随后作为机器学习流水线加载（序列化文件 -&gt; `tf.data.Dataset`）。序列化仅进行一次。后续访问将直接从这些预处理的文件读取。

大多数预处理都自动进行。每个数据集都实现 `tfds.core.DatasetBuilder` 的一个子类，该子类指定以下信息：

- 数据从何处来（即它的网址）；
- 数据集看起来像什么（即它的特征）；
- 数据应如何拆分（例如 `TRAIN` 和 `TEST`）；
- 以及数据集中的各个样本。

## 编写数据集

### 默认模板：`tfds new`

使用 [TFDS CLI](https://github.com/tensorflow/datasets/blob/master/CONTRIBUTING.md) 生成所需的模板 Python 文件。

```sh
cd path/to/project/datasets/  # Or use `--dir=path/to/project/datasets/` below
tfds new my_dataset
```

此命令将生成一个具有以下结构的新 `my_dataset/` 文件夹：

```sh
my_dataset/
    __init__.py
    README.md # Markdown description of the dataset.
    CITATIONS.bib # Bibtex citation for the dataset.
    TAGS.txt # List of tags describing the dataset.
    my_dataset_dataset_builder.py # Dataset definition
    my_dataset_dataset_builder_test.py # Test
    dummy_data/ # (optional) Fake data (used for testing)
    checksum.tsv # (optional) URL checksums (see `checksums` section).
```

在此处搜索 `TODO(my_dataset)` 并进行相应修改。

### 数据集样本

所有数据集都作为可以处理大多数样板的 `tfds.core.DatasetBuilder` 的子类实现。它支持：

- 可以在单台计算机上生成的中小型数据集（本教程）。
- 需要分布式生成的特大型数据集（使用 [Apache Beam](https://beam.apache.org/)，请参阅我们的[大型数据集指南](https://www.tensorflow.org/datasets/beam_datasets#implementing_a_beam_dataset)）。

以下是基于 `tfds.core.GeneratorBasedBuilder` 的数据集构建工具的最简单示例：

```python
class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata (homepage, citation,...)."""
    return self.dataset_info_from_configs(
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

请注意，对于某些特定的数据格式，我们提供了现成的[数据集构建工具](https://www.tensorflow.org/datasets/format_specific_dataset_builders)来负责大多数数据处理。

我们来详细了解要覆盖的 3 个抽象方法。

### `_info`：数据集元数据

`_info` 可返回包含[数据集元数据](https://www.tensorflow.org/datasets/overview#access_the_dataset_metadata)的 <code>tfds.core.DatasetInfo</code>。

```python
def _info(self):
  # The `dataset_info_from_configs` base method will construct the
  # `tfds.core.DatasetInfo` object using the passed-in parameters and
  # adding: builder (self), description/citations/tags from the config
  # files located in the same package.
  return self.dataset_info_from_configs(
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
  )
```

大多数字段均一目了然。以下是一些具体信息：

- `features`：该属性指定数据集结构、形状等内容。支持复杂数据类型（音频、视频、嵌套序列等）。有关详细信息，请参阅[可用特征](api_docs/python/tfds/core/DatasetInfo.md)或[特征连接器指南](https://www.tensorflow.org/datasets/features)。
- `disable_shuffling`：请参阅[维护数据集顺序](#maintain-dataset-order)部分。

编写 `BibText` `CITATIONS.bib` 文件：

- 在数据集网站中搜索引用说明（使用 BibTex 格式）。
- 对于 [arXiv](https://arxiv.org/) 论文：查找论文并点击右侧的 `BibText` 链接。
- 在 [Google Scholar](https://scholar.google.com) 上查找论文，并点击标题下方的双引号标志，然后在弹出窗口中点击 `BibTeX`。
- 如果没有相关的论文（例如，只有一个网站），您可以使用 [BibTeX 在线编辑器](https://truben.no/latex/bibtex/)创建一个自定义 BibTeX 条目（下拉菜单有一个 `Online` 条目类型）。

更新 `TAGS.txt` 文件：

- 所有允许的标签都预先填充在生成的文件中。
- 移除所有不适用于数据集的标签。
- [tensorflow_datasets/core/valid_tags.txt](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/valid_tags.txt) 中列出了有效标签。
- 要向该列表添加标签，请发送 PR。

#### 维护数据集顺序

认情况下，数据集记录在存储时会重排以使数据集中各个类的分布更加均匀，因为通常属于同一类的记录是连续的。为了指定应按 `_generate_examples` 提供的生成键对数据集进行排序，应将字段 `disable_shuffling` 设置为 `True`。该字段在默认情况下设置为 `False`。

```python
def _info(self):
  return self.dataset_info_from_configs(
    # [...]
    disable_shuffling=True,
    # [...]
  )
```

请记住，停用重排会对性能产生影响，因为将无法并行读取分片。

### `_split_generators`：下载和拆分数据

#### 下载和提取源数据

大多数据集都需要从网络下载数据。可使用 `_split_generators` 的输入参数 `tfds.download.DownloadManager` 实现。`dl_manager` 具有以下方法：

- `download`：支持 `http(s)://`、`ftp(s)://`
- `extract`：目前支持 `.zip`、`.gz` 和 `.tar` 文件。
- `download_and_extract`：与 `dl_manager.extract(dl_manager.download(urls))` 相同

上述所有方法均返回 `tfds.core.Path`（[`epath.Path`](https://github.com/google/etils) 的别名），后者是[类 pathlib.Path](https://docs.python.org/3/library/pathlib.html) 对象。

这些方法支持任意嵌套结构（`list`、`dict`），例如：

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

#### 手动下载和提取

某些数据无法自动下载（例如需要登录），在这种情况下，用户将手动下载源数据并将其放置在 `manual_dir/` 中（默认为 `~/tensorflow_datasets/downloads/manual/`）。

然后即可通过 `dl_manager.manual_dir` 访问文件：

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

`manual_dir` 的位置可以使用 `tfds build --manual_dir=` 或使用 `tfds.download.DownloadConfig` 进行自定义。

#### 直接读取归档

`dl_manager.iter_archive` 可以在不提取的情况下按顺序读取归档。这样可以节省存储空间并提高某些文件系统的性能。

```python
for filename, fobj in dl_manager.iter_archive('path/to/archive.zip'):
  ...
```

`fobj` 具有与 `with open('rb') as fobj:` 相同的方法（例如 `fobj.read()`）

#### 指定数据集拆分

如果数据集带有预定义的拆分（例如 `MNIST` 具有 `train` 和 `test` 拆分），请保留这些拆分。否则，请仅指定一个 `all` 拆分。用户可以使用 [subsplit API](https://www.tensorflow.org/datasets/splits) 动态创建自己的子拆分（例如 `split='train[80%:]'`）。请注意，除了上述 `all` 之外，任何字母字符串都可以用作拆分名称。

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

### `_generate_examples`：样本生成器

`_generate_examples` 可为元数据中的每项拆分生成样本。

此方法通常将读取源数据集工件（例如 CSV 文件）并产生 `(key, feature_dict)` 元组：

- `key`：样本标识符。用于使用 `hash(key)` 确定性地重排样本，或者在停用重排时根据键排序（请参阅[维护数据集顺序](#maintain-dataset-order)部分）。应为：
    - **唯一**：如果两个样本使用相同的键，则会引发异常。
    - **确定**：不应取决于 `download_dir`、`os.path.listdir` 顺序等。两次生成数据应产生相同的键。
    - **可比**：如果停用重排，将使用键对数据集排序。
- `feature_dict`：包含样本值的 `dict`。
    - 该结构应与 `tfds.core.DatasetInfo` 中定义的 `features=` 结构相匹配。
    - 复杂数据类型（图像、视频、音频等）将自动编码。
    - 每个特征通常都可接受多种输入类型（例如，视频接受 `/path/to/vid.mp4`、`np.array(shape=(l, h, w, c))`、`List[paths]`、`List[np.array(shape=(h, w, c)]`、`List[img_bytes]` 等）。
    - 如需了解详情，请参阅[特征连接器指南](https://www.tensorflow.org/datasets/features)。

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

警告：从字符串或整数解析布尔值时，请使用效用函数 `tfds.core.utils.bool_utils.parse_bool`，以避免解析错误（例如，`bool("False") == True`）。

#### 文件访问和 <code>tf.io.gfile</code>

为了支持云存储系统，请避免使用 Python 内置 I/O 运算。

`dl_manager` 将返回直接与 Google Cloud Storage 兼容的[类 pathlib](https://docs.python.org/3/library/pathlib.html) 对象：

```python
path = dl_manager.download_and_extract('http://some-website/my_data.zip')

json_path = path / 'data/file.json'

json.loads(json_path.read_text())
```

或者，使用 `tf.io.gfile` API 而非内置功能进行文件操作：

- `open` -&gt; `tf.io.gfile.GFile`
- `os.rename` -&gt; `tf.io.gfile.rename`
- ...

Pathlib 应优先于 `tf.io.gfile`（请参阅[原因](https://www.tensorflow.org/datasets/common_gotchas#prefer_to_use_pathlib_api)）。

#### 额外依赖项

某些数据集仅在生成期间需要额外 Python 依赖项。例如，SVHN 数据集会使用 `scipy` 来加载某些数据。

如果要将数据集添加到 TFDS 仓库中，请使用 `tfds.core.lazy_imports` 以控制 `tensorflow-datasets` 软件包的大小。用户将仅在需要时安装额外依赖项。

要使用 `lazy_imports`，请执行以下操作：

- 将数据集的条目添加到 [`setup.py`](https://github.com/tensorflow/datasets/tree/master/setup.py). 的 `DATASET_EXTRAS` 中。这样一来，用户就可以执行诸如 `pip install 'tensorflow-datasets[svhn]'` 来安装额外依赖项。
- 将要导入的条目添加到 [`LazyImporter`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/lazy_imports_lib.py) 和 [`LazyImportsTest`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/lazy_imports_lib_test.py)。
- 使用 `tfds.core.lazy_imports` 在您的 `DatasetBuilder` 中访问依赖项（例如，`tfds.core.lazy_imports.scipy`）。

#### 损坏的数据

某些数据集不是完全干净，包含一些损坏的数据（例如，图像在 JPEG 文件中，但有些是无效的 JPEG）。应跳过这些样本，但在数据集描述中要注明舍弃了多少样本及其原因。

### 数据集配置/变体 (tfds.core.BuilderConfig)

某些数据集可能具有多种变体，或在数据预处理和磁盘写入方式方面具有多种选项。例如，[cycle_gan](https://www.tensorflow.org/datasets/catalog/cycle_gan) 为每个对象对（`cycle_gan/horse2zebra`、`cycle_gan/monet2photo` 等）都提供了一个配置。

这可通过 `tfds.core.BuilderConfig` 实现：

1. 将您的配置对象定义为 `tfds.core.BuilderConfig` 的子类。例如 `MyDatasetConfig`。

    ```python
    @dataclasses.dataclass
    class MyDatasetConfig(tfds.core.BuilderConfig):
      img_size: Tuple[int, int] = (0, 0)
    ```

    注：必须使用默认值，原因请参见 https://bugs.python.org/issue33129。

2. 在 `MyDataset` 中定义 `BUILDER_CONFIGS = []` 类成员，该成员列出数据集公开的 `MyDatasetConfig`。

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

    注：必须使用 `# pytype: disable=wrong-keyword-args`，原因是数据类继承的 [Pytype 错误](https://github.com/google/pytype/issues/628)。

3. 在 `MyDataset` 中使用 `self.builder_config` 配置数据生成（例如 `shape=self.builder_config.img_size`）。这可能包括在 `_info()` 中设置不同的值，或更改下载数据的访问权限。

注：

- 每个配置都具有唯一的名称。配置的完全限定名称为 `dataset_name/config_name`（例如 `coco/2017`）。
- 如果未指定，将使用 `BUILDER_CONFIGS` 中的第一个配置（例如 `tfds.load('c4')` 默认值为 `c4/en`）

请参阅 [`anli`](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/anli.py#L69) 以查看使用 `BuilderConfig` 的数据集样本。

### 版本

版本有两种不同含义：

- “外部”原始数据版本：例如 COCO v2019、v2017 等
- “内部”TFDS 代码版本：例如，重命名 `tfds.features.FeaturesDict` 中的特征、修正 `_generate_examples` 中的错误

要更新数据集，请执行以下操作：

- 对于“外部”数据更新：可能同时会有多个用户希望访问特定的年份/版本。这可以通过对每个版本使用一个 `tfds.core.BuilderConfig`（例如 `coco/2017`、`coco/2019`）或对每个版本使用一个类（例如 `Voc2007`、`Voc2012`）来实现。
- 对于“内部”代码更新：用户仅下载最新版本。任何代码更新都应按照[语义化版本控制](https://www.tensorflow.org/datasets/datasets_versioning#semantic)提高 `VERSION` 类特性（例如从 `1.0.0` 到 `VERSION = tfds.core.Version('2.0.0')`）。

### 添加要注册的导入

不要忘记将数据集模块导入到项目 `__init__` 中，以在 `tfds.load`、`tfds.builder` 中自动注册。

```python
import my_project.datasets.my_dataset  # Register MyDataset

ds = tfds.load('my_dataset')  # MyDataset available
```

例如，如果您要向 `tensorflow/datasets` 贡献数据集，请将模块导入添加到其子目录的 `__init__.py`（例如 [`image/__init__.py`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/image/__init__.py)）中。

### 检查有无常见实现问题

请检查有无[常见实现问题](https://www.tensorflow.org/datasets/common_gotchas)。

## 测试您的数据集

### 下载并准备：`tfds build`

要生成数据集，请从 `my_dataset/` 目录运行 `tfds build`：

```sh
cd path/to/datasets/my_dataset/
tfds build --register_checksums
```

一些适用于开发的实用标志：

- `--pdb`：如果引发异常情况，则进入调试模式。
- `--overwrite`：如果数据集已经生成，则删除现有文件。
- `--max_examples_per_split`：仅生成前 X 个样本（默认为 1），而非完整数据集。
- `--register_checksums`：记录下载网址的校验和。应仅在开发时使用。

有关标志的完整列表，请参阅 [CLI 文档](https://www.tensorflow.org/datasets/cli#tfds_build_download_and_prepare_a_dataset)。

### 校验和

建议记录数据集的校验和以保证确定性，以及帮助编写文档等。可通过使用 `--register_checksums` 生成数据集予以实现（请参阅上一部分内容）。

如果您通过 PyPI 发布数据集，请不要忘记导出 `checksums.tsv` 文件（例如，在 `setup.py` 的 `package_data` 中）。

### 对数据集执行单元测试

`tfds.testing.DatasetBuilderTestCase` 是用于完整训练数据集的基础 `TestCase`。它使用“虚拟数据”作为测试数据来模拟源数据集的结构。

- 测试数据应放置在 `my_dataset/dummy_data/` 目录中，并应模拟下载和提取的源数据集工件。可以手动创建，也可以使用脚本（[示例脚本](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/datasets/bccd/dummy_data_generation.py)）自动创建。
- 确保在测试数据拆分中使用不同的数据，因为如果数据集拆分重叠，测试将失败。
- **测试数据不应包含任何受版权保护的材料**。如有疑问，请勿使用原始数据集中的材料创建数据。

```python
import tensorflow_datasets as tfds
from . import my_dataset_dataset_builder


class MyDatasetTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for my_dataset dataset."""
  DATASET_CLASS = my_dataset_dataset_builder.Builder
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  DL_EXTRACT_RESULT = {
      'name1': 'path/to/file1',  # Relative to my_dataset/dummy_data dir.
      'name2': 'file2',
  }


if __name__ == '__main__':
  tfds.testing.test_main()
```

运行以下命令以测试数据集。

```sh
python my_dataset_test.py
```

## 向我们发送反馈

我们一直在努力改进数据集创建工作流，但只有在我们意识到这些问题的情况下才能这样做。您在创建数据集时遇到了哪些问题或错误？是否有部分令人困惑，或者第一次没有运行？

请在 [GitHub](https://github.com/tensorflow/datasets/issues) 上分享您的反馈。
