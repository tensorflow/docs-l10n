# 数据集版本控制

## 定义

版本有两种不同含义：

- TFDS API 版本（pip 版本）：`tfds.__version__`
- 公共数据集版本，独立于 TFDS（例如 [Voc2007](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)、Voc2012）。在 TFDS 中，每个公共数据集版本都应作为一个独立的数据集实现：
    - 通过[构建器配置](https://www.tensorflow.org/datasets/add_dataset#dataset_configurationvariants_tfdscorebuilderconfig)：例如 `voc/2007`、`voc/2012`
    - 作为 2 个独立的数据集：例如 `wmt13_translate`、`wmt14_translate`
- TFDS 中的数据集生成代码版本 (`my_dataset:1.0.0`)：例如，如果在 `voc/2007` 的 TFDS 实现中发现错误，将更新 `voc.py` 生成代码 (`voc/2007:1.0.0` -&gt; `voc/2007:2.0.0`)。

本指南的其余部分仅关注最后一个定义（TFDS 仓库中的数据集代码版本）。

## 支持的版本

作为一般规则：

- 只能生成上一个最新版本。
- 可以读取之前生成的所有数据集（注：这需要使用 TFDS 4+ 生成的数据集）。

```python
builder = tfds.builder('my_dataset')
builder.info.version  # Current version is: '2.0.0'

# download and load the last available version (2.0.0)
ds = tfds.load('my_dataset')

# Explicitly load a previous version (only works if
# `~/tensorflow_datasets/my_dataset/1.0.0/` already exists)
ds = tfds.load('my_dataset:1.0.0')
```

## 语义

TFDS 中定义的每个 `DatasetBuilder` 都有一个版本，例如：

```python
class MNIST(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('2.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release',
      '2.0.0': 'Update dead download url',
  }
```

该版本遵循[语义化版本控制规范 2.0.0](https://semver.org/spec/v2.0.0.html)：`MAJOR.MINOR.PATCH`。版本的目的是为了保证重现性：加载固定版本的指定数据集会产生相同的数据。进一步来说：

- 如果增大 `PATCH` 版本，则客户端读取的数据是相同的，尽管数据可能会在磁盘上以不同的方式序列化，或者元数据可能已发生变化。对于任何给定的切片，slicing API 都会返回相同的记录集。
- 如果增大 `MINOR` 版本，则客户端读取的现有数据是相同的，但是还包含其他数据（每条记录中的特征）。对于任何给定的切片，slicing API 都会返回相同的记录集。
- 如果增大 `MAJOR` 版本，则现有数据已更改，并且/或者 slicing API 不一定会为给定切片返回相同的记录集。

对 TFDS 库进行代码更改并且该代码更改影响客户端对数据集进行序列化和/或读取的方式时，则根据上述准则增大相应的构建器版本。

请注意，上述语义化方案并非完美，当版本未递增时，可能会出现一些未被注意的错误对数据集产生影响。此类错误最终会得到修复，但是如果您严重依赖版本控制，我们建议您使用已发布版本（而非 `HEAD`）中的 TFDS。

还要注意，某些数据集具有独立于 TFDS 版本的另一种版本控制方案。例如，Open Images 数据集具有多个版本，在 TFDS 中，相应的构建器是 `open_images_v4`、`open_images_v5`...

## 加载特定版本

加载数据集或 `DatasetBuilder` 时，您可以指定要使用的版本。例如：

```python
tfds.load('imagenet2012:2.0.1')
tfds.builder('imagenet2012:2.0.1')

tfds.load('imagenet2012:2.0.0')  # Error: unsupported version.

# Resolves to 3.0.0 for now, but would resolve to 3.1.1 if when added.
tfds.load('imagenet2012:3.*.*')
```

如果使用 TFDS 发布，我们建议您：

- **仅修复版本的 `MAJOR` 部分**；
- **公布结果中使用了哪个版本的数据集。**

这样做可便于您在未来，或便于读者和审阅者重现您的结果。

## BUILDER_CONFIGS 和版本

有些数据集定义了多项 `BUILDER_CONFIGS`。此时，`version` 和 `supported_versions` 是在配置对象自身上定义的。除此之外，语义和用法相同。例如：

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

## 实验版本

注：下面是不佳的做法，容易出错，应当阻止。

可以允许同时生成 2 个版本。一个默认版本和一个实验版本。例如：

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

在代码中，您需要确保支持 2 个版本：

```python
class MNIST(tfds.core.GeneratorBasedBuilder):

  ...

  def _generate_examples(self, path):
    if self.info.version >= '2.0.0':
      ...
    else:
      ...
```
