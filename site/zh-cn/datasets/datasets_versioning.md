# 数据集版本控制

- [语义化](#semantic)
- [支持的版本](#supported-versions)
- [加载特定版本](#loading-a-specific-version)
- [实验](#experiments)
- [BUILDER_CONFIGS 和版本](#builder-configs-and-versions)

## 语义化

TFDS 中定义的每个 `DatasetBuilder` 都有一个版本，例如：

```py
class MNIST(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version("1.0.0")
```

该版本遵循[语义化版本控制规范 2.0.0](https://semver.org/spec/v2.0.0.html)：`MAJOR.MINOR.PATCH`。版本的目的是为了保证重现性：加载固定版本的指定数据集会产生相同的数据。进一步来说：

- 如果增大 `PATCH` 版本，尽管数据在磁盘上的序列化可能不同，或者元数据可能已更改，但客户端读取的数据是相同的。对于任何给定的切片，切片 API 会返回相同的记录集。
- 如果增大 `MINOR` 版本，则客户端读取的现有数据是相同的，但是还包含其他数据（每条记录中的特征）。对于任何给定的切片，切片 API 会返回相同的记录集。
- 如果增大 `MAJOR` 版本，则现有数据已更改，并且/或者切片 API 不一定会为给定切片返回相同的记录集。

对 TFDS 库进行代码更改并且该代码更改影响客户端对数据集进行序列化和/或读取的方式时，则根据上述准则增大相应的构建器版本。

请注意，上述语义化方案并非完美，当版本未递增时，可能会出现一些未被注意的错误对数据集产生影响。此类错误最终会得到修复，但是如果您严重依赖版本控制，我们建议您使用已发布版本（而非 `HEAD`）中的 TFDS。

还要注意，某些数据集具有独立于 TFDS 版本的另一种版本控制方案。例如，Open Images 数据集具有多个版本，在 TFDS 中，相应的构建器是 `open_images_v4`、`open_images_v5`...

## 支持的版本

`DatasetBuilder` 可以支持多个版本，这些版本可以高于或低于规范版本。例如：

```py
class Imagenet2012(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('2.0.1', 'Encoding fix. No changes from user POV')
  SUPPORTED_VERSIONS = [
      tfds.core.Version('3.0.0', 'S3: tensorflow.org/datasets/splits'),
      tfds.core.Version('1.0.0'),
      tfds.core.Version('0.0.9', tfds_version_to_prepare="v1.0.0"),
  ]
```

是否选择继续支持较旧版本取决于具体情况，这主要是基于数据集和版本的普及程度。最终，我们的目标是每个数据集仅支持有限数量的版本，最好是一个。在上面的示例中，我们可以看到不再支持 `2.0.0` 版本，从读者的角度来看，该版本与 `2.0.1` 相同。

数字高于规范版本号的受支持版本被视为实验版本，并且可能遭到破坏。但是，它们最终会被规范化。

版本可以指定 `tfds_version_to_prepare`。这意味着如果数据集已由旧版代码准备而成，则该数据集版本只能与当前版本的 TFDS 代码一起使用。`tfds_version_to_prepare` 的值指定最新已知版本的 TFDS，该版本可用于下载和准备此版本的数据集。

## 加载特定版本

加载数据集或 `DatasetBuilder` 时，您可以指定要使用的版本。例如：

```py
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

## 实验

为了逐步落实 TFDS 中的更改以减小对大量数据集构建器造成的影响，我们引入了实验的概念。首次引入时，实验在默认情况下处于禁用状态，但在特定数据集版本中可以决定是否启用实验。首先，通常会在“未来”版本（尚不规范化）上实施。例如：

```py
class MNIST(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version("1.0.0")
  SUPPORTED_VERSIONS = [
      tfds.core.Version("2.0.0", "EXP1: Opt-in for experiment 1",
                        experiments={tfds.core.Experiment.EXP1: True}),
  ]
```

实验经验证可行后，将扩展到所有或大多数数据集，此时可默认启用，以上定义将如下所示：

```py
class MNIST(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version("1.0.0",
                              experiments={tfds.core.Experiment.EXP1: False})
  SUPPORTED_VERSIONS = [
      tfds.core.Version("2.0.0", "EXP1: Opt-in for experiment 1"),
  ]
```

当实验被应用于所有数据集版本后（没有任何数据集版本指定 `{experiment: False}`），就可以删除该实验。

实验及其说明在 `core/utils/version.py` 中定义。

## BUILDER_CONFIGS 和版本

有些数据集定义了多项 `BUILDER_CONFIGS`。此时，`version` 和 `supported_versions` 是在配置对象自身上定义的。除此之外，语义和用法相同。例如：

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
