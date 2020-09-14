# 安装和使用说明

## 安装 tensorflow_hub

`tensorflow_hub` 库可与 TensorFlow 1 和 TensorFlow 2 一起安装。我们建议新用户立即从 TensorFlow 2 开始，并且当前用户升级到此版本。

### 与 TensorFlow 2 一起使用

像往常一样使用 [pip](https://pip.pypa.io/) 来[安装 TensorFlow 2](https://www.tensorflow.org/install)。（有关 GPU 支持的更多说明，请参阅相关文档。）随后在它旁边安装当前版本的 [`tensorflow-hub`](https://pypi.org/project/tensorflow-hub/)（必须为 0.5.0 或更高版本）。

```bash
$ pip install "tensorflow>=2.0.0"
$ pip install --upgrade tensorflow-hub
```

TensorFlow Hub 的 TF1 样式 API 适用于 TensorFlow 2 的 v1 兼容模式。

### 与旧版 TensorFlow 1 一起使用

`tensorflow_hub` 库需要 TensorFlow version 1.7 或更高版本。

我们强烈建议您使用 TensorFlow 1.15 进行安装，此版本默认采用与 TF1 兼容的行为，但其底层包含了许多 TF2 功能，允许使用 TensorFlow Hub 的 TF2 样式 API。

```bash
$ pip install "tensorflow>=1.15,<2.0"
$ pip install --upgrade tensorflow-hub
```

### 使用预发布版本

pip 软件包 `tf-nightly` 和 `tf-hub-nightly` 是从 GitHub 上的源代码自动构建的，没有版本测试。这样，开发者无需[从源代码构建](build_from_source.md)便可试用最新代码。

### 可选：设置下载的缓存位置。

默认情况下，`tensorflow_hub` 使用系统范围的临时目录来缓存下载和未压缩的模型。有关使用其他可能更持久位置的选项，请参阅[缓存](caching.md)。

## API 稳定性

尽管我们希望避免重大更改，但此项目仍在积极开发中，尚不能保证具有一个稳定的 API 或模型格式。

## 公平性

与所有机器学习一样，[公平性](http://ml-fairness.com)是一个[重要的](https://research.googleblog.com/2016/10/equality-of-opportunity-in-machine.html)考量因素。许多预训练模型都是基于大型数据集训练的。在重用任何模型时，请务必牢记该模型基于哪些数据进行了训练（以及其中是否存在任何现有偏差）以及这些数据如何影响您的使用。

## 安全性

由于模型包含任意 TensorFlow 计算图，可以将它们视为程序。[安全地使用 TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md) 介绍了从不受信任的来源引用模型的安全隐患。
