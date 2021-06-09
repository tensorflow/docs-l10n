<!--* freshness: { owner: 'wgierke' reviewed: '2021-03-09' } *-->

# 安装

## 安装 tensorflow_hub

`tensorflow_hub` 库可与 TensorFlow 1 和 TensorFlow 2 一起安装。我们建议新用户立即从 TensorFlow 2 开始，并且当前用户应升级到此版本。

### 与 TensorFlow 2 一起使用

像往常一样使用 [pip](https://pip.pypa.io/) 来[安装 TensorFlow 2](https://www.tensorflow.org/install)。（有关 GPU 支持的更多说明，请参阅相关文档。）随后安装当前版本的 [`tensorflow-hub`](https://pypi.org/project/tensorflow-hub/)（必须为 0.5.0 或更高版本）。

```bash
$ pip install "tensorflow>=2.0.0"
$ pip install --upgrade tensorflow-hub
```

TensorFlow Hub 的 TF1 样式 API 适用于 TensorFlow 2 的 v1 兼容模式。

### 与旧版 TensorFlow 1 一起使用

TensorFlow 1.15 是 TensorFlow 1.x 仍受 `tensorflow_hub` 库（自版本 0.11.0 起）支持的唯一版本。TensorFlow 1.15 默认采用与 TF1 兼容的行为，但其底层包含了许多 TF2 功能，允许使用 TensorFlow Hub 的 TF2 样式 API。

```bash
$ pip install "tensorflow>=1.15,<2.0"
$ pip install --upgrade tensorflow-hub
```

### 使用预发布版本

pip 软件包 `tf-nightly` 和 `tf-hub-nightly` 是从 GitHub 上的源代码自动构建的，没有版本测试。这样，开发者无需[从源代码构建](build_from_source.md)便可试用最新代码。

```bash
$ pip install tf-nightly
$ pip install --upgrade tf-hub-nightly
```

## 后续步骤

- [库概览](lib_overview.md)
- 教程：
    - [文本分类](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb)
    - [图像分类](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb)
    - [GitHub](https://github.com/tensorflow/hub/blob/master/examples/README.md) 上的更多示例
- 在 [tfhub.dev](https://tfhub.dev) 上查找模型
