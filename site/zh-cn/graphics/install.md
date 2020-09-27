# 安装 TensorFlow Graphics

## 稳定版本

TensorFlow Graphics 依赖于 [TensorFlow](https://www.tensorflow.org/install) 1.13.1 或更高版本。此外，它还支持 Nightly 版本的 TensorFlow (tf-nightly)。

要从 [PyPI](https://pypi.org/project/tensorflow-graphics/) 安装最新的 CPU 版本，请运行以下代码：

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --upgrade tensorflow-graphics
```

然后，安装最新的 GPU 版本，请运行以下代码：

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --upgrade tensorflow-graphics-gpu
```

有关其他安装帮助、安装先决条件指导以及（可选）设置虚拟环境的信息，请参阅 [TensorFlow 安装指南](https://www.tensorflow.org/install)。

## 从源代码安装 - macOS/Linux

您还可以通过执行以下命令从源代码安装：

```shell
git clone https://github.com/tensorflow/graphics.git
sh build_pip_pkg.sh
pip install --upgrade dist/*.whl
```

## 安装可选包 - Linux

要使用 TensorFlow Graphics EXR 数据加载器，需要安装 OpenEXR。可以通过运行以下命令来完成此操作：

```
sudo apt-get install libopenexr-dev
pip install --upgrade OpenEXR
```
