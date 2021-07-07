# 安装 TensorFlow 模型优化

建议在安装之前创建一个 Python 虚拟环境。请参阅 TensorFlow 安装[指南](https://www.tensorflow.org/install/pip#2.-create-a-virtual-environment-recommended)获取更多信息 。

### 稳定版本

要安装最新版本，请运行以下代码：

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version. pip install --user --upgrade tensorflow-model-optimization
```

有关版本的详细信息，请参阅[版本说明](https://github.com/tensorflow/model-optimization/releases)。

有关所需的 TensorFlow 版本和其他兼容性信息，请根据您打算使用的技术，参阅其“概述”页面的“API 兼容性矩阵”部分。例如，对于剪枝，“概述”页面位于[此处](https://www.tensorflow.org/model_optimization/guide/pruning)。

由于TensorFlow *未*包含在 TensorFlow 模型优化软件包（位于 `setup.py` 中）的依赖项中，您必须显式安装 TensorFlow 软件包（`tf-nightly` 或 `tf-nightly-gpu`）。如此一来，我们只需维护一个软件包，而无需分别维护支持 CPU 和 GPU 的 TensorFlow 软件包。

### 从源代码安装

您也可以从源代码安装。这需要 [Bazel](https://bazel.build/) 构建系统。

```shell
# To install dependencies on Ubuntu: # sudo apt-get install bazel git python-pip # For other platforms, see Bazel docs above. git clone https://github.com/tensorflow/model-optimization.git cd model_optimization bazel build --copt=-O3 --copt=-march=native :pip_pkg PKGDIR=$(mktemp -d) ./bazel-bin/pip_pkg $PKGDIR pip install --user --upgrade $PKGDIR/*.whl
```
