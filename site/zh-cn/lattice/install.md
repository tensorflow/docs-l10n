# 安装 TensorFlow Lattice

要使用 TensorFlow Lattice (TFL)，您可以通过几种方式设置环境：

- 最简单的 TFL 学习和使用方法无需安装：只需运行任何教程（例如[Canned Estimator 教程](tutorials/canned_estimators.ipynb)）。
- 要在本地计算机上使用 TFL，请安装 `tensorflow-lattice` pip 软件包。
- 如果您的计算机配置比较独特，您可以从源代码构建软件包。

## 使用 pip 安装 TensorFlow Lattice

使用 pip 安装。

```shell
pip install --upgrade tensorflow-lattice
```

## 从源代码构建

克隆 GitHub 仓库：

```shell
git clone https://github.com/tensorflow/lattice.git
```

从源代码构建 pip 软件包：

```shell
python setup.py sdist bdist_wheel --universal --release
```

安装包：

```shell
pip install --user --upgrade /path/to/pkg.whl
```
