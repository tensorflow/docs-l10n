<!-- Copyright 2018 The TensorFlow Hub Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=============================================================================-->

# 使用 Linux 创建 TensorFlow Hub pip 软件包

注：本文档适用于有兴趣修改 TensorFlow Hub 本身的开发者。要*使用* TensorFlow Hub，请参阅[安装说明](installation.md)

如果您对 TensorFlow Hub pip 软件包进行更改，则可能需要从源代码重新构建 pip 软件包来尝试进行更改。

这需要：

- Python
- TensorFlow
- Git
- [Bazel](https://docs.bazel.build/versions/master/install.html)

或者，如果您安装 protobuf 编译器，则可以[在不使用 bazel 的情况下尝试更改](#develop)。

## 设置 virtualenv

<a id="setup"></a>

### 激活 virtualenv

如果尚未安装 virtualenv，请先安装：

```shell
~$ sudo apt-get install python-virtualenv
```

创建一个用于创建软件包的虚拟环境：

```shell
~$ virtualenv --system-site-packages tensorflow_hub_env
```

然后激活它：

```shell
~$ source ~/tensorflow_hub_env/bin/activate  # bash, sh, ksh, or zsh
~$ source ~/tensorflow_hub_env/bin/activate.csh  # csh or tcsh
```

### 克隆 TensorFlow Hub 仓库。

```shell
(tensorflow_hub_env)~/$ git clone https://github.com/tensorflow/hub
(tensorflow_hub_env)~/$ cd hub
```

## 测试您的更改

### 运行 TensorFlow Hub 的测试

```shell
(tensorflow_hub_env)~/hub/$ bazel test tensorflow_hub:all
```

## 构建并安装软件包

### 构建 TensorFlow Hub pip 打包脚本

要为 TensorFlow Hub 构建 pip 软件包，请运行以下代码：

```shell
(tensorflow_hub_env)~/hub/$ bazel build tensorflow_hub/pip_package:build_pip_package
```

### 创建 TensorFlow Hub pip 软件包

```shell
(tensorflow_hub_env)~/hub/$ bazel-bin/tensorflow_hub/pip_package/build_pip_package \
/tmp/tensorflow_hub_pkg
```

### 安装并测试 pip 软件包（可选）

运行以下命令来安装 pip 软件包。

```shell
(tensorflow_hub_env)~/hub/$ pip install /tmp/tensorflow_hub_pkg/*.whl
```

测试导入 TensorFlow Hub：

```shell
(tensorflow_hub_env)~/hub/$ cd ..  # exit the directory to avoid confusion
(tensorflow_hub_env)~/$ python -c "import tensorflow_hub as hub"
```

## “开发者”安装（实验性）

<a id="develop"></a>

警告：这种运行 TensorFlow 的方法是实验性的，未获得 TensorFlow Hub 团队的官方支持。

使用 bazel 构建软件包是唯一受官方支持的方法。但是，如果您不熟悉 bazel，则使用开源工具更为简单。为此，您可以对该软件包执行“开发者安装”。

这种安装方法允许您将工作目录安装到 Python 环境中，以便在导入软件包时反映正在进行的更改。

### 设置仓库

如[上文](#setup)所述，首先设置 virtualenv 和仓库。

### 安装 `protoc`

由于 TensorFlow Hub 使用 protobuf，您将需要 protobuf 编译器以从 `.proto` 文件创建必要的 Python `_pb2.py` 文件。

#### 在 Mac 上：

```
(tensorflow_hub_env)~/hub/$ brew install protobuf
```

#### 在 Linux 上：

```
(tensorflow_hub_env)~/hub/$ sudo apt install protobuf-compiler
```

### 编译 `.proto` 文件

最初，目录中没有 `_pb2.py` 文件：

```
(tensorflow_hub_env)~/hub/$ ls -1 tensorflow_hub/*_pb2.py
```

运行 `protoc` 来创建这些文件：

```
(tensorflow_hub_env)~/hub/$ protoc -I=tensorflow_hub --python_out=tensorflow_hub tensorflow_hub/*.proto
(tensorflow_hub_env)~/hub/$ ls -1 tensorflow_hub/*_pb2.py
```

<pre> tensorflow_hub/image_module_info_pb2.py tensorflow_hub/module_attachment_pb2.py tensorflow_hub/module_def_pb2.py</pre>

注：如果对 `.proto` 定义进行更改，请不要忘记重新编译 `_pb2.py` 文件。

### 直接从仓库导入

在 `_pb2.py` 文件准备好后，您可以直接在 TensorFlow Hub 目录中试用您的修改：

```
(tensorflow_hub_env)~/$ python -c "import tensorflow_hub as hub"
```

### 在“开发者”模式下安装

或者，要从仓库根目录之外使用它，可以使用 `setup.py develop` 安装：

```
(tensorflow_hub_env)~/hub/$ python tensorflow_hub/pip_package/setup.py develop
```

现在，您可以在常规 Python virtualenv 中使用本地更改，而无需为每个新更改重新构建并安装 pip 软件包：

```shell
(tensorflow_hub_env)~/hub/$ cd ..  # exit the directory to avoid confusion
(tensorflow_hub_env)~/$ python -c "import tensorflow_hub as hub"
```

## 停用 virtualenv

```shell
(tensorflow_hub_env)~/hub/$ deactivate
```
