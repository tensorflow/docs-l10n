## 开发

本文档包含在各种平台上设置开发环境和从源代码构建 `tensorflow-io` 软件包所需的必要信息。设置完成后，请参阅 [STYLE_GUIDE](https://github.com/tensorflow/io/blob/master/STYLE_GUIDE.md) 获取有关添加新操作的指南。

### IDE 设置

有关如何配置 Visual Studio Code 以开发 TensorFlow I/O 的说明，请参阅本[文档](https://github.com/tensorflow/io/blob/master/docs/vscode.md)。

### Lint

TensorFlow I/O 的代码符合 Bazel Buildifier、Clang Format、Black 和 Pyupgrade。请使用以下命令检查源代码并找出 lint 问题：

```
# Install Bazelisk (manage bazel version implicitly)
$ curl -sSOL https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
$ sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
$ sudo chmod +x /usr/local/bin/bazel
$ bazel run //tools/lint:check
```

对于 Bazel Buildifier 和 Clang Format，以下命令将自动找出并修正任何 lint 错误：

```
$ bazel run //tools/lint:lint
```

或者，如果您只想使用个别 linter 执行 lint 检查，则可以选择性地将 `black`、`pyupgrade`、`bazel` 或 `clang` 传递给上述命令。

例如，可以使用以下代码完成 `black` 特定 lint 检查：

```
$ bazel run //tools/lint:check -- black
```

可以使用 Bazel Buildifier 和 Clang Format 完成 lint 修正，代码如下：

```
$ bazel run //tools/lint:lint -- bazel clang
```

可以针对单个 python 文件使用 `black` 和 `pyupgrade` 完成 lint 检查，代码如下：

```
$ bazel run //tools/lint:check -- black pyupgrade -- tensorflow_io/python/ops/version_ops.py
```

使用 black 和 pyupgrade 对单个 python 文件进行 lint 修正，代码如下：

```
$ bazel run //tools/lint:lint -- black pyupgrade --  tensorflow_io/python/ops/version_ops.py
```

### Python

#### macOS

在 macOS Catalina 10.15.7 上，可以使用系统提供的 python 3.8.2 构建 tensorflow-io。要完成此操作，需要同时使用 `tensorflow` 和 `bazel`。

注：macOS 10.15.7 上的系统默认 python 3.8.2 会导致 `regex` 安装错误，该错误因编译器选项 `-arch arm64 -arch x86_64` 所致（类似于 https://github.com/giampaolo/psutil/issues/1832 中提到的议题）。为了克服此问题，需要使用 `export ARCHFLAGS="-arch x86_64"` 删除 arm64 构建选项。

```sh
#!/usr/bin/env bash

# Disable arm64 build by specifying only x86_64 arch.
# Only needed for macOS's system default python 3.8.2 on macOS 10.15.7
export ARCHFLAGS="-arch x86_64"

# Use following command to check if Xcode is correctly installed:
xcodebuild -version

# Show macOS's default python3
python3 --version

# Install Bazelisk (manage bazel version implicitly)
brew install bazelisk

# Install tensorflow and configure bazel
sudo ./configure.sh

# Add any optimization on bazel command, e.g., --compilation_mode=opt,
#   --copt=-msse4.2, --remote_cache=, etc.
# export BAZEL_OPTIMIZATION=

# Build shared libraries
bazel build -s --verbose_failures $BAZEL_OPTIMIZATION //tensorflow_io/... //tensorflow_io_gcs_filesystem/...

# Once build is complete, shared libraries will be available in
# `bazel-bin/tensorflow_io/core`, `bazel-bin/tensorflow_io/python/ops` and
# it is possible to run tests with `pytest`, e.g.:
sudo python3 -m pip install pytest
TFIO_DATAPATH=bazel-bin python3 -m pytest -s -v tests/test_serialization.py
```

注：运行 pytest 时，必须传递 `TFIO_DATAPATH=bazel-bin`，这样 python 才能在构建过程之后使用生成的共享库。

##### 故障排除

如果已安装 Xcode，但 `$ xcodebuild -version` 未显示预期输出，则您可能需要使用以下命令启用 Xcode 命令行：

`$ xcode-select -s /Applications/Xcode.app/Contents/Developer`.

可能需要重新启动终端才能使更改生效。

示例输出：

```
$ xcodebuild -version
Xcode 12.2
Build version 12B45b
```

#### Linux

Linux 上的 tensorflow-io 开发与 macOS 类似。所需的软件包为 gcc、g++、git、bazel 和 python 3。但可能需要较新版本的 gcc 或 python，而不是默认的系统安装版本。

##### Ubuntu 20.04

Ubuntu 20.04 需要 gcc/g++、git 和 python 3。下列代码将在 Ubuntu 20.04 上安装依赖项并构建共享库：

```sh
#!/usr/bin/env bash

# Install gcc/g++, git, unzip/curl (for bazel), and python3
sudo apt-get -y -qq update
sudo apt-get -y -qq install gcc g++ git unzip curl python3-pip

# Install Bazelisk (manage bazel version implicitly)
curl -sSOL https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
sudo chmod +x /usr/local/bin/bazel

# Upgrade pip
sudo python3 -m pip install -U pip

# Install tensorflow and configure bazel
sudo ./configure.sh

# Alias python3 to python, needed by bazel
sudo ln -s /usr/bin/python3 /usr/bin/python

# Add any optimization on bazel command, e.g., --compilation_mode=opt,
#   --copt=-msse4.2, --remote_cache=, etc.
# export BAZEL_OPTIMIZATION=

# Build shared libraries
bazel build -s --verbose_failures $BAZEL_OPTIMIZATION //tensorflow_io/... //tensorflow_io_gcs_filesystem/...

# Once build is complete, shared libraries will be available in
# `bazel-bin/tensorflow_io/core`, `bazel-bin/tensorflow_io/python/ops` and
# it is possible to run tests with `pytest`, e.g.:
sudo python3 -m pip install pytest
TFIO_DATAPATH=bazel-bin python3 -m pytest -s -v tests/test_serialization.py
```

##### CentOS 8

为 CentOS 8 构建共享库的步骤与上面针对 Ubuntu 20.04 的操作类似，不同之处在于应改用以下代码

```
sudo yum install -y python3 python3-devel gcc gcc-c++ git unzip which make
```

上述代码用于安装 gcc/g++、git, unzip/which（适用于 bazel）和 python3。

##### CentOS 7

在 CentOS 7 上，默认的 python 和 gcc 版本太旧，无法构建 tensorflow-io 的共享库 (.so)。应改用 Developer Toolset 和 rh-python36 提供的 gcc。此外，必须以静态方式链接 libstdc++，以避免安装在 CentOS 上的 libstdc++ 与 devtoolset 的较新 gcc 版本之间存在差异。

此外，必须将特殊标志 `--//tensorflow_io/core:static_build` 传递给 Bazel，以避免文件系统插件的静态链接库中出现重复符号。

以下代码将安装 bazel、devtoolset-9、rh-python36，并构建共享库：

```sh
#!/usr/bin/env bash

# Install centos-release-scl, then install gcc/g++ (devtoolset), git, and python 3
sudo yum install -y centos-release-scl
sudo yum install -y devtoolset-9 git rh-python36 make

# Install Bazelisk (manage bazel version implicitly)
curl -sSOL https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
sudo chmod +x /usr/local/bin/bazel

# Upgrade pip
scl enable rh-python36 devtoolset-9 \
    'python3 -m pip install -U pip'

# Install tensorflow and configure bazel with rh-python36
scl enable rh-python36 devtoolset-9 \
    './configure.sh'

# Add any optimization on bazel command, e.g., --compilation_mode=opt,
#   --copt=-msse4.2, --remote_cache=, etc.
# export BAZEL_OPTIMIZATION=

# Build shared libraries, notice the passing of --//tensorflow_io/core:static_build
BAZEL_LINKOPTS="-static-libstdc++ -static-libgcc" BAZEL_LINKLIBS="-lm -l%:libstdc++.a" \
  scl enable rh-python36 devtoolset-9 \
    'bazel build -s --verbose_failures $BAZEL_OPTIMIZATION --//tensorflow_io/core:static_build //tensorflow_io/...'

# Once build is complete, shared libraries will be available in
# `bazel-bin/tensorflow_io/core`, `bazel-bin/tensorflow_io/python/ops` and
# it is possible to run tests with `pytest`, e.g.:
scl enable rh-python36 devtoolset-9 \
    'python3 -m pip install pytest'

TFIO_DATAPATH=bazel-bin \
  scl enable rh-python36 devtoolset-9 \
    'python3 -m pytest -s -v tests/test_serialization.py'
```

#### Docker

对于 Python 开发，[此处](tools/docker/devel.Dockerfile)的参考 Dockerfile 可用于从源代码构建 TensorFlow I/O 软件包 (`tensorflow-io`)。此外，还可以使用预先构建的 devel 映像：

```sh
# Pull (if necessary) and start the devel container
$ docker run -it --rm --name tfio-dev --net=host -v ${PWD}:/v -w /v tfsigio/tfio:latest-devel bash

# Inside the docker container, ./configure.sh will install TensorFlow or use existing install
(tfio-dev) root@docker-desktop:/v$ ./configure.sh

# Clean up exisiting bazel build's (if any)
(tfio-dev) root@docker-desktop:/v$ rm -rf bazel-*

# Build TensorFlow I/O C++. For compilation optimization flags, the default (-march=native)
# optimizes the generated code for your machine's CPU type.
# Reference: https://www.tensorflow.orginstall/source#configuration_options).

# NOTE: Based on the available resources, please change the number of job workers to:
# -j 4/8/16 to prevent bazel server terminations and resource oriented build errors.

(tfio-dev) root@docker-desktop:/v$ bazel build -j 8 --copt=-msse4.2 --copt=-mavx --compilation_mode=opt --verbose_failures --test_output=errors --crosstool_top=//third_party/toolchains/gcc7_manylinux2010:toolchain //tensorflow_io/... //tensorflow_io_gcs_filesystem/...


# Run tests with PyTest, note: some tests require launching additional containers to run (see below)
(tfio-dev) root@docker-desktop:/v$ pytest -s -v tests/
# Build the TensorFlow I/O package
(tfio-dev) root@docker-desktop:/v$ python setup.py bdist_wheel
```

构建成功后将生成一个软件包文件 `dist/tensorflow_io-*.whl`。

注：在 Python 开发容器中工作时，环境变量 `TFIO_DATAPATH` 会自动设为将 tensorflow-io 指向 Bazel 构建的共享 C++ 库，以运行 `pytest` 并构建 `bdist_wheel`。Python `setup.py` 也可以接受 `--data [path]` 作为参数，例如 `python setup.py --data bazel-bin bdist_wheel`。

注：虽然 tfio-dev 容器为开发者提供了一个易于使用的环境，但由于 linux2010 要求众多，已发布的 whl 软件包的构建方式有所不同。有关如何生成已发布的 whl 软件包的更多详细信息，请查看 [构建状态和 CI] 部分。

#### Python Wheel

在 bazel 构建完成后，可以使用以下命令构建 Python Wheel：

```
$ python setup.py bdist_wheel --data bazel-bin
```

.whl 文件将在 dist 目录中提供。请注意，bazel 二进制目录 `bazel-bin` 必须与 `--data` 参数一起传递，这样 setup.py 才能找到必要的共享对象，因为 `bazel-bin` 不在 `tensorflow_io` 软件包目录内。

或者，可以使用以下代码完成源安装：

```
$ TFIO_DATAPATH=bazel-bin python -m pip install .
```

同时传递 `TFIO_DATAPATH=bazel-bin`，原因同上。

请注意，使用 `-e` 进行安装与上述不同。

```
$ TFIO_DATAPATH=bazel-bin python -m pip install -e .
```

上述代码不会自动安装共享对象，即使使用 `TFIO_DATAPATH=bazel-bin` 也是如此。相反，在安装后每次运行程序时，都必须传递 `TFIO_DATAPATH=bazel-bin`：

```
$ TFIO_DATAPATH=bazel-bin python

>>> import tensorflow_io as tfio
>>> ...
```

#### 测试

某些测试需要在运行之前启动测试容器或启动关联工具的本地实例。例如，要运行将启动 kafka、zookeeper 和 schema-registry 的本地实例的 kafka 相关测试，请使用：

```sh
# Start the local instances of kafka, zookeeper and schema-registry
$ bash -x -e tests/test_kafka/kafka_test.sh

# Run the tests
$ TFIO_DATAPATH=bazel-bin pytest -s -vv tests/test_kafka.py
```

要测试与 `Elasticsearch` 或 `MongoDB` 等工具关联的 `Datasets`，系统上需要提供 docker。在这种情况下，请使用：

```sh
# Start elasticsearch within docker container
$ bash tests/test_elasticsearch/elasticsearch_test.sh start

# Run the tests
$ TFIO_DATAPATH=bazel-bin pytest -s -vv tests/test_elasticsearch.py

# Stop and remove the container
$ bash tests/test_elasticsearch/elasticsearch_test.sh stop
```

此外，测试 `tensorflow-io` 的某些功能不需要您启动任何其他工具，因为 `tests` 目录本身中已提供该数据。例如，要运行与 `parquet` 数据集相关的测试，请使用：

```sh
# Just run the test
$ TFIO_DATAPATH=bazel-bin pytest -s -vv tests/test_parquet.py
```

### R

我们在[此处](R-package/scripts/Dockerfile)为您提供参考 Dockerfile，以便您可以直接使用 R 软件包进行测试。可通过以下方式构建该软件包：

```sh
$ docker build -t tfio-r-dev -f R-package/scripts/Dockerfile .
```

在容器内，您可以启动您的 R 会话，从示例 <a>Hadoop SequenceFile</a> [string.seq](https://wiki.apache.org/hadoop/SequenceFile) 实例化 <code>SequenceFileDataset</code>，然后对数据集使用 [tfdatasets 软件包](https://tensorflow.rstudio.com/tools/tfdatasets/articles/introduction.html#transformations)提供的任何[转换函数](https://tensorflow.rstudio.com/tools/tfdatasets/)，如下所示：

```r
library(tfio)
dataset <- sequence_file_dataset("R-package/tests/testthat/testdata/string.seq") %>%
    dataset_repeat(2)

sess <- tf$Session()
iterator <- make_iterator_one_shot(dataset)
next_batch <- iterator_get_next(iterator)

until_out_of_range({
  batch <- sess$run(next_batch)
  print(batch)
})
```
