## 開発

このドキュメントには、開発環境をセットアップし、さまざまなプラットフォームのソースから `tensorflow-io` パッケージを構築するために必要な情報が含まれています。セットアップが完了したら、新しい ops を追加するためのガイドラインについて [STYLE_GUIDE](https://github.com/tensorflow/io/blob/master/STYLE_GUIDE.md) を参照してください。

### IDE のセットアップ

TensorFlow I/O を開発するために Visual Studio Code を構成する方法については、この[ドキュメント](https://github.com/tensorflow/io/blob/master/docs/vscode.md)を参照してください。

### Lint

TensorFlow I/O のコードは、Bazel Buildifier、Clang Format、Black、および Pyupgrade に準拠しています。次のコマンドを使用して、ソースコードを確認し、Lintの問題を特定してください。

```
# Install Bazelisk (manage bazel version implicitly)
$ curl -sSOL https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
$ sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
$ sudo chmod +x /usr/local/bin/bazel
$ bazel run //tools/lint:check
```

Bazel Buildifier と Clang Format の場合、次のコマンドは Lint エラーを自動的に識別して修正します。

```
$ bazel run //tools/lint:lint
```

または、個々の Linter を使用して Lint チェックのみを実行する場合は、上記のコマンドに `black`、`pyupgrade`、`bazel`、または `clang` を選択的に渡すことができます。

たとえば、`black` の特定の Lint チェックは次を使用して実行できます。

```
$ bazel run //tools/lint:check -- black
```

Bazel Buildifier と Clang Format を使用した Lint 修正は、次を使用して実行できます。

```
$ bazel run //tools/lint:lint -- bazel clang
```

個々の Python ファイルの `black` と `pyupgrade` を使用した Lint チェックは、次を使用して実行できます。

```
$ bazel run //tools/lint:check -- black pyupgrade -- tensorflow_io/python/ops/version_ops.py
```

Lint は、以下を使用して black と pyupgrade で個々の Python ファイルを修正します。

```
$ bazel run //tools/lint:lint -- black pyupgrade --  tensorflow_io/python/ops/version_ops.py
```

### Python

#### macOS

macOS Catalina 10.15.7 では、システム提供の python 3.8.2 を使用して tensorflow-io をビルドできます。そのためには `tensorflow` と `bazel` の両方が必要です。

注: macOS 10.15.7 のシステムデフォルトの python 3.8.2 は、コンパイラーオプションの `-arch arm64 -arch x86_64` が原因で `regex` のインストールエラーが発生します（https://github.com/giampaolo/psutil/issues/1832 に記載されている問題と同じです）。この問題を解決するには、arm64 ビルドオプションを削除するために `export ARCHFLAGS="-arch x86_64"` が必要になります。

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

注意: pytest を実行するときは、ビルドプロセス後に Python が生成された共有ライブラリを利用できるように、 `TFIO_DATAPATH=bazel-bin` を渡す必要があります。

##### トラブルシューティング

Xcode がインストールされているが、`$ xcodebuild -version` が期待される出力を表示しない場合は、次のコマンドで Xcode コマンドラインを有効にする必要があります。

`$ xcode-select -s /Applications/Xcode.app/Contents/Developer`.

変更を有効にするには、端末の再起動が必要になる場合があります。

サンプル出力：

```
$ xcodebuild -version
Xcode 12.2
Build version 12B45b
```

#### Linux

Linux での tensorflow-io の開発は、macOS に似ています。必要なパッケージは、gcc、g++、git、bazel、および python 3です。ただし、デフォルトのシステムインストールバージョン以外の新しいバージョンの gcc または python が必要になる場合があります。

##### Ubuntu 20.04

Ubuntu 20.04 には gcc/g++、git、および python 3 が必要です。以下は依存関係をインストールし、Ubuntu 20.04 で共有ライブラリをビルドします。

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

CentOS 8 の共有ライブラリをビルドする手順は、次を除き上記の Ubuntu 20.04 と似ています。

```
sudo yum install -y python3 python3-devel gcc gcc-c++ git unzip which make
```

gcc/g++、git、unzip/which（bazel の場合）、および python3 をインストールするために使用する必要があります。

##### CentOS 7

CentOS 7 では、デフォルトの python と gcc のバージョンが古すぎて、tensorflow-io の共有ライブラリ（.so）をビルドできません。代わりに、Developer Toolset および rh-python36 によって提供される gcc を使用する必要があります。また、CentOS にインストールされた libstdc++ と devtoolset による新しい gcc バージョンの不一致を回避するために、libstdc++ を静的にリンクする必要があります。

さらには、ファイルシステムプラグインの静的にリンクされたライブラリのシンボルの重複を避けるために、特別なフラグ `--//tensorflow_io/core:static_build` を Bazel に渡す必要あります。

以下は、bazel、devtoolset-9、rh-python36 をインストールし、共有ライブラリをビルドします。

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

Python 開発の場合、[ここ](tools/docker/devel.Dockerfile)にある参照 Dockerfile を使用して、ソースから TensorFlow I/O パッケージ（`tensorflow-io`）をビルドできます。さらに、事前に作成された開発イメージも使用できます。

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

ビルドが成功すると、パッケージファイル `dist/tensorflow_io-*.whl` が生成されます。

注意: Python 開発コンテナで作業する場合、環境変数 `TFIO_DATAPATH` は、 `pytest` を実行して `bdist_wheel` をビルドするために Bazel によってビルドされた共有 Bazel ライブラリに tensorflow-io をポイントするように自動的に設定されます。 Python の `setup.py` は、引数として `--data [path]` を受け入れることもできます。たとえば `python setup.py --data bazel-bin bdist_wheel` です。

注意: tfio-dev コンテナを使用すると、開発者は環境を簡単に操作できますが、リリースされた whl パッケージは、 manylinux2010 <br> の要件のために異なる方法でビルドされます。リリースされた whl パッケージの生成方法の詳細については、[ビルドステータスと CI]セクションを確認してください。

#### Python Wheels

次のコマンドで bazel のビルドが完了した後に、python wheel をビルドできます。

```
$ python setup.py bdist_wheel --data bazel-bin
```

.whl ファイルが dist ディレクトリで利用可能になります。 `bazel-bin` は `tensorflow_io` パッケージディレクトリの外部にあるため、setup.py が必要な共有オブジェクトを見つけるには、bazel バイナリディレクトリ `bazel-bin` を `--data` とともに渡す必要があることに注意してください。

または、ソースのインストールを次の方法で実行できます。

```
$ TFIO_DATAPATH=bazel-bin python -m pip install .
```

同じ理由で `TFIO_DATAPATH=bazel-bin` が渡されました。

`-e` を使用したインストールは、上記とは異なることに注意してください。

```
$ TFIO_DATAPATH=bazel-bin python -m pip install -e .
```

は、`TFIO_DATAPATH=bazel-bin` を使用しても、共有オブジェクトを自動的にインストールしません。代わりに、インストール後にプログラムを実行するたびに、 `TFIO_DATAPATH=bazel-bin` を渡す必要があります。

```
$ TFIO_DATAPATH=bazel-bin python

>>> import tensorflow_io as tfio
>>> ...
```

#### テスト

一部のテストでは、実行する前に、テストコンテナを起動するか、関連するツールのローカルインスタンスを起動する必要があります。たとえば、kafka、zookeeper、schema-registry のローカルインスタンスを開始する kafka 関連のテストを実行するには、次を使用します。

```sh
# Start the local instances of kafka, zookeeper and schema-registry
$ bash -x -e tests/test_kafka/kafka_test.sh

# Run the tests
$ TFIO_DATAPATH=bazel-bin pytest -s -vv tests/test_kafka.py
```

`Elasticsearch` や `MongoDB` などのツールに関連付けられた `Datasets` をテストするには、システムで docker が使用可能である必要があります。このようなシナリオでは、次を使用します。

```sh
# Start elasticsearch within docker container
$ bash tests/test_elasticsearch/elasticsearch_test.sh start

# Run the tests
$ TFIO_DATAPATH=bazel-bin pytest -s -vv tests/test_elasticsearch.py

# Stop and remove the container
$ bash tests/test_elasticsearch/elasticsearch_test.sh stop
```

さらに、 `tensorflow-io` の一部の機能をテストする場合、データは `tests` ディレクトリ自体に提供されているため、追加のツールを起動する必要はありません。たとえば、 `parquet` のデータセットに関連するテストを実行するには、次を使用します。

```sh
# Just run the test
$ TFIO_DATAPATH=bazel-bin pytest -s -vv tests/test_parquet.py
```

### R

R パッケージを直接テストに使用できるように、[ここ](R-package/scripts/Dockerfile)に参照 Dockerfile を提供します。次の方法でビルドできます。

```sh
$ docker build -t tfio-r-dev -f R-package/scripts/Dockerfile .
```

コンテナ内で R セッションを開始し、例の [Hadoop SequenceFile](https://wiki.apache.org/hadoop/SequenceFile) [string.seq](R-package/tests/testthat/testdata/string.seq) から `SequenceFileDataset` をインスタンス化してから、次のように [tfdatasets パッケージ](https://tensorflow.rstudio.com/tools/tfdatasets/) によって提供される[変換関数](https://tensorflow.rstudio.com/tools/tfdatasets/)をデータセットで使用できます。

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
