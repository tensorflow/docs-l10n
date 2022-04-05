<!--* freshness: { owner: 'akhorlin' reviewed: '2022-03-19' } *-->

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

# Linux を使用して TensorFlow Hub の pip パケージを作成する

注意: このドキュメントは TensorFlow Hub 自体の変更に関心がある開発者を対象としています。TensorFlow Hub を*使用*するには、[インストール手順](installation.md)をご覧ください。

TensorFlow Hub の pip パッケージを変更すると、ほとんどの場合、ソースから pip パッケージを再構築して変更内容を試すことになるでしょう。

それを行う場合は、次が必要となります。

- Python
- TensorFlow
- Git
- [Bazel](https://docs.bazel.build/versions/master/install.html)

または、protobuf コンパイラをインストールする場合は、[bazel を使用せずに変更を試す](#develop)ことができます。

## virtualenv をセットアップする {:#setup}

### virtualenv の有効化

すでにインストール済みでない場合は、virtualenv をインストールします。

```shell
~$ sudo apt-get install python-virtualenv
```

パッケージ作成用の仮想環境を作成します。

```shell
~$ virtualenv --system-site-packages tensorflow_hub_env
```

その環境を有効化します。

```shell
~$ source ~/tensorflow_hub_env/bin/activate  # bash, sh, ksh, or zsh
~$ source ~/tensorflow_hub_env/bin/activate.csh  # csh or tcsh
```

### TensorFlow Hub リポジトリの複製

```shell
(tensorflow_hub_env)~/$ git clone https://github.com/tensorflow/hub
(tensorflow_hub_env)~/$ cd hub
```

## 変更をテストする

### TensorFlow Hub のテストの実行

```shell
(tensorflow_hub_env)~/hub/$ bazel test tensorflow_hub:all
```

## パッケージを構築してインストールする

### TensorFlow Hub の pip パッケージ作成スクリプトの構築

TensorFlow Hub の pip パッケージを構築するには、次のコードを実行します。

```shell
(tensorflow_hub_env)~/hub/$ bazel build tensorflow_hub/pip_package:build_pip_package
```

### TensorFlow Hub の pip パッケージの作成

```shell
(tensorflow_hub_env)~/hub/$ bazel-bin/tensorflow_hub/pip_package/build_pip_package \
/tmp/tensorflow_hub_pkg
```

### pip パッケージのインストールとテスト（オプション）

次のコマンドを実行して、pip パッケージをインストールします。

```shell
(tensorflow_hub_env)~/hub/$ pip install /tmp/tensorflow_hub_pkg/*.whl
```

TensorFlow Hub のインポートのテスト:

```shell
(tensorflow_hub_env)~/hub/$ cd ..  # exit the directory to avoid confusion
(tensorflow_hub_env)~/$ python -c "import tensorflow_hub as hub"
```

## "Developer" インストール（実験的）

<a id="develop"></a>

警告: TensorFlow を実行するための次のアプローチは実験的であり、TensorFlow Hub チームが正式にサポートするものではありません。

bazel を使用してパッケージを構築する方法が唯一サポートされている手法です。ただし、bazel に不慣れな場合は、オープンソースツールを使用することもできます。これには、パッケージの「developer install」を実行することができます。

このインストール方法では、作業ディレクトリを Python 環境にインストールできるため、継続的な変更は、パッケージをインポートする際に反映されます。

### リポジトリのセットアップ

まず、[上述](#setup)のとおり、virtualenv とリポジトリをセットアップします。

### `protoc` のインストール

TensorFlow Hub は protobufs を使用するため、`.proto` ファイルから必要な Python `_pb2.py` ファイルを作成するには、protobuf コンパイラが必要となります。

#### Mac:

```
(tensorflow_hub_env)~/hub/$ brew install protobuf
```

#### Linux

```
(tensorflow_hub_env)~/hub/$ sudo apt install protobuf-compiler
```

### `.proto` ファイルのコンパイル

最初は、ディレクトリに `_pb2.py` ファイルは存在しません。

```
(tensorflow_hub_env)~/hub/$ ls -1 tensorflow_hub/*_pb2.py
```

`protoc` を実行して、ファイルを作成します。

```
(tensorflow_hub_env)~/hub/$ protoc -I=tensorflow_hub --python_out=tensorflow_hub tensorflow_hub/*.proto
(tensorflow_hub_env)~/hub/$ ls -1 tensorflow_hub/*_pb2.py
```

<pre>tensorflow_hub/image_module_info_pb2.py
tensorflow_hub/module_attachment_pb2.py
tensorflow_hub/module_def_pb2.py
</pre>

注意: `.proto` の定義を変更した場合は、忘れずに `_pb2.py` ファイルをリコンパイルしてください。

### リポジトリから直接インポートする

`_pb2.py` ファイルが配置されたら、TensorFlow Hub ディレクトリから直接変更を使用して試すことができます。

```
(tensorflow_hub_env)~/$ python -c "import tensorflow_hub as hub"
```

### "developer" モードでのインストール

リポジトリルートの外部からファイルを使用するには、`setup.py develop` インストールを使用できます。

```
(tensorflow_hub_env)~/hub/$ python tensorflow_hub/pip_package/setup.py develop
```

これで、通常の Python 仮想環境のローカル変更を使用できるようになりました。変更するたびに、pip パッケージを再構築してインストールする必要はありません。

```shell
(tensorflow_hub_env)~/hub/$ cd ..  # exit the directory to avoid confusion
(tensorflow_hub_env)~/$ python -c "import tensorflow_hub as hub"
```

## virtualenv を無効化する

```shell
(tensorflow_hub_env)~/hub/$ deactivate
```
