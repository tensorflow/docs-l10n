# TensorFlow モデル最適化をインストールする

インストールに進む前に、Python 仮想環境を作成することをお勧めします。詳細は、TnesorFlow インストール[ガイド](https://www.tensorflow.org/install/pip#2.-create-a-virtual-environment-recommended)をご覧ください。

### 安定ビルド

最新バージョンをインストールするには、次のように実行します。

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version. pip install --user --upgrade tensorflow-model-optimization
```

リリースの詳細については、[リリースノート](https://github.com/tensorflow/model-optimization/releases)をご覧ください。

必要なバージョンの TensorFlow とその他の互換性に関する情報については、使用するテクニックの概要ページにある「API 互換性マトリクス」セクションをご覧ください。たとえば、プルーニングの場合の概要ページは[こちら](https://www.tensorflow.org/model_optimization/guide/pruning)にあります。

TensorFlow は、TensorFlow モデル最適化パッケージの依存ファイルとして（`setup.py` に）*含まれていない*ため、TensorFlow パッケージ（`tf-nightly` または `tf-nightly-gpu`）を明示的にインストールする必要があります。こうすることで、CPU と GPU 対応の TensorFlow パッケージを個別にではなく、まとめて管理できるようになります。

### ソースからインストールする

ソースからインストールすることも可能です。これには [Bazel](https://bazel.build/) 構築システムが必要です。

```shell
# To install dependencies on Ubuntu:
# sudo apt-get install bazel git python-pip
# For other platforms, see Bazel docs above.
git clone https://github.com/tensorflow/model-optimization.git
cd model-optimization
bazel build --copt=-O3 --copt=-march=native :pip_pkg
PKGDIR=$(mktemp -d)
./bazel-bin/pip_pkg $PKGDIR
pip install --user --upgrade $PKGDIR/*.whl
```
