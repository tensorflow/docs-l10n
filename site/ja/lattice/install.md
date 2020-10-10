# TensorFlow Lattice のインストール

TensorFlow Lattice (TFL) の使用環境のセットアップには、いくつかの方法があります。

- 最も簡単な方法で TFL を学習および使用する場合には、インストールの必要はありません。任意のチュートリアル（例: [Canned Estimator チュートリアル](tutorials/canned_estimators.ipynb)）を実行します。
- TFL をローカルマシンで使用する場合には、`tensorflow-lattice` pip パッケージをインストールします。
- 固有のマシン構成を使用する場合には、パッケージをソースから構築することができます。

## pip を使用して TensorFlow Lattice をインストールする

pip を使用してインストールします。

```shell
pip install --upgrade tensorflow-lattice
```

## ソースから構築する

GitHub のリポジトリをクローンする:

```shell
git clone https://github.com/tensorflow/lattice.git
```

pip パッケージをソースから構築する:

```shell
python setup.py sdist bdist_wheel --universal --release
```

パッケージをインストールする:

```shell
pip install --user --upgrade /path/to/pkg.whl
```
