# TensorFlow Graphics をインストールする

## 安定ビルド

TensorFlow Graphics には [TensorFlow](https://www.tensorflow.org/install) 1.13.1 以上が必要です。TensorFlow のナイトリービルド（tf-nightly）もサポートされています。

[PyPI](https://pypi.org/project/tensorflow-graphics/) から最新の CPU バージョンをインストールするには、次のコードを実行します。

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --upgrade tensorflow-graphics
```

最新の GPU バージョンをインストールするには、次のコードを実行します。

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --upgrade tensorflow-graphics-gpu
```

インストールに関するヘルプや、前提要件のインストールと仮想環境のセットアップ（オプション）に関する案内については、[TensorFlow インストールガイド](https://www.tensorflow.org/install)をご覧ください。

## ソースからインストールする（macOS/Linux）

次のコマンドを実行して、ソースからインストールすることも可能です。

```shell
git clone https://github.com/tensorflow/graphics.git
sh build_pip_pkg.sh
pip install --upgrade dist/*.whl
```

## オプションパッケージのインストール（Linux）

TensorFlow Graphics EXR データローダーを使用するには、OpenEXR がインストールされている必要があります。このインストールは次のコマンドを実行して行います。

```
sudo apt-get install libopenexr-dev
pip install --upgrade OpenEXR
```
