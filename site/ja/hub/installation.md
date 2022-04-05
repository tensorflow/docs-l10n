<!--* freshness: { owner: 'wgierke' reviewed: '2022-01-05' } *-->

# インストール

## tensorflow_hub をインストールする

`tensorflow_hub` ライブラリは、TensorFlow 1 と TensorFlow 2 と併せてインストールできます。TensorFlow 2 から使用し始める新規ユーザーと現行ユーザーはこのライブラリにアップグレードすることをお勧めします。

### TensorFlow 2 との使用

[pip](https://pip.pypa.io/) を使用して、通常どおり [TensorFlow 2 をインストール](https://www.tensorflow.org/install)します（リンク先で GPU サポートに関する追加手順をご覧ください）。その後で、[`tensorflow-hub`](https://pypi.org/project/tensorflow-hub/)（0.5.0 以降である必要があります） の横にその現在のバージョンをインストールします。

```bash
$ pip install "tensorflow>=2.0.0"
$ pip install --upgrade tensorflow-hub
```

TensorFlow Hub の TF1 スタイルの API は、TensorFlow 2 の v1 互換モードで動作します。

### TensorFlow 1 とのレガシー使用

TensorFlow 1.15 は、`tensorflow_hub` ライブラリ（リリース 0.11.0 時点）がサポートする TensorFlow 1.x の唯一のバージョンです。TensorFlow 1.15 はデフォルトでは TF1 対応で動作しますが、内部には多数の TF2 機能を備えているため、TensorFlow Hub の TF2 式 API を一部使用できます。

```bash
$ pip install "tensorflow>=1.15,<2.0"
$ pip install --upgrade tensorflow-hub
```

### プレリリースバージョンの使用

pip パッケージの `tf-nightly` と `tf-hub-nightly` は、github のソースコードからテストを行わずに自動的に構築されるビルドです。このため、開発者は[ソースから構築](build_from_source.md)することなく、最新のコードを試すことができます。

```bash
$ pip install tf-nightly
$ pip install --upgrade tf-hub-nightly
```

## 次のステップ

- [Library overview](lib_overview.md)
- チュートリアル:
    - [テキスト分類](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb)
    - [画像分類](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb)
    - [GitHub](https://github.com/tensorflow/hub/blob/master/examples/README.md) のその他の例
- [tfhub.dev](https://tfhub.dev) でモデルを検索してください
