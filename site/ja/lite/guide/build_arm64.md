# ARM64 ボード用の TensorFlow Lite を構築する

このページでは、ARM64 ベースのコンピュータ用の TensorFlow Lite 静的ライブラリと共有ライブラリを構築する方法について説明します。モデルを実行するためのみに TensorFlow Lite を使用する場合は、[Python クイックスタート](python.md)に示されているように、最も迅速なオプションとして TensorFlow Lite ランタイムパッケージをインストールします。

注: このページでは、TensorFlow Lite の C ++静的ライブラリと共有ライブラリのみをコンパイルする方法を示します。次のインストールオプションがあります。[Python インタープリター API のみをインストール](python.md)する（推論のみ）。 [pip から TensorFlow パッケージ全体をインストールする](https://www.tensorflow.org/install/pip)。または[完全な TensorFlow パッケージを構築する](https://www.tensorflow.org/install/source)。

## Make を使用した ARM64 のクロスコンパイル

適切なビルド環境を確保するには、[tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/) などの TensorFlow Docker イメージの 1 つを使用することをお勧めします。

はじめるには、まず、ツールチェーンとライブラリをインストールします。

```bash
sudo apt-get update
sudo apt-get install crossbuild-essential-arm64
```

Docker を使用する場合は`sudo`は使用できません。

次に、TensorFlow リポジトリ(`https://github.com/tensorflow/tensorflow`)を git クローンします。TensorFlow Docker イメージを使用している場合、リポジトリはすでに`/tensorflow_src/`で提供されています。その後、TensorFlow リポジトリのルートでこのスクリプトを実行して、すべてのビルドの依存関係をダウンロードします。

```bash
./tensorflow/lite/tools/make/download_dependencies.sh
```

これは一度だけ行う必要があることに注意してください。

次にコンパイルします。

```bash
./tensorflow/lite/tools/make/build_aarch64_lib.sh
```

静的ライブラリが次にコンパイルされます。`tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a`。

## ARM64 でネイティブにコンパイルする

これらの手順は、HardKernel Odroid C2、gcc バージョン 5.4.0 でテストされています。

ボードにログインして、ツールチェーンをインストールします。

```bash
sudo apt-get install build-essential
```

次に、TensorFlow リポジトリ(`https://github.com/tensorflow/tensorflow`) を git クローンし、リポジトリのルートでこれを実行します。

```bash
./tensorflow/lite/tools/make/download_dependencies.sh
```

これは一度だけ行う必要があることに注意してください。

次にコンパイルします。

```bash
./tensorflow/lite/tools/make/build_aarch64_lib.sh
```

静的ライブラリが次にコンパイルされます。`tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a`。

## Bazel を使用した ARM64 のクロスコンパイル

Bazel で[ ARM GCC ツールチェーン](https://github.com/tensorflow/tensorflow/tree/master/third_party/toolchains/embedded/arm-linux)を使用して、ARM64 共有ライブラリを構築します。

注: 生成された共有ライブラリを実行するには、glibc 2.28 以降が必要です。

次の手順は、Ubuntu 16.04.3 64 ビット PC (AMD64) および TensorFlow devel Docker イメージ [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/) でテストされています。

TensorFlow Lite を Bazel とクロスコンパイルするには、次の手順に従います。

#### ステップ 1. Bazel をインストールする

Bazel は TensorFlow の主要なビルドシステムです。[Bazel ビルドシステム](https://bazel.build/versions/master/docs/install.html)の最新バージョンをインストールします。

**注:** TensorFlow Docker イメージを使用している場合、Bazel はすでに利用可能です。

#### ステップ 2. TensorFlow レポジトリをクローンする

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**注:** TensorFlow Docker イメージを使用している場合、リポジトリは`/tensorflow_src/`ですでに提供されています。

#### ステップ 3. ARM64 バイナリを構築する

##### C ライブラリ

```bash
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/c:libtensorflowlite_c.so
```

詳細については [TensorFlow Lite C API](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/c) ページを参照してください。

##### C++ ライブラリ

```bash
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite:libtensorflowlite.so
```

共有ライブラリは次からご覧いただけます。`bazel-bin/tensorflow/lite/libtensorflowlite.so`

現在、必要なすべてのヘッダーファイルを抽出する簡単な方法はないため、TensorFlow リポジトリから tensorflow/lite/ にすべてのヘッダーファイルを含める必要があります。さらに、FlatBuffers および Abseil からのヘッダーファイルが必要になります。

##### その他

ツールチェーンを使用して他の Bazel ターゲットを構築することもできます。以下は有用なターゲットです。

- //tensorflow/lite/tools/benchmark:benchmark_model
- //tensorflow/lite/examples/label_image:label_image
