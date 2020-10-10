# Raspberry Pi 用の TensorFlow Lite を構築する

このページでは、Raspberry Pi 用の TensorFlow Lite 静的ライブラリと共有ライブラリを構築する方法について説明します。モデルを実行するためのみに TensorFlow Lite を使用する場合は、[Python クイックスタート](python.md)に示されているように、最も迅速なオプションとして TensorFlow Lite ランタイムパッケージをインストールします。

注: このページでは、TensorFlow Lite の C ++静的ライブラリと共有ライブラリをコンパイルする方法を示します。次のインストールオプションがあります。<a>Python インタープリタ API のみをインストール</a>する（推論するためのみ）。 [pip から TensorFlow パッケージ全体をインストールする](python.md)。または[完全な TensorFlow パッケージを構築する](https://www.tensorflow.org/install/pip)。

**注:** このページでは、32 ビットのビルドのみを扱います。64 ビットのビルドを探している場合は、[ARM64 用のビルド](build_arm64.md)ページを確認してください。

## Make を使用した Raspberry Pi のクロスコンパイル

次の手順は、Ubuntu 16.04.3 64 ビット PC (AMD64) および TensorFlow devel Docker イメージ [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/) でテストされています。

TensorFlow Lite をクロスコンパイルするには、次の手順に従います。

#### ステップ 1. 公式 Raspberry Pi クロスコンパイルツールチェーンをクローンする

```sh
git clone https://github.com/raspberrypi/tools.git rpi_tools
```

#### ステップ 2. TensorFlow レポジトリをクローンする

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**注:** TensorFlow Docker イメージを使用している場合、リポジトリは`/tensorflow_src/`ですでに提供されています。

#### ステップ 3. TensorFlow リポジトリのルートで次のスクリプトを実行してダウンロードする

すべてのビルド依存関係：

```sh
cd tensorflow_src && ./tensorflow/lite/tools/make/download_dependencies.sh
```

**注:** これは一度だけ実行する必要があります。

#### ステップ 4a. Raspberry Pi 2、3、4 用の ARMv7 バイナリを構築する

```sh
PATH=../rpi_tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin:$PATH \
  ./tensorflow/lite/tools/make/build_rpi_lib.sh
```

<strong>注:</strong> 静的ライブラリが次にコンパイルされます。<code>tensorflow/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a</code>

Make の追加オプションまたはターゲット名を`build_rpi_lib.sh`スクリプトに追加できます。これは、TFLite [Makefile](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/make/Makefile) がある Make のラッパーであるためです。可能なオプションは次のとおりです。

```sh
./tensorflow/lite/tools/make/build_rpi_lib.sh clean # clean object files
./tensorflow/lite/tools/make/build_rpi_lib.sh -j 16 # run with 16 jobs to leverage more CPU cores
./tensorflow/lite/tools/make/build_rpi_lib.sh label_image # # build label_image binary
```

#### ステップ 4b. Raspberry Pi Zero 用の ARMv6 バイナリを構築する

```sh
PATH=../rpi_tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin:$PATH \
  ./tensorflow/lite/tools/make/build_rpi_lib.sh TARGET_ARCH=armv6
```

**注:** 静的ライブラリが次にコンパイルされます。`tensorflow/lite/tools/make/gen/rpi_armv6/lib/libtensorflow-lite.a`

## Raspberry Pi でネイティブにコンパイルする

以下の手順は、Raspberry Pi Zero、Raspbian GNU/Linux 10 (buster)、gcc バージョン 8.3.0 (Raspbian 8.3.0-6 + rpi1) でテストされています。

TensorFlow Lite をネイティブにコンパイルするには、次の手順に従います。

#### ステップ 1.  Raspberry Pi にログインして、ツールチェーンをインストールする

```sh
sudo apt-get install build-essential
```

#### ステップ 2. TensorFlow レポジトリをクローンする

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

#### ステップ 3. TensorFlow リポジトリのルートで次のスクリプトを実行してすべてのビルド依存関係をダウンロードする

```sh
cd tensorflow_src && ./tensorflow/lite/tools/make/download_dependencies.sh
```

**注:** これは一度だけ実行する必要があります。

#### ステップ 4. 以下を使用して TensorFlow Lite をコンパイルする

```sh
./tensorflow/lite/tools/make/build_rpi_lib.sh
```

**注:** 静的ライブラリが次にコンパイルされます。`tensorflow/lite/tools/make/gen/lib/rpi_armv6/libtensorflow-lite.a`

## Bazel を使用した  armhf のクロスコンパイル

[ ARM GCC ツールチェーン](https://github.com/tensorflow/tensorflow/tree/master/third_party/toolchains/embedded/arm-linux)を Bazel で使用して、Raspberry Pi 2、3、4 と互換性のある armhf 共有ライブラリを構築できます。

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

#### ステップ 3. Raspberry Pi 2、3、4 用の ARMv7 バイナリを構築する

##### C ライブラリ

```bash
bazel build --config=elinux_armhf -c opt //tensorflow/lite/c:libtensorflowlite_c.so
```

詳細については [TensorFlow Lite C API](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/c) ページを参照してください。

##### C++ ライブラリ

```bash
bazel build --config=elinux_armhf -c opt //tensorflow/lite:libtensorflowlite.so
```

共有ライブラリは次からご覧いただけます。`bazel-bin/tensorflow/lite/libtensorflowlite.so`

現在、必要なすべてのヘッダーファイルを抽出する簡単な方法はないため、TensorFlow リポジトリから tensorflow/lite/ にすべてのヘッダーファイルを含める必要があります。さらに、FlatBuffers および Abseil からのヘッダーファイルが必要になります。

##### その他

ツールチェーンを使用して他の Bazel ターゲットを構築することもできます。以下は有用なターゲットです。

- //tensorflow/lite/tools/benchmark:benchmark_model
- //tensorflow/lite/examples/label_image:label_image
