# ARM ボード用の TesorFlow Lite を構築する

このページでは、ARM を基盤としたコンピュータ向けに TensorFlow Lite ライブラリを構築する方法を説明します。

TensorFlow Lite では 2 つのビルドシステムがサポートされていますが、それぞれのビルドシステムがサポートする機能は同一ではありません。次の表を参考に、適切なビルドシステムを選んでください。

機能 | Bazel | CMake
--- | --- | ---
事前定義済みツールチェーン | armhf、aarch64 | armel、armhf、aarch64
カスタムツールチェーン | 比較的使いにくい | 使いやすい
[セレクト TF 演算](https://www.tensorflow.org/lite/guide/ops_select) | サポート対象 | サポート対象外
[GPU デリゲート](https://www.tensorflow.org/lite/performance/gpu) | Android のみ | OpenCL をサポートするすべてのプラットフォーム
XNNPack | サポート対象 | サポート対象
[Python Wheel](https://www.tensorflow.org/lite/guide/build_cmake_pip) | サポート対象 | サポート対象
[C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/README.md) | サポート対象 | [サポート対象](https://www.tensorflow.org/lite/guide/build_cmake#build_tensorflow_lite_c_library)
[C++ API](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c) | Bazel プロジェクトでサポート | CMake プロジェクトでサポート

## CMake を使用した ARM のクロスコンパイル

CMake プロジェクトがある場合、またはカスタムツールチェーンを使用する場合は、CMake を使ってクロスコンパイルをじっこうすることが推奨されます。これを行うための「[CMake を使用した TensorFlow Lite クロスコンパイル](https://www.tensorflow.org/lite/guide/build_cmake_arm)」というページが別途用意されています。

## Bazel を使用した ARM のクロスコンパイル

Bazel プロジェクトがある場合、または TF 演算を使用する場合は、Bazel ビルドシステムの使用が推奨されます。ARM32/64 共有ライブラリを構築するには、統合されている [ARM GCC 8.3 ツールチェーン](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/toolchains/embedded/arm-linux)を Bazel を使用します。

ターゲットアーキテクチャ | Bazel 構成 | 対応デバイス
--- | --- | ---
armhf（ARM32） | --config=elinux_armhf | RPI3、32ビットの RPI4
:                     :                         : Raspberry Pi OS            : |  |
AArch64（ARM64） | --config=elinux_aarch64 | Coral、Ubuntu 64ビット の RPI4
:                     :                         : bit                        : |  |

注: 生成された共有ライブラリを実行するには、glibc 2.28 以降が必要です。

次の手順は、Ubuntu 16.04.3 64 ビット PC（AMD64）および TensorFlow devel Docker イメージ [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/) でテストされています。

TensorFlow Lite を Bazel とクロスコンパイルするには、次の手順に従います。

#### 手順 1. Bazel をインストールする

Bazel は TensorFlow の主なビルドシステムです。[Bazel ビルドシステム](https://bazel.build/versions/master/docs/install.html)の最新バージョンをインストールします。

**注意:** TensorFlow Docker イメージを使用している場合、Bazel はすでに利用可能です。

#### 手順 2. TensorFlow レポジトリをクローンする

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**注意:** TensorFlow Docker イメージを使用している場合、リポジトリは `/tensorflow_src/` にあります。

#### 手順 3. ARM バイナリを構築する

##### C ライブラリ

```bash
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/c:libtensorflowlite_c.so
```

共有ライブラリは、`bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so` にあります。

**注意:** `elinux_armhf` を [32bit ARM ハードフロート](https://wiki.debian.org/ArmHardFloatPort)ビルドに使用します。

詳細については、[TensorFlow Lite C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/README.md) ページをご覧ください。

##### C++ ライブラリ

```bash
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite:libtensorflowlite.so
```

共有ライブラリは、`bazel-bin/tensorflow/lite/libtensorflowlite.so` にあります。

現在、必要なすべてのヘッダーファイルを抽出する簡単な方法はないため、TensorFlow リポジトリから tensorflow/lite/ にすべてのヘッダーファイルを含める必要があります。さらに、FlatBuffers および Abseil からのヘッダーファイルが必要になります。

##### その他

ツールチェーンを使用してほかの Bazel ターゲットを構築することもできます。 以下は有用なターゲットです。

- //tensorflow/lite/tools/benchmark:benchmark_model
- //tensorflow/lite/examples/label_image:label_image
