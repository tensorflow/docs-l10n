# TensorFlow Lite Python Wheel パッケージの構築

このページでは、x86_64 およびさまざまな ARM デバイスの TensorFlow Lite `tflite_runtime` Python ライブラリを構築する方法について説明します。

次の手順は、Ubuntu 16.04.3 64-bit PC (AMD64)、macOS Catalina (x86_64)、および TensorFlow devel Docker イメージ [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/) でテストされています。

**注意:** この機能はバージョン 2.4 以降で利用できます。

#### 前提条件

CMake をインストールし、TensorFlow ソースコードをコピーする必要があります。詳細については、[CMake を使用した TensorFlow Lite の構築](https://www.tensorflow.org/lite/guide/build_cmake)ページを参照してください。

ワークステーション用の PIP パッケージを構築するには、次のコマンドを実行できます。

```sh
PYTHON=python3 tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh native
```

**注意:** 複数の Python インタープリタがある場合は、`PYTHON` 変数を使用して、正確な Python バージョンを指定します。(現在は Python 3.7 以降がサポートされています。)

## ARM クロスコンパイル

ARM クロスコンパイルでは、Docker を使用することをお勧めします。この方法では、クロスビルド環境の設定が簡単になります。また、`target` オプションを使用して、ターゲットアーキテクチャを指定する必要があります。

Makefile `tensorflow/lite/tools/pip_package/Makefile` にはヘルパーツールがあり、定義済みの Docker コンテナを使用して、ビルドコマンドを実行できます。Docker ホストコンピュータでは、次のようにビルドコマンドを実行できます。

```sh
make -C tensorflow/lite/tools/pip_package docker-build \
  TENSORFLOW_TARGET=<target> PYTHON_VERSION=<python3 version>
```

**注意:** Python バージョン 3.7 以降がサポートされています。

### 使用可能なターゲット名

`tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh` スクリプトでは、ターゲットアーキテクチャを識別するために、ターゲット名が必要です。サポートされているターゲットの一覧は次のとおりです。

ターゲット | ターゲットアーキテクチャ | コメント
--- | --- | ---
armhf | ARMv7 VFP と Neon | Raspberry Pi 3 および 4 に対応
rpi0 | ARMv6 | Raspberry Pi Zero に対応
aarch64 | aarch64 (ARM 64-bit) | [Coral Mendel Linux 4.0](https://coral.ai/) <br> Raspberry Pi と [Ubuntu Server 20.04.01 LTS 64-bit](https://ubuntu.com/download/raspberry-pi)
native | ワークステーション | "-mnative" 最適化で構築
<default></default> | ワークステーション | 既定のターゲット

### 構築の例

次に、使用できるコマンドの例をいくつか示します。

#### Python 3.7 の armhf ターゲット

```sh
make -C tensorflow/lite/tools/pip_package docker-build \
  TENSORFLOW_TARGET=armhf PYTHON_VERSION=3.7
```

#### Python 3.8 の aarch64 ターゲット

```sh
make -C tensorflow/lite/tools/pip_package docker-build \
  TENSORFLOW_TARGET=aarch64 PYTHON_VERSION=3.8
```

#### カスタムツールチェーンを使用する方法

生成されたバイナリがターゲットと互換性がない場合は、独自のツールチェーンを使用するか、ビルドフラグを指定する必要があります。(ターゲット環境を確認するには、[こちら](https://www.tensorflow.org/lite/guide/build_cmake_arm#check_your_target_environment)を参照してください。) この場合、独自のツールチェーンを使用するには、`tensorflow/lite/tools/cmake/download_toolchains.sh` を修正する必要があります。ツールチェーンスクリプトでは、`build_pip_package_with_cmake.sh` スクリプトの次の 2 つの変数が定義されます。

変数 | 目的 | 例
--- | --- | ---
ARMCC_PREFIX | ツールチェーンプレフィックスの定義 | arm-linux-gnueabihf-
ARMCC_FLAGS | コンパイルフラグ | -march=armv7-a -mfpu=neon-vfpv4

**注意:** ARMCC_FLAGS には Python ライブラリパスを含める必要があります。リファレンスについては、`download_toolchains.sh` を参照してください。
