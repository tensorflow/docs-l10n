# CMake を使用した TensorFlow Lite の構築

このページでは、[CMake](https://cmake.org/) を使用して、TensorFlow Lite ライブラリを構築する方法について説明します。

次の手順は、Ubuntu 16.04.3 64-bit PC (AMD64)、macOS Catalina (x86_64)、Windows 10、および TensorFlow devel Docker イメージ [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/) でテストされています。

**注意:** この機能はバージョン 2.4 以降で利用できます。

### 手順 1. CMake ツールをインストールする

CMake 3.16 以上が必要です。Ubuntu では、次のコマンドを実行できます。

```sh
sudo apt-get install cmake
```

[公式の cmake インストールガイド](https://cmake.org/install/)の手順でもインストールできます。

### Step 2. Clone TensorFlow repository

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**Note:** If you're using the TensorFlow Docker image, the repo is already provided in `/tensorflow_src/`.

### 手順 3. CMake ビルドディレクトリを作成する

```sh
mkdir tflite_build
cd tflite_build
```

### 手順 4. CMake ツールと構成を実行する

#### リリースビルド

既定では、最適化されたリリースバイナリが生成されます。ワークステーション向けにビルドする場合は、次のコマンドを実行します。

```sh
cmake ../tensorflow_src/tensorflow/lite
```

#### デバッグビルド

シンボル情報を含むデバッグビルドを生成する必要がある場合は、`-DCMAKE_BUILD_TYPE=Debug` オプションを指定する必要があります。

```sh
cmake ../tensorflow_src/tensorflow/lite -DCMAKE_BUILD_TYPE=Debug
```

#### カーネル単体テストによる構築

カーネルテストを実行するには、'-DTFLITE_KERNEL_TEST=on' フラグを設定する必要があります。単体テストクロスコンパイル仕様については、次のサブセクションで説明します。

```sh
cmake ../tensorflow_src/tensorflow/lite -DTFLITE_KERNEL_TEST=on
```

#### Build installable package

To build an installable package that can be used as a dependency by another CMake project with `find_package(tensorflow-lite CONFIG)`, use the `-DTFLITE_ENABLE_INSTALL=ON` option.

You should ideally also provide your own versions of library dependencies. These will also need to used by the project that depends on TF Lite. You can use the `-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON` and set the `<PackageName>_DIR` variables to point to your library installations.

```sh
cmake ../tensorflow_src/tensorflow/lite -DTFLITE_ENABLE_INSTALL=ON \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON \
  -Dabsl_DIR=<install path>/lib/cmake/absl \
  -DEigen3_DIR=<install path>/share/eigen3/cmake \
  -DFlatbuffers_DIR=<install path>/lib/cmake/flatbuffers \
  -DNEON_2_SSE_DIR=<install path>/lib/cmake/NEON_2_SSE \
  -Dcpuinfo_DIR=<install path>/share/cpuinfo \
  -Druy_DIR=<install path>/lib/cmake/ruy
```

**Note:** Refer to CMake documentation for [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html) to learn more about handling and locating packages.

#### クロスコンパイル

CMake を使用して、ARM64 または Android ターゲットアーキテクチャ向けにバイナリを構築できます。

TF Lite のクロスコンパイルを実行するには、`-DCMAKE_TOOLCHAIN_FILE` フラグで SDK (例: ARM64 SDK または Android では NDK) へのパスを指定する必要があります。

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=<CMakeToolchainFileLoc> ../tensorflow/lite/
```

##### Android クロスコンパイルの仕様

Android クロスコンパイルでは、[Android NDK](https://developer.android.com/ndk) をインストールし、上記の `-DCMAKE_TOOLCHAIN_FILE` フラグで NDK パスを指定する必要があります。また、`-DANDROID_ABI` フラグでターゲット ABI を設定する必要もあります。

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=<NDK path>/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a ../tensorflow_src/tensorflow/lite
```

##### カーネル (単体) テストクロスコンパイルの仕様

単体テストのクロスコンパイルには、ホストアーキテクチャ向けの flatc コンパイラが必要です。この目的のために、CMakeLists が `tensorflow/lite/tools/cmake/native_tools/flatbuffers` にあり、ホストツールチェーンを使用して、事前に CMake で別のビルドディレクトリに flatc コンパイラを構築できます。

```sh
mkdir flatc-native-build && cd flatc-native-build
cmake ../tensorflow_src/tensorflow/lite/tools/cmake/native_tools/flatbuffers
cmake --build .
```

また、*flatc* をカスタムインストール先に**インストール**することもできます (例: CMake ビルドディレクトリではなく、他のネイティブに構築されたツールを含むディレクトリ)。

```sh
cmake -DCMAKE_INSTALL_PREFIX=<native_tools_dir> ../tensorflow_src/tensorflow/lite/tools/cmake/native_tools/flatbuffers
cmake --build .
```

TF Lite クロスコンパイル自体では、ネイティブ *flatc* バイナリを含むディレクトリを参照する追加のパラメータ `-DTFLITE_HOST_TOOLS_DIR=<flatc_dir_path>` と上記の `-DTFLITE_KERNEL_TEST=on` フラグを指定する必要があります。

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=${OE_CMAKE_TOOLCHAIN_FILE} -DTFLITE_KERNEL_TEST=on -DTFLITE_HOST_TOOLS_DIR=<flatc_dir_path> ../tensorflow/lite/
```

##### ターゲットでのクロスコンパイル済みカーネル (単体) テストの実行

単体テストは、別の実行ファイルとして実行するか、CTest ユーティリティを使用して実行できます。CTest については、パラメータ `TFLITE_ENABLE_NNAPI, TFLITE_ENABLE_XNNPACK` または `TFLITE_EXTERNAL_DELEGATE` の少なくともいずれかを TF Lite ビルド用に有効にする必要がある場合は、2 つの異なる**ラベル** (同じ実行ファイルを利用) で結果のテストが生成されます。- *plain* - CPU バックエンドで実行されたテスト - *delegate* - 使用されたデリゲート仕様で追加の実行引数が使用されるテスト

`CTestTestfile.cmake` と `run-tests.cmake` (以下を参照) のいずれも `<build_dir>/kernels` にあります。

CPU バックエンドでの単体テストの実行 (`CTestTestfile.cmake` がターゲットの現在のディレクトリに存在する場合)

```sh
ctest -L plain
```

デリゲートを使用した単体テストの実行の例 (`CTestTestfile.cmake` と `run-tests.cmake` ファイルがターゲットの現在のディレクトリに存在する場合)

```sh
cmake -E env TESTS_ARGUMENTS=--use_nnapi=true\;--nnapi_accelerator_name=vsi-npu ctest -L delegate
cmake -E env TESTS_ARGUMENTS=--use_xnnpack=true ctest -L delegate
cmake -E env TESTS_ARGUMENTS=--external_delegate_path=<PATH> ctest -L delegate
```

この方法で追加のデリゲート関連の実行引数を単体テストに渡すときの**確認済みの制限事項**は、**想定された戻り値が 0** の引数しかサポートされていないということです。それ以外の戻り値はテストエラーとして報告されます。

#### OpenCL GPU デリゲート

ターゲットコンピュータで OpenCL がサポートされている場合は、GPU 能力を活用できる [GPU デリゲート](https://www.tensorflow.org/lite/performance/gpu)を使用できます。

OpenCL GPU デリゲートサポートを構成するには、次のコマンドを実行します。

```sh
cmake ../tensorflow_src/tensorflow/lite -DTFLITE_ENABLE_GPU=ON
```

**注意:** このコマンドはまだ実験段階であり、TensorFlow 2.5 以降で提供されています。互換性の問題が生じる可能性があります。Android デバイスと NVidia CUDA OpenCL 1.2 でのみ検証されています。

### 手順 5. TensorFlow Lite を構築する

tflite_build ディレクトリで次のコマンドを実行します。

```sh
cmake --build . -j
```

**注意:** これにより、静的ライブラリ `libtensorflow-lite.a` が現在のディレクトリに生成されます。ただし、一部の推移的な依存関係が含まれないため、このライブラリは自己完結型ではありません。このライブラリを適切な方法で使用するには、CMake プロジェクトを作成する必要があります。[TensorFlow Lite を使用する CMake プロジェクトの作成](#create_a_cmake_project_which_uses_tensorflow_lite)セクションを参照してください。

### 手順 6. TensorFlow Lite ベンチマークツールとラベル画像例の構築 (任意)

tflite_build ディレクトリで次のコマンドを実行します。

```sh
cmake --build . -j -t benchmark_model
```

```sh
cmake --build . -j -t label_image
```

## TensorFlow Lite の構築で使用可能なオプション

使用可能なオプションは次のとおりです。`-D<option_name>=[ON|OFF]` で上書きできます。たとえば、`-DTFLITE_ENABLE_XNNPACK=OFF` を使用すると、既定で有効な XNNPACK が無効になります。

オプション名 | 機能 | Android | Linux | macOS | Windows
--- | --- | --- | --- | --- | ---
TFLITE_ENABLE_RUY | RUY を有効にする | ON | OFF | OFF | OFF
:                       : matrix         :         :       :       :         : |  |  |  |  |
:                       : multiplication :         :       :       :         : |  |  |  |  |
:                       : library        :         :       :       :         : |  |  |  |  |
TFLITE_ENABLE_NNAPI | NNAPI を有効にする | ON | OFF | N/A | N/A
:                       : delegate       :         :       :       :         : |  |  |  |  |
TFLITE_ENABLE_GPU | GPU を有効にする | OFF | OFF | N/A | N/A
:                       : delegate       :         :       :       :         : |  |  |  |  |
TFLITE_ENABLE_XNNPACK | XNNPACK を有効にする | ON | ON | ON | ON
:                       : delegate       :         :       :       :         : |  |  |  |  |
TFLITE_ENABLE_MMAP | MMAP を有効にする | ON | ON | ON | N/A

## TensorFlow Lite を使用する CMake プロジェクトの作成

次に、[TFLite の最小限の例](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/minimal)の CMakeLists.txt を示します。

TensorFlow Lite ディレクトリ用に add_subdirectory() を追加し、`tensorflow-lite` を target_link_libraries() に関連付ける必要があります。

```
cmake_minimum_required(VERSION 3.16)
project(minimal C CXX)

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project" )
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

add_executable(minimal minimal.cc)
target_link_libraries(minimal tensorflow-lite)
```

## TensorFlow Lite C ライブラリの構築

[C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/README.md) 用の TensorFlow Lite 共有ライブラリを構築する場合は、まず[手順 1](#step-1-install-cmake-tool) と[手順 3](#step-3-create-cmake-build-directory) に従います。その後、次のコマンドを実行します。

```sh
cmake ../tensorflow_src/tensorflow/lite/c
cmake --build . -j
```

このコマンドでは、現在のディレクトリに次の共有ライブラリが生成されます。

Platform | Library name
--- | ---
Linux | libtensorflowlite_c.so
macOS | libtensorflowlite_c.dylib
Windows | tensorflowlite_c.dll

**注意:** 生成された共有ライブラリを使用するには、必須のヘッダー (c_api.h、c_api_experimental.h、common.h) が必要です。
