# TensorFlow Lite バイナリサイズを縮小する方法

## 概要

オンデバイス機械学習 (ODML) アプリケーションのモデルをデプロイする場合、モバイルデバイスで使用できるメモリの制限に注意することが重要です。モデルのバイナリサイズは、モデルで使用される演算の数と密接に相関しています。TensorFlow Lite では、選択的ビルドを使用してモデルのバイナリサイズを削減できます。選択的ビルドは、モデルセット内の未使用の演算をスキップし、モバイルデバイスでモデルを実行するために必要なランタイムと演算カーネルのみを含むコンパクトなライブラリを生成します。

選択ビルドは、次の 3 つの演算ライブラリに適用されます。

1. [TensorFlow Lite 組み込み演算ライブラリ](https://www.tensorflow.org/lite/guide/ops_compatibility)
2. [TensorFlow Lite カスタム演算](https://www.tensorflow.org/lite/guide/ops_custom)
3. [Select TensorFlow 演算ライブラリ](https://www.tensorflow.org/lite/guide/ops_select)

次の表は、いくつかの一般的なユースケースでの選択的ビルドの影響を示しています。

<table>
  <thead>
    <tr>
      <th>モデル名</th>
      <th>ドメイン</th>
      <th>ターゲットアーキテクチャ</th>
      <th>AAR ファイルサイズ</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="2">       Mobilenet_1.0_224(float)     </td>
    <td rowspan="2">画像分類</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar (296,635 バイト)</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar (382,892 バイト)</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://tfhub.dev/google/lite-model/spice/">SPICE</a>
</td>
    <td rowspan="2">音声のピッチ抽出</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar (375,813 bytes)<br>tensorflow-lite-select-tf-ops.aar (1,676,380 バイト)</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar (421,826 bytes)<br>tensorflow-lite-select-tf-ops.aar (2,298,630 バイト)</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://tfhub.dev/deepmind/i3d-kinetics-400/1">i3d-kinetics-400</a>
</td>
    <td rowspan="2">動画分類</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar (240,085 bytes)<br>tensorflow-lite-select-tf-ops.aar (1,708,597 バイト)</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar (273,713 bytes)<br>tensorflow-lite-select-tf-ops.aar (2,339,697 バイト)</td>
  </tr>
 </table>

注意: 現在、この機能は試験段階にありバージョン 2.4 以降で利用可能になりますが、変更される可能性があります。

## Bazel を使用して TensorFlow Lite を選択的に構築する

このセクションでは、TensorFlow ソースコードをダウンロードし、Bazel に[ローカル開発環境をセットアップ](https://www.tensorflow.org/lite/guide/android#build_tensorflow_lite_locally)していることを前提としています。

### Android プロジェクトの AAR ファイルを構築する

次のようにモデルファイルのパスを指定することで、カスタム TensorFlow Lite AAR を構築できます。

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

上記のコマンドは、TensorFlow Lite 組み込み演算子およびカスタム演算子用の AAR ファイル`bazel-bin/tmp/tensorflow-lite.aar`を生成します。モデルに Select TensorFlow 演算子が含まれている場合、オプションで、AAR ファイル`bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar`を生成します。これにより、複数の異なるアーキテクチャをもつファットな AAR が構築されることに注意してください。それらのすべてが必要ではない場合は、デプロイメント環境に適したサブセットを使用してください。

### カスタム op でビルドする

カスタム演算を使用して Tensorflow Lite モデルを開発した場合は、ビルドコマンドに次のフラグを追加することでモデルを構築できます。

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a \
  --tflite_custom_ops_srcs=/e/f/file1.cc,/g/h/file2.h \
  --tflite_custom_ops_deps=dep1,dep2
```

`tflite_custom_ops_srcs`フラグにはカスタム演算のソースファイルが含まれ、`tflite_custom_ops_deps`フラグにはそれらのソースファイルを構築するための依存関係が含まれます。これらの依存関係は TensorFlow リポジトリに存在する必要があることに注意してください。

### 高度な使用方法: カスタム Bazel ルール

プロジェクトで Bazel を使用しており、特定のモデルセットにカスタム TFLite 依存関係を定義する場合は、プロジェクトリポジトリに、以下のルールを定義できます。

ビルトイン op のみを使用するモデルの場合:

```bazel
load(
    "@org_tensorflow//tensorflow/lite:build_def.bzl",
    "tflite_custom_android_library",
    "tflite_custom_c_library",
    "tflite_custom_cc_library",
)

# A selectively built TFLite Android library.
tflite_custom_android_library(
    name = "selectively_built_android_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)

# A selectively built TFLite C library.
tflite_custom_c_library(
    name = "selectively_built_c_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)

# A selectively built TFLite C++ library.
tflite_custom_cc_library(
    name = "selectively_built_cc_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)
```

[Select TF op](../guide/ops_select.md) を使用するモデルの場合:

```bazel
load(
    "@org_tensorflow//tensorflow/lite/delegates/flex:build_def.bzl",
    "tflite_flex_android_library",
    "tflite_flex_cc_library",
)

# A Select TF ops enabled selectively built TFLite Android library.
tflite_flex_android_library(
    name = "selective_built_tflite_flex_android_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)

# A Select TF ops enabled selectively built TFLite C++ library.
tflite_flex_cc_library(
    name = "selective_built_tflite_flex_cc_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)
```

### 高度な使用方法: カスタム C/C++ 共有ライブラリを構築する

特定のモデルに対し、独自のカスタム TFLite C/C++ 共有オブジェクトを構築する場合、以下の手順に従います。

TensorFlow ソースコードのルートディレクトリで以下のコマンドを実行して、一時BUILDファイルを作成します。

```sh
mkdir -p tmp && touch tmp/BUILD
```

#### カスタム C 共有オブジェクトを構築する

カスタム TFLite C 共有オブジェクトを構築する場合、以下を `tmp/BUILD` ファイルに追加します。

```bazel
load(
    "//tensorflow/lite:build_def.bzl",
    "tflite_custom_c_library",
    "tflite_cc_shared_object",
)

tflite_custom_c_library(
    name = "selectively_built_c_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)

# Generates a platform-specific shared library containing the TensorFlow Lite C
# API implementation as define in `c_api.h`. The exact output library name
# is platform dependent:
#   - Linux/Android: `libtensorflowlite_c.so`
#   - Mac: `libtensorflowlite_c.dylib`
#   - Windows: `tensorflowlite_c.dll`
tflite_cc_shared_object(
    name = "tensorflowlite_c",
    linkopts = select({
        "//tensorflow:ios": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite/c:exported_symbols.lds)",
        ],
        "//tensorflow:macos": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite/c:exported_symbols.lds)",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script,$(location //tensorflow/lite/c:version_script.lds)",
        ],
    }),
    per_os_targets = True,
    deps = [
        ":selectively_built_c_lib",
        "//tensorflow/lite/c:exported_symbols.lds",
        "//tensorflow/lite/c:version_script.lds",
    ],
)
```

新たに追加されたターゲットは、以下のようにして構築できます。

```sh
bazel build -c opt --cxxopt=--std=c++14 \
  //tmp:tensorflowlite_c
```

また、Android については以下のようにします（64 ビットの場合は、`android_arm` を `android_arm64` に置き換えます）。

```sh
bazel build -c opt --cxxopt=--std=c++14 --config=android_arm \
  //tmp:tensorflowlite_c
```

#### カスタム C++ 共有オブジェクトを構築する

カスタム TFLite C++ 共有オブジェクトを構築する場合、以下を `tmp/BUILD` ファイルに追加します。

```bazel
load(
    "//tensorflow/lite:build_def.bzl",
    "tflite_custom_cc_library",
    "tflite_cc_shared_object",
)

tflite_custom_cc_library(
    name = "selectively_built_cc_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)

# Shared lib target for convenience, pulls in the core runtime and builtin ops.
# Note: This target is not yet finalized, and the exact set of exported (C/C++)
# APIs is subject to change. The output library name is platform dependent:
#   - Linux/Android: `libtensorflowlite.so`
#   - Mac: `libtensorflowlite.dylib`
#   - Windows: `tensorflowlite.dll`
tflite_cc_shared_object(
    name = "tensorflowlite",
    # Until we have more granular symbol export for the C++ API on Windows,
    # export all symbols.
    features = ["windows_export_all_symbols"],
    linkopts = select({
        "//tensorflow:macos": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite:tflite_exported_symbols.lds)",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-Wl,-z,defs",
            "-Wl,--version-script,$(location //tensorflow/lite:tflite_version_script.lds)",
        ],
    }),
    per_os_targets = True,
    deps = [
        ":selectively_built_cc_lib",
        "//tensorflow/lite:tflite_exported_symbols.lds",
        "//tensorflow/lite:tflite_version_script.lds",
    ],
)
```

新たに追加されたターゲットは、以下のようにして構築できます。

```sh
bazel build -c opt  --cxxopt=--std=c++14 \
  //tmp:tensorflowlite
```

また、Android については以下のようにします（64 ビットの場合は、`android_arm` を `android_arm64` に置き換えます）。

```sh
bazel build -c opt --cxxopt=--std=c++14 --config=android_arm \
  //tmp:tensorflowlite
```

Select TF op を使用するモデルの場合は、以下の共有ライブラリも構築する必要があります。

```bazel
load(
    "@org_tensorflow//tensorflow/lite/delegates/flex:build_def.bzl",
    "tflite_flex_shared_library"
)

# Shared lib target for convenience, pulls in the standard set of TensorFlow
# ops and kernels. The output library name is platform dependent:
#   - Linux/Android: `libtensorflowlite_flex.so`
#   - Mac: `libtensorflowlite_flex.dylib`
#   - Windows: `libtensorflowlite_flex.dll`
tflite_flex_shared_library(
  name = "tensorflowlite_flex",
  models = [
      ":model_one.tflite",
      ":model_two.tflite",
  ],
)

```

新たに追加されたターゲットは、以下のようにして構築できます。

```sh
bazel build -c opt --cxxopt='--std=c++14' \
      --config=monolithic \
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
      //tmp:tensorflowlite_flex
```

また、Android については以下のようにします（64 ビットの場合は、`android_arm` を `android_arm64` に置き換えます）。

```sh
bazel build -c opt --cxxopt='--std=c++14' \
      --config=android_arm \
      --config=monolithic \
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
      //tmp:tensorflowlite_flex
```

## Docker を使用して TensorFlow Lite を選択的に構築する

このセクションでは、ローカルマシンに [Docker](https://docs.docker.com/get-docker/) をインストールし、[こちら](https://www.tensorflow.org/lite/guide/build_android#set_up_build_environment_using_docker)の TensorFlow Lite Docker ファイルをダウンロードしていることを前提としています。

上記の Dockerfile をダウンロードした後、次のコマンドを実行して Docker イメージを構築できます。

```shell
docker build . -t tflite-builder -f tflite-android.Dockerfile
```

### Android プロジェクトの AAR ファイルを構築する

次のコマンドを実行して、Docker で構築するためのスクリプトをダウンロードします。

```sh
curl -o build_aar_with_docker.sh \
  https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/tools/build_aar_with_docker.sh &&
chmod +x build_aar_with_docker.sh
```

次のようにモデルファイルのパスを指定することで、カスタム TensorFlow Lite AAR を構築できます。

```sh
sh build_aar_with_docker.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a \
  --checkpoint=master \
  [--cache_dir=<path to cache directory>]
```

`checkpoint`フラグは、ライブラリを構築する前に確認する TensorFlow リポジトリのコミット、ブランチ、またはタグです。デフォルトでは、最新のリリースブランチです。上記のコマンドは、TensorFlow Lite 組み込みおよびカスタム演算子用の AAR ファイル`tensorflow-lite.aar`を生成します。また、オプションで、現在のディレクトリにある Select TensorFlow 演算子用の AAR ファイル`tensorflow-lite-select-tf-ops.aar`を生成します。

--cache_dir は、キャッシュディレクトリを指定します。指定しない場合、スクリプトはキャッシュ用の現在の作業ディレクトリの下に`bazel-build-cache`という名前のディレクトリを作成します。

## プロジェクトに AAR ファイルを追加する

直接[プロジェクトに AAR をインポート](https://www.tensorflow.org/lite/guide/android#add_aar_directly_to_project)するか、[カスタム AAR をローカルの Maven リポジトリに公開](https://www.tensorflow.org/lite/guide/android#install_aar_to_local_maven_repository)して、AAR ファイルを追加します。生成する場合は、`tensorflow-lite-select-tf-ops.aar`の AAR ファイルも追加する必要があることに注意してください。

## iOS 用の選択的ビルド

ビルド環境のセットアップと TensorFlow ワークスペースの構成を行うには、[「ローカルで構築する」セクション](../guide/build_ios.md#building_locally)をご覧ください。その後で、[ガイド](../guide/build_ios.md#selectively_build_tflite_frameworks)に従って、iOS 用の選択的ビルドスクリプトを使用します。
