# TensorFlow Lite バイナリサイズを縮小する方法

## 概要

デバイス上の機械学習 (ODML) アプリケーションのモデルをデプロイする場合、モバイルデバイスで使用できるメモリの制限に注意することが重要です。モデルのバイナリサイズは、モデルで使用される演算の数と密接に相関しています。TensorFlow Lite では、選択的ビルドを使用してモデルのバイナリサイズを削減できます。選択的ビルドは、モデルセット内の未使用の演算をスキップし、モバイルデバイスでモデルを実行するために必要なランタイムと演算カーネルのみを含むコンパクトなライブラリを生成します。

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
    <td rowspan="2"><a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(float)</a></td>
    <td rowspan="2">画像分類</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar (296,635 バイト)</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar (382,892 バイト)</td>
  </tr>
  <tr>
    <td rowspan="2"><a href="https://tfhub.dev/google/lite-model/spice/">SPICE</a></td>
    <td rowspan="2">音声のピッチ抽出</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar (375,813 bytes)<br>tensorflow-lite-select-tf-ops.aar (1,676,380 バイト)</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar (421,826 bytes)<br>tensorflow-lite-select-tf-ops.aar (2,298,630 バイト)</td>
  </tr>
  <tr>
    <td rowspan="2"><a href="https://tfhub.dev/deepmind/i3d-kinetics-400/1">i3d-kinetics-400</a></td>
    <td rowspan="2">動画分類</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar (240,085 bytes)<br>tensorflow-lite-select-tf-ops.aar (1,708,597 バイト)</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar (273,713 bytes)<br>tensorflow-lite-select-tf-ops.aar (2,339,697 バイト)</td>
  </tr>
 </table>

注: 現在、この機能は実験段階にありバージョン 2.4 以降で利用可能になりますが、変更される可能性があります。

## 既知の問題/制限

1. C API および iOS バージョンの選択的ビルドは現在サポートされていません。

## Bazel を使用して TensorFlow Lite を選択的に構築する

このセクションでは、TensorFlow ソースコードをダウンロードし、Bazel に[ローカル開発環境をセットアップ](https://www.tensorflow.org/lite/guide/android#build_tensorflow_lite_locally)していることを前提としています。

### Android プロジェクトの AAR ファイルを構築する

次のようにモデルファイルのパスを指定することで、カスタム TensorFlow Lite AAR を構築できます。

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

上記のコマンドは、TensorFlow Lite 組み込みおよびカスタム演算用の AAR ファイル`bazel-bin/tmp/tensorflow-lite.aar`を生成します。モデルに Select TensorFlow 演算が含まれている場合、オプションで、AAR ファイル`bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar`を生成します。これにより、複数の異なるアーキテクチャをもつファットな AAR が構築されることに注意してください。それらのすべてが必要ではない場合は、デプロイメント環境に適したサブセットを使用してください。

### 高度な使用法：カスタム演算で構築する

カスタム演算を使用して Tensorflow Lite モデルを開発した場合は、ビルドコマンドに次のフラグを追加することでモデルを構築できます。

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a \
  --tflite_custom_ops_srcs=/e/f/file1.cc,/g/h/file2.h \
  --tflite_custom_ops_deps=dep1,dep2
```

`tflite_custom_ops_srcs`フラグにはカスタム演算のソースファイルが含まれ、`tflite_custom_ops_deps`フラグにはそれらのソースファイルを構築するための依存関係が含まれます。これらの依存関係は TensorFlow リポジトリに存在する必要があることに注意してください。

## Docker を使用して TensorFlow Lite を選択的に構築する

このセクションでは、ローカルマシンに [Docker](https://docs.docker.com/get-docker/) をインストールし、[TensorFlow Lite Docker ファイルを構築](https://www.tensorflow.org/lite/guide/android#set_up_build_environment_using_docker)していることを前提としています。

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
  --checkpoint=master
```

`チェックポイント`フラグは、ライブラリを構築する前に確認する TensorFlow リポジトリのコミット、ブランチ、またはタグです。上記のコマンドは、TensorFlow Lite 組み込みおよびカスタム演算用の AAR ファイル`tensorflow-lite.aar`を生成します。また、オプションで、現在のディレクトリにある Select TensorFlow 演算用の AAR ファイル`tensorflow-lite-select-tf-ops.aar`を生成します。

## プロジェクトに AAR ファイルを追加する

直接[プロジェクトに AAR をインポート](https://www.tensorflow.org/lite/guide/android#add_aar_directly_to_project)するか、[カスタム AAR をローカルの Maven リポジトリに公開](https://www.tensorflow.org/lite/guide/android#install_aar_to_local_maven_repository)して、AAR ファイルを追加します。生成する場合は、`tensorflow-lite-select-tf-ops.aar`の AAR ファイルも追加する必要があることに注意してください。
