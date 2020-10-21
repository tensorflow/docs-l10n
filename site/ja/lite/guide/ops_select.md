# TensorFlow 演算子を選択する

注意: これは実験的な機能です。

TensorFlow Lite のビルトイン演算子ライブラリがサポートする TensorFlow 演算子は制限されているため、すべてのモデルが互換しているわけではありません。詳細は、[演算子の互換性](ops_compatibility.md)をご覧ください。

変換を可能するために、TensorFlow Lite モデルで[特定の TensorFlow 演算子](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/flex/allowlisted_flex_ops.cc)を使用できるようにします。ただし、TensorFlow 演算子を使って TensorFlow Lite モデルを実行するには、コア TensorFlow ランタイムをプルする必要があるため、TensorFlow Lite インタプリタのバイナリサイズが増大してしまいます。Android では、必要な TensorFlow 演算子のみを選択的に構築することで、このサイズの増大を回避できます。詳細については、[バイナリサイズを縮小する](../guide/reduce_binary_size.md)をご覧ください。

このドキュメントでは、選択したプラットフォームで、TensorFlow 演算子を含む TensorFlow Lite モデルを[変換](#convert_a_model)および[実行](#run_inference)する方法を説明します。また、[パフォーマンスとサイズメトリック](#metrics)および[既知の制限](#known_limitations)についても記載しています。

## モデルを変換する

次の例では、セレクト TensorFlow 演算子を使って TensorFLow Lite モデルを生成する方法を示します。

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

## 推論を実行する

セレクト TensorFlow 演算子のサポートによって変換済みの TensorFlow Lite モデルを使用する場合、クライアントは、TensorFlow 演算子の必要なライブラリを含む TensorFlow Lite ランタイムも使用する必要があります。

### Android AAR

バイナリサイズを縮小するには、[次のセクション](#building-the-android-aar)に示す方法で、独自のカスタム AAR ファイルを構築してください。バイナリサイズがそれほど大きいわけではない場合は、事前構築された、[JCenter でホストされている TensorFlow 演算子を使用した AAR](https://bintray.com/google/tensorflow/tensorflow-lite-select-tf-ops) を使用することをお勧めします。

これは、次のように、`build.gradle` 依存関係に標準の TensorFlow Lite AAR とともに追加することで、指定できます。

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly'
    // This dependency adds the necessary TF op support.
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:0.0.0-nightly'
}
```

依存関係を追加したら、グラフの TensorFlow 演算子を処理するために必要なデリゲートが、それを必要とするグラフに自動的にインストールされます。

*注意*: TensorFlow 演算子の依存関係は比較的に大きいため、`abiFilters` をセットアップして、`.gradle` ファイルの不要な x86 ABI を除外するとよいでしょう。

```build
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

#### Android AAR を構築する

バイナリサイズを縮小するケースやより高度なケースでは、手動でライブラリを構築することもできます。<a href="android.md">稼働中の TensorFlow Lite ビルド環境</a>がある場合、次のように、セレクト TensorFlow 演算子を使用して Android AAR を構築することができます。

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

上記のコードは、TensorFlow Lite のビルトインとカスタム演算子用に、AAR ファイル `bazel-bin/tmp/tensorflow-lite.aar` を生成します。稼働中のビルド環境がない場合は、[Docker を使って上記のファイルを作成](../guide/reduce_binary_size.md#selectively_build_tensorflow_lite_with_docker)することも可能です。

生成したら、AAR ファイルを直接プロジェクトにインポートするか、カスタム AAR ファイルをローカルの Maven リポジトリに公開することが可能です。

```sh
mvn install:install-file \
  -Dfile=bazel-bin/tmp/tensorflow-lite.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite -Dversion=0.1.100 -Dpackaging=aar
mvn install:install-file \
  -Dfile=bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite-select-tf-ops -Dversion=0.1.100 -Dpackaging=aar
```

最後に、アプリの `build.gradle` で、`mavenLocal()` 依存関係があることを確認し、標準の TensorFlow Lite 依存関係を、セレクト TensorFlow 演算子をサポートする依存関係に置き換えます。

```build
allprojects {
    repositories {
        jcenter()
        mavenLocal()
    }
}

dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.1.100'
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:0.1.100'
}
```

### iOS

#### CocoaPods を使用する

構築済みのセレクト TF 演算子 CocoaPods をナイトリーで提供しており、`TensorFlowLiteSwift` または `TensorFlowLiteObjC` CocoaPods とともに利用することができます。

```ruby
# In your Podfile target:
  pod 'TensorFlowLiteSwift'   # or 'TensorFlowLiteObjC'
  pod 'TensorFlowLiteSelectTfOps', '~> 0.0.1-nightly'
```

`pod install` を実行後、セレクト TF 演算子フレームワークをプロジェクトに強制読み込みできるように、追加のリンカーフラグを指定する必要があります。Xcode プロジェクトで、`Build Settings` -> `Other Linker Flags` に移動し、次を追加します。

```text
-force_load $(SRCROOT)/Pods/TensorFlowLiteSelectTfOps/Frameworks/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps
```

すると、`SELECT_TF_OPS` で変換されたモデルを iOS アプリで実行できるようになります。たとえば、[Image Classification iOS アプリ](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios)を変更して、セレクト TF 演算子の機能をテストすることができます。

- モデルファイルを `SELECT_TF_OPS` が有効化された状態で変換されたファイルと置き換えます。
- 指示されたとおりに、`TensorFlowLiteSelectTfOps` 依存関係を `Podfile` に追加します。
- 上記のように、リンカーフラグを追加します。
- サンプルアプリを実行し、モデルが正しく機能するかどうかを確認します。

#### Bazel + Xcode を使用する

iOS 用のセレクト TensorFlow 演算子を使った TensorFlow Lite は Bazel を使って構築できます。まず、[iOS の構築手順](build_ios.md)に従って、 Bazel ワークスペースと `.bazelrc` ファイルを正しく構成します。

iOS サポートを有効化したワークスペースを構成したら、次のコマンドを使用して、セレクト TF 演算子のアドオンフレームワークを構築します。これは、通常の `TensorFlowLiteC.framework` の上に追加されるフレームワークです。セレクト TF 演算子フレームワークは、`i386` アーキテクチャには構築できまないため、`i386` を除くターゲットアーキテクチャのｎリストを明示的に指定する必要があります。

```sh
bazel build -c opt --config=ios --ios_multi_cpus=armv7,arm64,x86_64 \
  //tensorflow/lite/experimental/ios:TensorFlowLiteSelectTfOps_framework
```

これにより、`bazel-bin/tensorflow/lite/experimental/ios/` ディレクトリにフレームワークが生成されます。iOS 構築ガイドの [Xcode プロジェクト設定](./build_ios.md#modify_xcode_project_settings_directly)セクションに説明された手順と同様の手順を実行し、この新しいフレームワークを Xcode プロジェクトに追加することができます。

フレームワークをアプリのプロジェクトに追加したら、セレクト TF 演算子フレームワークを強制読み込みできるように、追加のリンカーフラグをアプリのプロジェクトに指定する必要があります。Xcode プロジェクトで、`Build Settings` -> `Other Linker Flags` に移動し、次を追加します。

```text
-force_load <path/to/your/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps>
```

### C++

bazel パイプラインを使用して TensorFlow Lite ライブラリを構築する際に、追加の TensorFlow 演算子ライブラリを含めて次のように有効化することができます。

- 必要に応じて、`--config=monolithic` ビルドフラグを追加して、モノリシックビルドを有効化します。
- 次のように、TensorFlow 演算子デリゲートライブラリの依存関係をビルド依存関係に追加します。 `tensorflow/lite/delegates/flex:delegate`

デリゲートがクライアントライブラリにリンクされている限り、必要な `TfLiteDelegate` は、ランタイム時にインタプリタを作成するときに自動的にインストールされます。通常ほかのデリゲートタイプでも必要とされるため、明示的にデリゲートインスタンスをインストールする必要はありません。

### Python

セレクト TensorFlow 演算子を使用する TensorFlow Lite は、自動的に [TensorFlow pip パッケージ](https://www.tensorflow.org/install/pip)と合わせてインストールされます。[TensorFlow Lite Interpreter pip パッケージ](https://www.tensorflow.org/lite/guide/python#install_just_the_tensorflow_lite_interpreter)のみをインストールするように選択することも可能です。

注意: セレクト TensorFlow 演算子を使った TensorFlow Lite は、Linux の場合はバージョン 2.3 以降、その他の環境の場合はバージョン 2.4 以降の TensorFlow pip パッケージで利用可能です。

## メトリック

### パフォーマンス

ビルトインとセレクトの両方の TensorFlow 演算子を使用する場合、すべての同一の TensorFlow Lite 最適化と最適化済みのビルトイン演算子を利用できるようになり、変換済みモデルで使用できます。

次のテーブルは、Pixel 2 の MobileNet で推論を実行するためにかかる平均時間を示します。リストされている時間は、100 回の実行の平均に基づきます。これらのターゲットは、フラグ `--config=android_arm64 -c opt` を使用して、Android 向けに構築されています。

ビルド | 時間（ミリ秒）
--- | ---
ビルトイン演算子のみ（`TFLITE_BUILTIN`） | 260.7
TF 演算子のみを使用（`SELECT_TF_OPS`） | 264.5

### バイナリサイズ

次のテーブルは、TensorFlow Lite の各ビルドのバイナリサイズを示します。これらのターゲットは `--config=android_arm -c opt` を使って Android 向けに構築されています。

ビルド | C++ バイナリサイズ | Android APK サイズ
--- | --- | ---
ビルトイン演算子のみ | 796 KB | 561 KB
ビルトイン演算子 + TF 演算子 | 23.0 MB | 8.0 MB
ビルトイン演算子 + TF 演算子（1） | 4.1 MB | 1.8 MB

（1）上記のライブラリは、8 個の TFLite ビルトイン演算子と 3 個の TensorFLow 演算子を使って、[i3d-kinetics-400 モデル](https://tfhub.dev/deepmind/i3d-kinetics-400/1)向けに選択的に構築されています。詳細については、[TensorFlow Lite のバイナリサイズを縮小する](../guide/reduce_binary_size.md)セクションをご覧ください。

## 既知の制限

- 特定の TensorFlow 演算子は、ストック TensorFlow で通常利用可能な全セットの入力型/出力型をサポートしていない場合があります。
- サポートされていない演算: 制御フロー演算と`HashTableV2` など、リソースから明示的な初期化が必要な演算は、まだサポートされていません。
- サポートされていない最適化: [ポストトレーニング量子化](../performance/post_training_quantization.md)として知られる再帰化を適用する場合、量子化されるのは TensorFlow Lite 演算子のみであり、TensorFlow 演算は浮動小数点のまま（未最適化）になります。

## 今後の予定

次の項目は、現在取り組み中となっている、このパイプラインへの機能強化です。

- *パフォーマンスの改善* - TensorFlow 演算子を使った TensorFlow Lite が、NNPI や GPU デリゲートなどのハードウェアアクセラレーションによるデリゲートとうまく動作できるように作業が進められています。
