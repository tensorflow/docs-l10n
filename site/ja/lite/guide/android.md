# Android クイックスタート

Android で TensorFlow Lite を使い始めるには、次の例をご覧ください。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android">Android image classification example</a>

ソースコードの説明については [TensorFlow Lite 画像分類の例](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/EXPLORE_THE_CODE.md)を参照してください。

この例のアプリは[画像分類](https://www.tensorflow.org/lite/models/image_classification/overview)を使用して、デバイスの背面カメラがキャプチャした画像を継続的に分類します。アプリは、デバイスまたはエミュレータで実行できます。

推論は、TensorFlow Lite Java API と[TensorFlow Lite Android サポートライブラリ](../inference_with_metadata/lite_support.md)を使用して実行します。デモアプリはリアルタイムでフレームを分類し、最も可能性の高い分類を表示します。ユーザーは浮動小数点モデルまたは[量子化](https://www.tensorflow.org/lite/performance/post_training_quantization)モデルのいずれかを選択し、スレッド数を選択して、CPU、GPU、または[ NNAPI ](https://developer.android.com/ndk/guides/neuralnetworks)で実行するかを決定できます。

注: さまざまなユースケースで TensorFlow Lite をデモするその他の Android アプリは、[例](https://www.tensorflow.org/lite/examples)をご覧ください。

## Android Studio で構築する

Android Studio でサンプルをビルドするには、[README.md](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md) の指示に従ってください。

## 独自の Android アプリを作成する

独自の Android コードを迅速に記述するには、はじめに [ Android 画像分類の例](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android)を使用することをお勧めします。

次のセクションには、Android で TensorFlow Lite を使用する際に役立つ情報が含まれています。

### Android Studio ML モデルバインディングを使用する

注意: [Android Studio 4.1](https://developer.android.com/studio) 以上が必要です。

TensorFlow Lite（TFLite）モデルをインポートするには、次を行います。

1. TF Lite モデルを使用するモジュールを右クリックするか、`File` をクリックして、`New` &gt; `Other` &gt; `TensorFlow Lite Model` に移動します。![Right-click menus to access the TensorFlow Lite import functionality](../images/android/right_click_menu.png)

2. TFLite ファイルの場所を選択します。ユーザーに代わってツールが、ML Model バインディングとのモジュールの依存関係と Android モジュールの `build.gradle` ファイルに自動的に挿入されたすべての依存関係を構成します。

    オプション: [GPU アクセラレーション](../performance/gpu)を使用する場合は、TensorFlow GPU をインポートするための 2 番目のチェックをオンにしてください。 ![Import dialog for TFLite model](../images/android/import_dialog.png)

3. `Finish` をクリックします。

4. インポートが正常に完了すると、次の画面が表示されます。モデルを使用し始めるには、Kotlin または Java を選択し、`Sample Code` セクションにあるコードをコピーして貼り付けます。Android Studio の `ml` ディレクトリにある TFLite モデルをダブルクリックすると、この画面に戻ることができます。![Model details page in Android Studio](../images/android/model_details.png)

### TensorFlow Lite Task ライブラリを使用する

TensorFlow Lite Task ライブラリには、アプリ開発者が TFLite を使って ML エクスペリエンスを作成できるように、強力で使いやすいタスク固有の一連のライブラリが含まれています。画像の分類、質問と回答など、一般的な機械学習タスク用に最適化された は、画像の分類、質問と回答など、一般的な機械学習タスク用に最適化された、すぐに使用できるモデルインターフェースが得られます。モデルインターフェースは、最高のパフォーマンスと使いやすさを実現するために、タスクごとに特別に設計されています。Task ライブラリはクロスプラットフォームで動作し、Java、C++、および Swift （近日）でサポートされています。

To use the Support Library in your Android app, we recommend using the AAR hosted at MavenCentral for [Task Vision library](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-vision) and [Task Text library](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-text) , respectively.

これは、`build.gradle` 依存関係に次のように指定できます。

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-task-vision:0.3.0'
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.3.0'
    implementation 'org.tensorflow:tensorflow-lite-task-audio:0.3.0'
}
```

To use nightly snapshots, make sure that you have added [Sonatype snapshot repository](./build_android#use_nightly_snapshots).

詳細は、[TensorFlow Lite Task ライブラリの概要](../inference_with_metadata/task_library/overview.md)の概要のセクションをご覧ください。

### TensorFlow Lite Android Support ライブラリを使用する

TensorFlow Lite Android Support ライブラリを使用すると、モデルをアプリケーションに簡単に統合できます。生の入力データをモデルが必要とする形式に変換し、モデルの出力を解釈するのに役立つ高レベルの API を提供し、必要なボイラープレートコードの量を減らします。

画像や配列など、入力と出力の一般的なデータ形式をサポートしています。また、画像のサイズ変更やトリミングなどのタスクを実行する前処理ユニットと後処理ユニットも提供されています。

To use the Support Library in your Android app, we recommend using the [TensorFlow Lite Support Library AAR hosted at MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-support).

これは、`build.gradle` 依存関係に次のように指定できます。

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-support:0.3.0'
}
```

To use nightly snapshots, make sure that you have added [Sonatype snapshot repository](./build_android#use_nightly_snapshots).

始めるには、[TensorFlow Lite Android Support ライブラリ](../inference_with_metadata/lite_support.md)の手順に従ってください。

### Use the TensorFlow Lite AAR from MavenCentral

To use TensorFlow Lite in your Android app, we recommend using the [TensorFlow Lite AAR hosted at MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite).

これは、`build.gradle` 依存関係に次のように指定できます。

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
}
```

To use nightly snapshots, make sure that you have added [Sonatype snapshot repository](./build_android#use_nightly_snapshots).

NDK でアプリをビルドする場合、C++ で TFLite を使用する方法が 2 つあります。

これは*推薦される*アプローチです。[JCenter でホストされている TensorFlow Lite AAR ](https://bintray.com/google/tensorflow/tensorflow-lite) をダウンロードし、名前を `tensorflow-lite-*.zip` に変更して、解凍します。`headers/tensorflow/lite/` および `headers/tensorflow/lite/c/` フォルダに 4 つのヘッダーファイルを含め、NDK プロジェクトの `jni/` フォルダに関連する `libtensorflowlite_jni.so` 動的ライブラリを含める必要があります。

```build
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

`c_api.h` ヘッダーファイルには、TFLite C API の使用に関する基本的なドキュメントが含まれています。

## C++ を使用して Android アプリを構築する

C++ API を介して TFLite を使用する場合は、C++ 共有ライブラリを構築します。

### TFLite C++ API を使用する

This is the *recommended* approach. Download the [TensorFlow Lite AAR hosted at MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow/tensorflow-lite), rename it to `tensorflow-lite-*.zip`, and unzip it. You must include the four header files in `headers/tensorflow/lite/` and `headers/tensorflow/lite/c/` folder and the relevant `libtensorflowlite_jni.so` dynamic library in `jni/` folder in your NDK project.

`c_api.h` ヘッダーファイルには、TFLite C API の使用に関する基本的なドキュメントが含まれています。

### TFLite C++ API を使用する

現在、必要なすべてのヘッダーファイルを抽出する簡単な方法はないため、TensorFlow リポジトリから `tensorflow/lite/` にすべてのヘッダーファイルを含める必要があります。さらに、[FlatBuffers](https://github.com/google/flatbuffers) および [Abseil](https://github.com/abseil/abseil-cpp) からのヘッダーファイルが必要になります。

32bit armeabi-v7a:

```sh
bazel build -c opt --config=android_arm //tensorflow/lite:libtensorflowlite.so
```

64bit arm64-v8a:

```sh
bazel build -c opt --config=android_arm64 //tensorflow/lite:libtensorflowlite.so
```

現在、必要なすべてのヘッダーファイルを抽出する簡単な方法はないため、TensorFlow リポジトリから `tensorflow/lite/` にすべてのヘッダーファイルを含める必要があります。さらに、[FlatBuffers](https://github.com/google/flatbuffers) および [Abseil](https://github.com/abseil/abseil-cpp) からのヘッダーファイルが必要になります。

## TFLite の SDK 最低バージョン

ライブラリ | `minSdkVersion` | デバイス要件
--- | --- | ---
tensorflow-lite | 19 | NNAPI の使用が必要
:                             :                 : API 27+                : |  |
tensorflow-lite-gpu | 19 | GLES 3.1 または OpenCL
:                             :                 : （通常        : |  |
:                             :                 : API 21+ のみで利用可  : |  |
tensorflow-lite-hexagon | 19 | -
tensorflow-lite-support | 19 | -
tensorflow-lite-task-vision | 21 | android.graphics.Color
:                             :                 : 関連 API が必要   : |  |
:                             :                 : API 26+                : |  |
tensorflow-lite-task-text | 21 | -
tensorflow-lite-task-audio | 23 | -
tensorflow-lite-metadata | 19 | -
