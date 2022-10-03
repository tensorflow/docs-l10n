# Android 版開発ツール

TensorFlow Lite は、モデルを Android アプリに統合するためのさまざまなツールを提供しています。このページでは、Kotlin、Java、C++ でアプリを作成するために使用できる開発ツールと、Android Studio での TensorFlow Lite 開発のサポートについて説明します。

要点: 一般的に、ユースケースが [TensorFlow Lite タスクライブラリ](#task_library)でサポートされている場合は、このタスクライブラリを使用して、TensorFlow Lite を Android アプリに統合してください。このタスクライブラリでサポートされていない場合は、[TensorFlow Lite ライブラリ](#lite_lib)と[サポートライブラリ](#support_lib)を使用してください。

Android コードを簡単に作成するための基本事項については、[Quickstart for Android](../android/quickstart) を参照してください。

## Kotlin および Java で作成するためのツール

次のセクションでは、Kotlin および Java 言語を使用する、TensorFlow Lite 用の開発ツールについて説明します。

### TensorFlow Lite Task Library {:#task_library}

TensorFlow Lite タスクライブラリには、強力で使いやすいタスク固有の一連のライブラリが含まれているので、アプリ開発者は TensorFlow Lite を使用して機械学習を利用できます。画像分類や質疑応答といった一般的な機械学習タスクにすぐに使用できるように、最適化されたモデルインターフェースが用意されており、モデルインターフェイスは、タスクごとに最高のパフォーマンスと使いやすさが得られるように設計されています。タスクライブラリはクロスプラットフォームで動作し、Java および C++ でサポートされています。

To use the Task Library in your Android app, use the AAR from MavenCentral for [Task Vision library](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-vision) , [Task Text library](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-text) and [Task Audio Library](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-audio) , respectively.

これは、`build.gradle` 依存関係に次のように指定できます。

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-task-vision:+'
    implementation 'org.tensorflow:tensorflow-lite-task-text:+'
    implementation 'org.tensorflow:tensorflow-lite-task-audio:+'
}
```

ナイトリーのスナップショットを使用する場合は、プロジェクトに [Sonatype スナップショットリポジトリ](./lite_build#use_nightly_snapshots)が追加されていることを確認してください。

詳細は、[TensorFlow Lite タスクライブラリの概要](../inference_with_metadata/task_library/overview.md)の概要のセクションをご覧ください。

### TensorFlow Lite ライブラリ {:#lite_lib}

[MavenCentral にホストされている AAR](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite) を開発プロジェクトに追加するには、Android アプリで TensorFlow Lite ライブラリを使用します。

これは、`build.gradle` 依存関係に次のように指定できます。

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:+'
}
```

毎日夜間に実行されるスナップショットを使用するには、[Sonatype スナップショットリポジトリ](./lite_build#use_nightly_snapshots)が追加されていることを確認してください。

この AAR には、すべての [Android ABI](https://developer.android.com/ndk/guides/abis) のバイナリが含まれています。サポートする必要のある ABI のみを含めることで、アプリケーションのバイナリのサイズを削減できます。

ほとんどの場合において、特定のハードウェアを対象としていないかぎりは、`x86`、`x86_64`、および `arm32` ABI を省略してください。これは、次の Gradle 構成を使って実現できます。この構成には、最新のほとんどの Android デバイスに対応する `armeabi-v7a` と `arm64-v8a` のみが含まれています。

```build
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

`abiFilters` の詳細については、Android NDK ドキュメントの [Android ABI](https://developer.android.com/ndk/guides/abis) を参照してください。

### TensorFlow Lite Support Library {:#support_lib}

TensorFlow Lite Android Support ライブラリを使用すると、モデルをアプリケーションに簡単に統合できます。生の入力データをモデルが必要とする形式に変換し、モデルの出力を解釈するのに役立つ高レベルの API を提供し、必要なボイラープレートコードの量を減らします。

画像や配列など、入力と出力の一般的なデータ形式をサポートしています。また、画像のサイズ変更やトリミングなどのタスクを実行する前処理ユニットと後処理ユニットも提供されています。

Android アプリで Support ライブラリを使用するには、[MavenCentral でホスティングされている TensorFlow Lite Support ライブラリ AAR](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-support) を追加してください。

これは、`build.gradle` 依存関係に次のように指定できます。

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-support:+'
}
```

ナイトリーのスナップショットを使用する場合は、プロジェクトに [Sonatype スナップショットリポジトリ](./lite_build#use_nightly_snapshots)が追加されていることを確認してください。

基本的な手順については、[TensorFlow Lite Android Support Library](../inference_with_metadata/lite_support.md) を参照してください。

### ライブラリに対応する最低 Android SDK バージョン

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

### Android Studio の使用

前述の開発ライブラリのほかに、次に示すように、Android Studio も TensorFlow Lite モデルを統合するためのサポートを提供します。

#### Android Studio ML モデルバインディング

Android Studio 4.1 以降の ML モデルバインディング機能では、`.tflite` モデルファイルを既存の Android アプリにインポートし、インターフェイスクラスを生成して、簡単にコードをモデルに統合できます。

TensorFlow Lite（TFLite）モデルをインポートするには、次を行います。

1. TF Lite モデルを使用するモジュールを右クリックするか、**File &gt; New &gt; Other &gt; TensorFlow Lite Model** をクリックします。

2. TensorFlow Lite ファイルの場所を選択します。ツールの ML モデルバインディングによって、モジュールの依存関係が構成され、すべての必要な依存関係が Android モジュールの `build.gradle` ファイルに自動的に追加されます。

    注意: [GPU アクセラレーション](../performance/gpu)を使用する場合は、TensorFlow GPU をインポートするための 2 番目のチェックボックスをオンにしてください。

3. `Finish` をクリックすると、インポート処理が開始します。インポートが完了したら、入力および出力テンソルを含むモデルを説明する画面が表示されます。

4. モデルの使用を開始するには、Kotlin または Java を選択し、コードをコピーして、**Sample Code** セクションにコードを貼り付けます。

モデル情報画面に戻るには、Android Studio の `ml` ディレクトリで TensorFlow Lite モデルをダブルクリックします。Android Studio のモデルバインディング機能の詳細については、Android Studio [リリースノート](https://developer.android.com/studio/releases#4.1-tensor-flow-lite-models)を参照してください。Android Studio でのモデルバインディングの使用に関する概要については、コード例の[手順](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md)を参照してください。

## C および C++ で作成するためのツール

TensorFlow Lite の C および C++ ライブラリは、主に、Android Native Development Kit (NDK) を使用してアプリを開発する開発者向けです。NDK でアプリを作成する場合、C++ で TFLite を使用するには、次の 2 つの方法があります。

### TFLite C API

この API の使用は、NDK を使用する開発者向けに*推薦される*アプローチです。[MavenCentral でホスティングされている TensorFlow Lite AAR ](https://search.maven.org/artifact/org.tensorflow/tensorflow/tensorflow-lite) ファイルをダウンロードし、名前を `tensorflow-lite-*.zip` に変更して、解凍します。`headers/tensorflow/lite/` および `headers/tensorflow/lite/c/` フォルダに 4 つのヘッダーファイルを含め、NDK プロジェクトの `jni/` フォルダに関連する `libtensorflowlite_jni.so` 動的ライブラリを含める必要があります。

`c_api.h` ヘッダーファイルには、TFLite C API の使用に関する基本的なドキュメントが含まれています。

### TFLite C++ API

C++ API を介して TFLite を使用する場合は、C++ 共有ライブラリを構築します。

32bit armeabi-v7a:

```sh
bazel build -c opt --config=android_arm //tensorflow/lite:libtensorflowlite.so
```

64bit arm64-v8a:

```sh
bazel build -c opt --config=android_arm64 //tensorflow/lite:libtensorflowlite.so
```

現在、必要なすべてのヘッダーファイルを抽出する簡単な方法はないため、TensorFlow リポジトリから`tensorflow/lite/` にすべてのヘッダーファイルを含める必要があります。さらに、[FlatBuffers](https://github.com/google/flatbuffers) および [Abseil](https://github.com/abseil/abseil-cpp) からのヘッダーファイルが必要になります。
