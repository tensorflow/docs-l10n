# TensorFlow Lite Hexagon デリゲート

このドキュメントでは、Java または C 、あるいはその両方の API を使用して、アプリで TensorFlow Lite Hexagon デレゲートを使用する方法について説明します。デリゲートは、Qualcomm Hexagon ライブラリを利用して、DSP で量子化されたカーネルを実行します。デリゲートは、特に NNAPI DSP アクセラレーションが利用できないデバイス（古いデバイス、または DSP NNAPI ドライバをまだ備えていないデバイスなど）の NNAPI 機能を*補完*することを目的としていることに注意してください。

注意: このデリゲートは実験（ベータ）段階です。

**サポートされているデバイス：**

現在、次のヘキサゴナルアーキテクチャがサポートされていますが、これらに限定されません。

- Hexagon 680
    - SoC の例：Snapdragon 821、820、660
- Hexagon 682
    - SoC の例：Snapdragon 835
- Hexagon 685
    - SoC の例：Snapdragon 845、Snapdragon 710、QCS605、QCS603
- Hexagon 690
    - SoC の例：Snapdragon 855、QCS610、QCS410、RB5

**サポートされているモデル：**

Hexagon デリゲートは、[トレーニング後の整数量子化](https://www.tensorflow.org/lite/performance/quantization_spec)を使用して生成されたものを含め、[8 ビット対称量子化仕様](https://www.tensorflow.org/lite/performance/post_training_integer_quant)に準拠するすべてのモデルをサポートしています。従来の[量子化認識トレーニング](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize)パスでトレーニングされた UInt8 モデルもサポートされています（例：ホステッドモデルページの[これらの量子化バージョン](https://www.tensorflow.org/lite/guide/hosted_models#quantized_models)）。

## Hexagon デレゲート Java API

```java
public class HexagonDelegate implements Delegate, Closeable {

  /*
   * Creates a new HexagonDelegate object given the current 'context'.
   * Throws UnsupportedOperationException if Hexagon DSP delegation is not
   * available on this device.
   */
  public HexagonDelegate(Context context) throws UnsupportedOperationException


  /**
   * Frees TFLite resources in C runtime.
   *
   * User is expected to call this method explicitly.
   */
  @Override
  public void close();
}
```

### 使用例

#### ステップ 1. app/build.gradle を編集して、ナイトリーの Hexagon デリゲート AAR を使用する

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
  implementation 'org.tensorflow:tensorflow-lite-hexagon:0.0.0-nightly-SNAPSHOT'
}
```

#### ステップ 2. Hexagon ライブラリを Android アプリに追加する

- hexagon_nn_skel.run をダウンロードして実行します。3 つの異なる共有ライブラリ「libhexagon_nn_skel.so」、「libhexagon_nn_skel_v65.so」、「libhexagon_nn_skel_v66.so」が提供されます。
    - v1.10.3
    - v1.14
    - v1.17
    - v1.20
    - [v1.20.0.1](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.1.run)

注意：ライセンス契約に同意する必要があります。

注意：2021 年 2 月 23 日時点では、v1.20.0.1 を使用する必要があります。

注意：hexagon_nn ライブラリは、互換性のあるバージョンのインターフェイスライブラリと使用する必要があります。インターフェイスライブラリは AAR の一部であり、[config](https://github.com/tensorflow/tensorflow/blob/master/third_party/hexagon/workspace.bzl) を通じて bazel によりフェッチされます。bazelconfig のバージョンを使用する必要があります。

- 他の共有ライブラリと共にアプリに 3 つすべて追加します。 [アプリに共有ライブラリを追加する方法](#how-to-add-shared-library-to-your-app)を参照してください。デリゲートは、デバイスに応じて最高のパフォーマンスを持つものを自動的に選択します。

注意: アプリが 32 ビットと 64 ビットの両方の ARM デバイス用に構築される場合、32 ビットと 64 ビットの両方の lib フォルダに Hexagon 共有ライブラリを追加する必要があります。

#### ステップ 3. デリゲートを作成して TensorFlow Lite インタプリタを初期化する

```java
import org.tensorflow.lite.HexagonDelegate;

// Create the Delegate instance.
try {
  hexagonDelegate = new HexagonDelegate(activity);
  tfliteOptions.addDelegate(hexagonDelegate);
} catch (UnsupportedOperationException e) {
  // Hexagon delegate is not supported on this device.
}

tfliteInterpreter = new Interpreter(tfliteModel, tfliteOptions);

// Dispose after finished with inference.
tfliteInterpreter.close();
if (hexagonDelegate != null) {
  hexagonDelegate.close();
}
```

## Hexagon デリゲート C API

```c
struct TfLiteHexagonDelegateOptions {
  // This corresponds to the debug level in the Hexagon SDK. 0 (default)
  // means no debug.
  int debug_level;
  // This corresponds to powersave_level in the Hexagon SDK.
  // where 0 (default) means high performance which means more power
  // consumption.
  int powersave_level;
  // If set to true, performance information about the graph will be dumped
  // to Standard output, this includes cpu cycles.
  // WARNING: Experimental and subject to change anytime.
  bool print_graph_profile;
  // If set to true, graph structure will be dumped to Standard output.
  // This is usually beneficial to see what actual nodes executed on
  // the DSP. Combining with 'debug_level' more information will be printed.
  // WARNING: Experimental and subject to change anytime.
  bool print_graph_debug;
};

// Return a delegate that uses Hexagon SDK for ops execution.
// Must outlive the interpreter.
TfLiteDelegate*
TfLiteHexagonDelegateCreate(const TfLiteHexagonDelegateOptions* options);

// Do any needed cleanup and delete 'delegate'.
void TfLiteHexagonDelegateDelete(TfLiteDelegate* delegate);

// Initializes the DSP connection.
// This should be called before doing any usage of the delegate.
// "lib_directory_path": Path to the directory which holds the
// shared libraries for the Hexagon NN libraries on the device.
void TfLiteHexagonInitWithPath(const char* lib_directory_path);

// Same as above method but doesn't accept the path params.
// Assumes the environment setup is already done. Only initialize Hexagon.
Void TfLiteHexagonInit();

// Clean up and switch off the DSP connection.
// This should be called after all processing is done and delegate is deleted.
Void TfLiteHexagonTearDown();
```

### 使用例

#### ステップ 1. app/build.gradle を編集して、ナイトリーの Hexagon デリゲート AAR を使用する

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
  implementation 'org.tensorflow:tensorflow-lite-hexagon:0.0.0-nightly-SNAPSHOT'
}
```

#### ステップ 2. Hexagon ライブラリを Android アプリに追加する

- hexagon_nn_skel.run をダウンロードして実行します。3 つの異なる共有ライブラリ「libhexagon_nn_skel.so」、「libhexagon_nn_skel_v65.so」、「libhexagon_nn_skel_v66.so」が提供されます。
    - v1.10.3
    - v1.14
    - v1.17
    - v1.20
    - [v1.20.0.1](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.1.run)

注意：ライセンス契約に同意する必要があります。

注意：2021 年 2 月 23 日時点では、v1.20.0.1 を使用する必要があります。

注意：hexagon_nn ライブラリは、互換性のあるバージョンのインターフェイスライブラリと使用する必要があります。インターフェイスライブラリは AAR の一部であり、[config](https://github.com/tensorflow/tensorflow/blob/master/third_party/hexagon/workspace.bzl) を通じて bazel によりフェッチされます。bazelconfig のバージョンを使用する必要があります。

- 他の共有ライブラリと共にアプリに 3 つすべて追加します。 [アプリに共有ライブラリを追加する方法](#how-to-add-shared-library-to-your-app)を参照してください。デリゲートは、デバイスに応じて最高のパフォーマンスを持つものを自動的に選択します。

注意: アプリが 32 ビットと 64 ビットの両方の ARM デバイス用に構築される場合、32 ビットと 64 ビットの両方の lib フォルダに Hexagon 共有ライブラリを追加する必要があります。

#### ステップ 3. C ヘッダーを含める

- ヘッダーファイル「hexagon_delegate.h」は、[GitHub](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/hexagon/hexagon_delegate.h) からダウンロードするか、Hexagon デリゲート AAR から抽出できます。

#### ステップ 4. デリゲートを作成して TensorFlow Lite インタプリタを初期化する

- コードで、ネイティブの Hexagon ライブラリが読み込まれていることを確認します。これは、Activity または Java エントリポイントで`System.loadLibrary("tensorflowlite_hexagon_jni");` <br>を呼び出すことで実行できます。

- デレゲートを作成します。例：

```c
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"

// Assuming shared libraries are under "/data/local/tmp/"
// If files are packaged with native lib in android App then it
// will typically be equivalent to the path provided by
// "getContext().getApplicationInfo().nativeLibraryDir"
const char[] library_directory_path = "/data/local/tmp/";
TfLiteHexagonInitWithPath(library_directory_path);  // Needed once at startup.
::tflite::TfLiteHexagonDelegateOptions params = {0};
// 'delegate_ptr' Need to outlive the interpreter. For example,
// If use case will need to resize input or anything that can trigger
// re-applying delegates then 'delegate_ptr' need to outlive the interpreter.
auto* delegate_ptr = ::tflite::TfLiteHexagonDelegateCreate(&params);
Interpreter::TfLiteDelegatePtr delegate(delegate_ptr,
  [](TfLiteDelegate* delegate) {
    ::tflite::TfLiteHexagonDelegateDelete(delegate);
  });
interpreter->ModifyGraphWithDelegate(delegate.get());
// After usage of delegate.
TfLiteHexagonTearDown();  // Needed once at end of app/DSP usage.
```

## アプリに共有ライブラリを追加する

- 「app/src/main/jniLibs」フォルダを作成し、各ターゲットアーキテクチャのディレクトリを作成します。例：
    - ARM 64-bit: `app/src/main/jniLibs/arm64-v8a`
    - ARM 32-bit: `app/src/main/jniLibs/armeabi-v7a`
- アーキテクチャに一致するディレクトリに .so を配置します。

注意: アプリケーションの公開に App Bundle を使用している場合は、gradle.properties ファイルで android.bundle.enableUncompressedNativeLibs = false を設定することをお勧めします。

## フィードバック

問題がありましたら、[GitHub](https://github.com/tensorflow/tensorflow/issues/new?template=50-other-issues.md) issue を作成して使用するモバイルデバイスのモデルとボードなど、再現に必要な詳細をすべて記載してください (`adb shell getprop ro.product.device` および `adb shell getprop ro.board.platform`)。

## よくある質問

- デリゲートがサポートする演算は？
    - 最新の[サポートされている演算と制約](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/hexagon/README.md)のリストをご覧ください
- デリゲートが有効になっている場合、モデルが DSP を使用していることをどのように確認できますか？
    - デリゲートを有効にすると、以下の 2 つのログメッセージが出力されます。1 つはデリゲートが作成されたかどうかを示し、もう 1 つはデリゲートを使用して実行されているノードの数を示します。<br> `Created TensorFlow Lite delegate for Hexagon.` <br> `Hexagon delegate: X nodes delegated out of Y nodes.`
- デリゲートを実行するには、モデル内のすべての演算がサポートされる必要がありますか？
    - いいえ、モデルはサポートされる演算に基づいてサブグラフに分割されます。サポートされていない演算はすべて CPU で実行されます。
- ソースから Hexagon デリゲート AAR を構築するにはどうすればよいですか？
    - `bazel build -c opt --config=android_arm64 tensorflow/lite/delegates/hexagon/java:tensorflow-lite-hexagon`を使用します。
- Android デバイスにサポートされている SoC があるのに、Hexagon デリゲートが初期化に失敗するのはなぜですか？
    - デバイスに実際にサポートされている SoC があるかどうかを確認します。`adb shell cat /proc/cpuinfo | grep Hardware`を実行して、「Hardware：Qualcomm Technologies、Inc MSMXXXX」のような結果が返されるかどうかを確認します。
    - 一部のモバイルデバイスメーカーは、同一のモバイルモデルに異なる SoC を使用しているため、Hexagon デリゲートは、モバイルデバイスのモデルが同じでも、一部のデバイスでしか機能しない可能性があります。
    - 一部のモバイルデバイスメーカーは、システム以外の Android アプリからの Hexagon DSP の使用を意図的に制限しているため、Hexagon デリゲートが機能しないこともあります。
- モバイルデバイスでは DSP アクセスがロックされています。携帯電話を root しましたが、デリゲートを実行できません。どうすればよいですか？
    - `adb shell setenforce 0`を実行して、SELinux の強制を必ず無効にしてください。
