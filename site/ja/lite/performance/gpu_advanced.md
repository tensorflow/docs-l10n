# GPU で TensorFlow Lite を使用する

[TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/) では、複数のハードウェアアクセラレータがサポートされています。このドキュメントでは、Android（OpenCL または OpenGL ES 3.1 以上）および iOS（iOS 8 以上）で TensorFlow Lite デリゲート API を使用して GPU バックエンドを使用する方法について説明します。

## GPU アクセラレーションの利点

### 速度

GPU は、大規模に実行する並列化可能なワークロードで高い処理能力を実現するように設計されています。そのため、これは多数の演算で構成されるディープニューラルネットに適しています。各演算は、より小さなワークロードに簡単に分割でき、並列に実行する入力テンソルで機能するため、通常レイテンシが低くなります。現在、最良のシナリオでは、GPU での推論は以前は利用できなかったリアルタイムアプリケーションで十分に速く実行できます。

### 精度

GPU は、16 ビットまたは 32 ビットの浮動小数点数を使用して計算を行い、（CPU とは異なり）最適なパフォーマンスを得るために量子化を必要としません。精度の低下によりモデルでの量子化が不可能になる場合、GPU でニューラルネットワークを実行すると、この問題が解消される場合があります。

### 電力効率

GPU の推論のもう 1 つの利点は、電力効率です。GPU は非常に効率的かつ最適化された方法で計算を実行するため、同じタスクを CPU で実行する場合よりも消費電力と発熱が少なくなります。

## サポートする演算子

GPU では TensorFlow Lite は、16 ビットおよび 32 ビットの浮動小数点精度で次の演算をサポートします。

- `ADD`
- `AVERAGE_POOL_2D`
- `CONCATENATION`
- `CONV_2D`
- `DEPTHWISE_CONV_2D v1-2`
- `EXP`
- `FULLY_CONNECTED`
- `LOGISTIC`
- `LSTM v2 (Basic LSTM only)`
- `MAX_POOL_2D`
- `MAXIMUM`
- `MINIMUM`
- `MUL`
- `PAD`
- `PRELU`
- `RELU`
- `RELU6`
- `RESHAPE`
- `RESIZE_BILINEAR v1-3`
- `SOFTMAX`
- `STRIDED_SLICE`
- `SUB`
- `TRANSPOSE_CONV`

デフォルトでは、すべての演算はバージョン 1 でのみサポートされています。[実験的な量子化サポート](gpu_advanced.md#running-quantized-models-experimental-android-only)を有効にすると、適切なバージョンが許可されます (ADD v2 など)。

## 基本的な使い方

Android でモデルアクセラレーションを呼び出す方法は 2 つありますが、[Android Studio ML Model Binding](../inference_with_metadata/codegen#acceleration) または TensorFlow Lite インタープリタを使用しているかによって、方法は異なります。

### TensorFlow Lite Interpreter を使用して Android でモデルアクセラレーションを呼び出す

`tensorflow-lite-gpu`パッケージを既存の`tensorflow-lite`パッケージと共に、既存の`dependencies`ブロックに追加します。

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite:2.3.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.3.0'
}
```

次に、`TfLiteDelegate`を使用して GPU で TensorFlow Lite を実行します。Java では、`Interpreter.Options`から`GpuDelegate`を指定できます。

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">    import org.tensorflow.lite.Interpreter
    import org.tensorflow.lite.gpu.CompatibilityList
    import org.tensorflow.lite.gpu.GpuDelegate

    val compatList = CompatibilityList()

    val options = Interpreter.Options().apply{
        if(compatList.isDelegateSupportedOnThisDevice){
            // if the device has a supported GPU, add the GPU delegate
            val delegateOptions = compatList.bestOptionsForThisDevice
            this.addDelegate(GpuDelegate(delegateOptions))
        } else {
            // if the GPU is not supported, run on 4 threads
            this.setNumThreads(4)
        }
    }

    val interpreter = Interpreter(model, options)

    // Run inference
    writeToInput(input)
    interpreter.run(input, output)
    readFromOutput(output)
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    import org.tensorflow.lite.Interpreter;
    import org.tensorflow.lite.gpu.CompatibilityList;
    import org.tensorflow.lite.gpu.GpuDelegate;

    // Initialize interpreter with GPU delegate
    Interpreter.Options options = new Interpreter.Options();
    CompatibilityList compatList = CompatibilityList();

    if(compatList.isDelegateSupportedOnThisDevice()){
        // if the device has a supported GPU, add the GPU delegate
        GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
        GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
        options.addDelegate(gpuDelegate);
    } else {
        // if the GPU is not supported, run on 4 threads
        options.setNumThreads(4);
    }

    Interpreter interpreter = new Interpreter(model, options);

    // Run inference
    writeToInput(input);
    interpreter.run(input, output);
    readFromOutput(output);
      </pre>
    </section>
  </devsite-selector>
</div>

### Android (C/C++)

Android C/C++ 向け TensorFlow Lite GPU を使用する場合、GPU デリゲートは`TfLiteGpuDelegateV2Create()`で作成し、`TfLiteGpuDelegateV2Delete()`で破棄できます。

```c++
// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr<Interpreter> interpreter;
InterpreterBuilder(*model, op_resolver)(&interpreter);

// NEW: Prepare GPU delegate.
auto* delegate = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
if (interpreter->Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));

// NEW: Clean up.
TfLiteGpuDelegateV2Delete(delegate);
```

`TfLiteGpuDelegateOptionsV2`を見て、カスタムオプションを使用してデリゲートインスタンスを作成します。`TfLiteGpuDelegateOptionsV2Default()`でデフォルトオプションを初期化し、必要に応じて変更します。

Android C/C++ 向け TFLite GPU では、[Bazel](https://bazel.io) ビルドシステムを使用します。デリゲートは、次のコマンドなどを使用して構築できます。

```sh
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:delegate                           # for static library
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so  # for dynamic library
```

注意: `Interpreter::ModifyGraphWithDelegate()`または`Interpreter::Invoke()`を呼び出す場合、呼び出し元はその時点のスレッドに`EGLContext`を持ち、`Interpreter::Invoke()`は、同じ`EGLContext`から呼び出す必要があります。`EGLContext`が存在しない場合、デリゲートは内部的に作成しますが、開発者は`Interpreter::Invoke()`が常に`Interpreter::ModifyGraphWithDelegate()`を呼び出すスレッドと同じスレッドから呼び出されるようにする必要があります。

### iOS (C++)

注意：Swift/Objective-C/C のユースケースについては、[GPU デリゲートガイド](gpu#ios)を参照してください。

注意：これは、bazel を使用している場合、または TensorFlow Lite を自分でビルドしている場合にのみ使用できます。C++ API は CocoaPods では使用できません。

GPU で TensorFlow Lite を使用するには、`TFLGpuDelegateCreate()`を介して GPU デリゲートを取得し、（`Interpreter::AllocateTensors()`を呼び出す代わりに）それを`Interpreter::ModifyGraphWithDelegate()`に渡します。

```c++
// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
tflite::ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr<Interpreter> interpreter;
InterpreterBuilder(*model, op_resolver)(&interpreter);

// NEW: Prepare GPU delegate.

auto* delegate = TFLGpuDelegateCreate(/*default options=*/nullptr);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
if (interpreter->Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));

// Clean up.
TFLGpuDelegateDelete(delegate);
```

## 高度な利用法

### iOS のデリゲートオプション

GPU デリゲートのコンストラクタは、オプションの`struct`を受け入れます。([Swift API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/swift/Sources/MetalDelegate.swift)、[Objective-C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/objc/apis/TFLMetalDelegate.h)、[C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/metal_delegate.h))

`nullptr`（C API）を初期化子に渡すと、または初期化子に何も渡さないと（Objective-C と Swift API）、デフォルトのオプションが設定されます（上記の基本的な使用例で説明されています）。

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    // THIS:
    var options = MetalDelegate.Options()
    options.isPrecisionLossAllowed = false
    options.waitType = .passive
    options.isQuantizationEnabled = true
    let delegate = MetalDelegate(options: options)

    // IS THE SAME AS THIS:
    let delegate = MetalDelegate()
      </pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">    // THIS:
    TFLMetalDelegateOptions* options = [[TFLMetalDelegateOptions alloc] init];
    options.precisionLossAllowed = false;
    options.waitType = TFLMetalDelegateThreadWaitTypePassive;
    options.quantizationEnabled = true;

    TFLMetalDelegate* delegate = [[TFLMetalDelegate alloc] initWithOptions:options];

    // IS THE SAME AS THIS:
    TFLMetalDelegate* delegate = [[TFLMetalDelegate alloc] init];
      </pre>
    </section>
    <section>
      <h3>C</h3>
      <p></p>
<pre class="prettyprint lang-c">    // THIS:
    const TFLGpuDelegateOptions options = {
      .allow_precision_loss = false,
      .wait_type = TFLGpuDelegateWaitType::TFLGpuDelegateWaitTypePassive,
      .enable_quantization = true,
    };

    TfLiteDelegate* delegate = TFLGpuDelegateCreate(options);

    // IS THE SAME AS THIS:
    TfLiteDelegate* delegate = TFLGpuDelegateCreate(nullptr);
      </pre>
    </section>
  </devsite-selector>
</div>

`nullptr`またはデフォルトのコンストラクタを使用すると便利ですが、オプションを明示的に設定して、将来デフォルト値が変更された場合の予期しない動作を回避することをお勧めします。

### GPU で量子化モデルを実行する

このセクションでは、GPU デリゲートが 8 ビットの量子化モデルを高速化する方法について説明します。以下のようなあらゆる種類の量子化が対象となります。

- [量子化認識トレーニング](https://www.tensorflow.org/lite/convert/quantization)でトレーニングされたモデル
- [Post-training dynamic-range quantization](https://www.tensorflow.org/lite/performance/post_training_quant)
- [Post-training full-integer quantization](https://www.tensorflow.org/lite/performance/post_training_integer_quant)

パフォーマンスを最適化するには、浮動小数点入出力テンソルを持つモデルを使用します。

#### 仕組み

GPU バックエンドは浮動小数点の実行のみをサポートするため、元のモデルの「浮動小数点ビュー」を与えて量子化モデルを実行します。上位レベルで、次のような手順が含まれます。

- *定数テンソル*（重み/バイアスなど）は、GPU メモリに一度逆量子化されます。これは、デリゲートが TFLite Interpreter に適用されるときに発生します。

- 8 ビット量子化されている場合、GPU プログラムへの*入出力*は、推論ごとにそれぞれ逆量子化および量子化されます。これは、TFLite の最適化されたカーネルを使用して CPU 上で行われます。

- GPU プログラムは、演算の間に*量子化シミュレータ*を挿入することにより、量子化された動作を模倣するように変更されます。これは、演算時にアクティベーションが量子化中に学習された境界に従うことが期待されるモデルに必要です。

この機能は、次のデリゲートオプションを使用して有効にできます。

#### Android

Android API は、デフォルトで量子化モデルをサポートしています。無効にするには、次の手順に従います。

**C++ API**

```c++
TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;

auto* delegate = TfLiteGpuDelegateV2Create(options);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;
```

**Java API**

```java
GpuDelegate delegate = new GpuDelegate(new GpuDelegate.Options().setQuantizedModelsAllowed(false));

Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
```

#### iOS

iOS API は、デフォルトで量子化モデルをサポートしています。無効にするには、次の手順に従います。

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    var options = MetalDelegate.Options()
    options.isQuantizationEnabled = false
    let delegate = MetalDelegate(options: options)
      </pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">    TFLMetalDelegateOptions* options = [[TFLMetalDelegateOptions alloc] init];
    options.quantizationEnabled = false;
      </pre>
    </section>
    <section>
      <h3>C</h3>
      <p></p>
<pre class="prettyprint lang-c">    TFLGpuDelegateOptions options = TFLGpuDelegateOptionsDefault();
    options.enable_quantization = false;

    TfLiteDelegate* delegate = TFLGpuDelegateCreate(options);
      </pre>
    </section>
  </devsite-selector>
</div>

### 入出力バッファ（iOS、C++ API のみ）

注意：これは、bazel を使用している場合、または TensorFlow Lite を自分でビルドしている場合にのみ使用できます。C++ API は CocoaPods では使用できません。

GPU で計算を実行するには、データを GPU で使用できるようにする必要があり、多くの場合、メモリコピーの実行が必要になります。これにはかなり時間がかかる可能性があるため、可能であれば CPU/GPU のメモリ境界を超えないようにしてください。通常、このような交差は避けられませんが、一部の特殊なケースでは、どちらか一方を省略できます。

If the network's input is an image already loaded in the GPU memory (for example, a GPU texture containing the camera feed) it can stay in the GPU memory without ever entering the CPU memory. Similarly, if the network's output is in the form of a renderable image (for example, [image style transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)) it can be directly displayed on the screen.

TensorFlow Lite では、最高のパフォーマンスを実現するために、TensorFlow ハードウェアバッファから直接読み書きできるので、回避可能なメモリコピーをバイパスできます。

画像入力が GPU メモリにある場合、最初に Metal の`MTLBuffer`オブジェクトに変換する必要があります。TfLiteTensor をユーザーが準備した`MTLBuffer`に`TFLGpuDelegateBindMetalBufferToTensor()`を関連付けることができます。`TFLGpuDelegateBindMetalBufferToTensor()`は、`Interpreter::ModifyGraphWithDelegate()`の後に呼び出す必要があることに注意してください。さらに、推論出力はデフォルトで、GPU メモリから CPU メモリにコピーされます。この動作は、初期化中に`Interpreter::SetAllowBufferHandleOutput(true)`を呼び出すことで無効にできます。

```c++
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#include "tensorflow/lite/delegates/gpu/metal_delegate_internal.h"

// ...

// Prepare GPU delegate.
auto* delegate = TFLGpuDelegateCreate(nullptr);

if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

interpreter->SetAllowBufferHandleOutput(true);  // disable default gpu->cpu copy
if (!TFLGpuDelegateBindMetalBufferToTensor(
        delegate, interpreter->inputs()[0], user_provided_input_buffer)) {
  return false;
}
if (!TFLGpuDelegateBindMetalBufferToTensor(
        delegate, interpreter->outputs()[0], user_provided_output_buffer)) {
  return false;
}

// Run inference.
if (interpreter->Invoke() != kTfLiteOk) return false;
```

注意: デフォルトの動作が無効になっている場合、GPU メモリから CPU メモリに推論出力をコピーするには、各出力テンソルに対して`Interpreter::EnsureTensorDataIsReadable()`を明示的に呼び出す必要があります。

注意: これは量子化モデルでも機能しますが、バッファは内部の逆量子化バッファにバインドされるため、**float32 データを含む float32 サイズのバッファ**が必要です。

### GPU デリゲートのシリアル化

前の初期化からの GPU カーネルコードとモデルデータのシリアル化を使用すると、GPU デリゲートの初期化のレイテンシーを 90% まで抑えることができます。この改善は、時間を節約するためにディスク容量を交換することで達成されます。この機能は、以下のサンプルコードで示されるように、いくつかの構成オプションで有効にすることができます。

<div>
  <devsite-selector>
    <section>
      <h3>C++</h3>
      <p></p>
<pre class="prettyprint lang-cpp">    TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
    options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
    options.serialization_dir = kTmpDir;
    options.model_token = kModelToken;

    auto* delegate = TfLiteGpuDelegateV2Create(options);
    if (interpreter-&gt;ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    GpuDelegate delegate = new GpuDelegate(
      new GpuDelegate.Options().setSerializationParams(
        /* serializationDir= */ serializationDir,
        /* modelToken= */ modelToken));

    Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
      </pre>
    </section>
  </devsite-selector>
</div>

シリアル化機能を使用する場合、コードが以下の実装ルールでコンパイルすることを確認してください。

- シリアル化データを他のアプリがアクセスできないディレクトリに保存します。Android デバイスでは、現在のアプリケーションに非公開の場所にポイントする [`getCodeCacheDir()`](https://developer.android.com/reference/android/content/Context#getCacheDir()) を使用します。
- モデルトークンは、特定のモデルのデバイスに一意である必要があります。モデルトークンは、モデルデータからフィンガープリントを生成することで計算できます（[`farmhash::Fingerprint64`](https://github.com/google/farmhash) を使用するなど）。

注意: この機能には、シリアル化サポートを提供する [OpenCL SDK](https://github.com/KhronosGroup/OpenCL-SDK) が必要です。

## ヒントとコツ

- 演算によっては CPU では簡単で GPU ではコストが高くなる可能性があります。このような演算の 1 つのクラスは、`BATCH_TO_SPACE`、`SPACE_TO_BATCH`、`SPACE_TO_DEPTH`など、さまざまな形の変形演算です。ネットワークアーキテクトの論理的思考のためだけにこれらの演算がネットワークに挿入されている場合、パフォーマンスのためにそれらを削除することをお勧めします。

- GPU では、テンソルデータは 4 チャネルにスライスされます。したがって、形状`[B,H,W,5]`のテンソルに対する計算は、形状`[B,H,W,8]`のテンソルに対する計算とほぼ同じように実行されますが、パフォーマンスは`[B,H,W,4]`と比べて大幅に低下します。

    - たとえば、カメラハードウェアが RGBA の画像フレームをサポートしている場合、メモリコピー (3 チャネル RGB から 4 チャネル RGBX へ) を回避できるため、4 チャネル入力のフィードは大幅に速くなります。

- 最高のパフォーマンスを得るには、モバイル向けに最適化されたネットワークアーキテクチャで分類器を再トレーニングします。これは、デバイス上の推論の最適化の重要な部分です。
