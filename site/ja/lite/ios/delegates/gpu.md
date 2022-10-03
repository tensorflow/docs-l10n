# iOS の GPU アクセラレーションデリゲート

グラフィックス処理装置（GPU）を使用して機械学習（ML）モデルを実行すると、モデルのパフォーマンスと ML 対応アプリケーションのユーザーエクスペリエンスが大幅に向上します。iOS デバイスでは、[*デリゲート*](../../performance/delegates)を使用して GPU で高速化されたモデルの実行を有効にできます。デリゲートは TensorFlow Lite のハードウェアドライバーとして機能し、モデルのコードを GPU プロセッサで実行できるようにします。

このページでは、iOS アプリで TensorFlow Lite モデルの GPU アクセラレーションを有効にする方法について説明します。ベストプラクティスや高度な手法など、TensorFlow Lite の GPU デリゲートの使用に関する詳細については、[GPU デリゲート](../../performance/gpu)のページを参照してください。

## Interpreter API による GPU の使用

TensorFlow Lite [Interpreter API](../../api_docs/swift/Classes/Interpreter) は、機械学習アプリケーションをビルドするための一連の汎用 API を提供します。次の手順では、GPU サポートを iOS アプリに追加する方法について説明します。このガイドは、TensorFlow Lite で ML モデルを正常に実行できる iOS アプリが既にあることを前提としています。

注意: TensorFlow Lite を使用する iOS アプリをまだお持ちでない場合は、[iOS クイックスタート](https://www.tensorflow.org/lite/guide/ios)に従ってデモアプリをビルドしてください。チュートリアルを完了したら、これらの手順に従って GPU サポートを有効にできます。

### Podfile を変更して GPU サポートを追加

TensorFlow Lite 2.3.0 リリース以降は、バイナリサイズを減らすために GPU デリゲートがポッドから除外されていますが、`TensorFlowLiteSwift` ポッドのサブスペックを指定して含めることができます。

```ruby
pod 'TensorFlowLiteSwift/Metal', '~> 0.0.1-nightly',
```

または

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly', :subspecs => ['Metal']
```

Objective-C（2.4.0 リリース以降）または C API を使用する場合は、`TensorFlowLiteObjC` または `TensorFlowLitC` を使用することもできます。

注意: TensorFlow Lite バージョン 2.1.0 から 2.2.0 まででは、GPU デリゲートは <code>TensorFlowLiteC</code> ポッド<em>に含まれています</em>。使用するプログラミング言語に応じて、 `TensorFlowLiteC` と `TensorFlowLiteSwift` のいずれかを選択できます。

### GPU デリゲートの初期化と使用

TensorFlow Lite [Interpreter API](../../api_docs/swift/Classes/Interpreter) で GPU デリゲートを使用すると、多くのプログラミング言語を使用できます。Swift と Objective-C が推奨されますが、C++ と C も使用できます。TensorFlow Lite の 2.4 より前のバージョンを使用している場合は、C を使用する必要があります。次のコード例は、これらの各言語でデリゲートを使用する方法の概要を示しています。

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">import TensorFlowLite

// Load model ...

// Initialize TensorFlow Lite interpreter with the GPU delegate.
let delegate = MetalDelegate()
if let interpreter = try Interpreter(modelPath: modelPath,
                                      delegates: [delegate]) {
  // Run inference ...
}
      </pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">// Import module when using CocoaPods with module support
@import TFLTensorFlowLite;

// Or import following headers manually
#import "tensorflow/lite/objc/apis/TFLMetalDelegate.h"
#import "tensorflow/lite/objc/apis/TFLTensorFlowLite.h"

// Initialize GPU delegate
TFLMetalDelegate* metalDelegate = [[TFLMetalDelegate alloc] init];

// Initialize interpreter with model path and GPU delegate
TFLInterpreterOptions* options = [[TFLInterpreterOptions alloc] init];
NSError* error = nil;
TFLInterpreter* interpreter = [[TFLInterpreter alloc]
                                initWithModelPath:modelPath
                                          options:options
                                        delegates:@[ metalDelegate ]
                                            error:&amp;error];
if (error != nil) { /* Error handling... */ }

if (![interpreter allocateTensorsWithError:&amp;error]) { /* Error handling... */ }
if (error != nil) { /* Error handling... */ }

// Run inference ...
      </pre>
    </section>
    <section>
      <h3>C++</h3>
      <p></p>
<pre class="prettyprint lang-cpp">// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
tflite::ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr&lt;Interpreter&gt; interpreter;
InterpreterBuilder(*model, op_resolver)(&amp;interpreter);

// Prepare GPU delegate.
auto* delegate = TFLGpuDelegateCreate(/*default options=*/nullptr);
if (interpreter-&gt;ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter-&gt;typed_input_tensor&lt;float&gt;(0));
if (interpreter-&gt;Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter-&gt;typed_output_tensor&lt;float&gt;(0));

// Clean up.
TFLGpuDelegateDelete(delegate);
      </pre>
    </section>
    <section>
      <h3>C（2.4.0 より前）</h3>
      <p></p>
<pre class="prettyprint lang-c">#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"

// Initialize model
TfLiteModel* model = TfLiteModelCreateFromFile(model_path);

// Initialize interpreter with GPU delegate
TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
TfLiteDelegate* delegate = TFLGPUDelegateCreate(nil);  // default config
TfLiteInterpreterOptionsAddDelegate(options, metal_delegate);
TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
TfLiteInterpreterOptionsDelete(options);

TfLiteInterpreterAllocateTensors(interpreter);

NSMutableData *input_data = [NSMutableData dataWithLength:input_size * sizeof(float)];
NSMutableData *output_data = [NSMutableData dataWithLength:output_size * sizeof(float)];
TfLiteTensor* input = TfLiteInterpreterGetInputTensor(interpreter, 0);
const TfLiteTensor* output = TfLiteInterpreterGetOutputTensor(interpreter, 0);

// Run inference
TfLiteTensorCopyFromBuffer(input, inputData.bytes, inputData.length);
TfLiteInterpreterInvoke(interpreter);
TfLiteTensorCopyToBuffer(output, outputData.mutableBytes, outputData.length);

// Clean up
TfLiteInterpreterDelete(interpreter);
TFLGpuDelegateDelete(metal_delegate);
TfLiteModelDelete(model);
      </pre>
    </section>
  </devsite-selector>
</div>

#### GPU API 言語の使用上の注意

- 2.4.0 より前の TensorFlow Lite バージョンでは、Objective-C 用の C API のみを使用できます。
- C++ API は、bazel を使用している場合、または TensorFlow Lite を自分でビルドしている場合にのみ使用できます。C++ API は CocoaPods では使用できません。
- C++ を使用した GPU デリゲートで TensorFlow Lite を使用する場合、`TFLGpuDelegateCreate()` 関数を介して GPU デリゲートを取得し、`Interpreter::AllocateTensors()` を呼び出す代わりに、`Interpreter::ModifyGraphWithDelegate()` に渡します。

### リリースモードでのビルドとテスト

適切な Metal API アクセラレータ設定でリリースビルドに変更して、パフォーマンスを向上させ、最終テストを行います。このセクションでは、リリースビルドを有効にして、Metal アクセラレーションの設定を構成する方法について説明します。

注意: これらの手順には、XCode v10.1 以降が必要です。

リリースビルドに変更するには:

1. **[Product] &gt; [Scheme] &gt; [Edit Scheme...]** を選択し、[**Run**] を選択して、ビルド設定を編集します。
2. [**Info**] タブで、[**Build Configuration**] を [**Release**] に変更し、 [**Debug executable**] のチェックを外します。![リリースの設定](../../../images/lite/ios/iosdebug.png)
3. [**Options**] タブをクリックし、 [**GPU Frame Capture**] を [**Disabled**] に変更し、[**Metal API Validation**] を [**Disabled**] に変更します。 <br>![メタルオプションの設定](../../../images/lite/ios/iosmetal.png)
4. 64 ビットアーキテクチャのリリース専用ビルドを選択してください。**[Project navigator] &gt; [tflite_camera_example] &gt; [PROJECT] &gt; [your_project_name] &gt; [Build Settings]** で **[Build Active Architecture Only] &gt; [Release]** を **[Yes]** に設定します。![リリース オプションの設定](../../../images/lite/ios/iosrelease.png)

## 高度な GPU サポート

このセクションでは、デリゲートオプション、入力および出力バッファー、量子化モデルの使用など、iOS の GPU デリゲートの高度な使用法について説明します。

### iOS のデリゲートオプション

GPU デリゲートのコンストラクターは、[Swift API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/swift/Sources/MetalDelegate.swift)、[Objective-C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/objc/apis/TFLMetalDelegate.h)、[C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/metal_delegate.h) のオプションの `struct` を受け入れます。初期化子に `nullptr` を渡す（C API）か、または何も渡さない（Objective-C および Swift API）ことにより、デフォルトオプション（上記の基本的な使用例で説明されています）が設定されます。

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">// THIS:
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
<pre class="prettyprint lang-objc">// THIS:
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
<pre class="prettyprint lang-c">// THIS:
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

ヒント: `nullptr` またはデフォルトコンストラクタを使用すると便利ですが、将来デフォルト値が変更された場合に予期しない動作が発生しないように、オプションを明示的に設定する必要があります。

### C++ API を使用した入出力バッファー

GPU での計算では、データが GPU で利用可能である必要があります。この要件は、多くの場合、メモリコピーを実行する必要があることを意味します。かなりの時間がかかる可能性があるため、データが CPU/GPU メモリの境界を超えることは可能な限り避ける必要があります。通常、このような交差は避けられませんが、特殊なケースではどちらか一方を省略できます。

注意: 次の手法は、Bazel を使用している場合、または TensorFlow Lite を自分でビルドしている場合にのみ使用できます。C++ API は CocoaPods では使用できません。

ネットワークの入力が GPU メモリに既に読み込まれている画像（たとえば、カメラフィードを含む GPU テクスチャ）である場合、CPU メモリに読み込むことなく、GPU メモリに保持できます。また、ネットワークの出力がレンダリング可能な画像（たとえば、[画像スタイルの転送](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)）の形式である場合は、画面に直接表示できます。

TensorFlow Lite では、最高のパフォーマンスを実現するために、TensorFlow ハードウェアバッファから直接読み書きできるので、回避可能なメモリコピーをバイパスできます。

画像入力が GPU メモリにある場合、最初に Metal の `MTLBuffer` オブジェクトに変換する必要があります。`TFLGpuDelegateBindMetalBufferToTensor()`関数を使用して、`MTLBuffer` をユーザーが準備した `TfLiteTensor` に関連付けることができます。この関数は `Interpreter::ModifyGraphWithDelegate()` 後に呼び出す*必要*があることに注意してください。さらに、推論出力はデフォルトで GPU メモリから CPU メモリにコピーされます。初期化中に `Interpreter::SetAllowBufferHandleOutput(true)` を呼び出すことで、この動作を無効にできます。

<div>
  <devsite-selector>
    <section>
      <h3>C++</h3>
      <p></p>
<pre class="prettyprint lang-swift">#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#include "tensorflow/lite/delegates/gpu/metal_delegate_internal.h"

// ...

// Prepare GPU delegate.
auto* delegate = TFLGpuDelegateCreate(nullptr);

if (interpreter-&gt;ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

interpreter-&gt;SetAllowBufferHandleOutput(true);  // disable default gpu-&gt;cpu copy
if (!TFLGpuDelegateBindMetalBufferToTensor(
        delegate, interpreter-&gt;inputs()[0], user_provided_input_buffer)) {
  return false;
}
if (!TFLGpuDelegateBindMetalBufferToTensor(
        delegate, interpreter-&gt;outputs()[0], user_provided_output_buffer)) {
  return false;
}

// Run inference.
if (interpreter-&gt;Invoke() != kTfLiteOk) return false;
      </pre>
    </section>
  </devsite-selector>
</div>

デフォルトの動作を無効にすると、推論出力を GPU メモリから CPU メモリにコピーするには、出力テンソルごとに `Interpreter::EnsureTensorDataIsReadable()` を明示的に呼び出す必要があります。このアプローチは量子化されたモデルでも機能しますが、バッファーが内部の非量子化バッファーにバインドされているため、**float32 データで float32 サイズのバッファー**を使用する必要があります。

### 量子化モデル {:#quantized-models}

iOS GPU デリゲートライブラリは、*デフォルトで量子化モデルをサポートします*。 GPU デリゲートで量子化モデルを使用するためにコードを変更する必要はありません。次のセクションでは、テストまたは実験目的で量子化サポートを無効にする方法について説明します。

#### 量子化モデルのサポートの無効化

次のコードは、量子化されたモデルのサポートを***無効***にする方法を示しています。

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

GPU アクセラレーションを使用した量子化モデルの実行の詳細については、[GPU デリゲート](../../performance/gpu#quantized-models)の概要を参照してください。
