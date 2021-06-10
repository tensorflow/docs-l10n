# GPU で TensorFlow Lite を使用する

[TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/) では、複数のハードウェアアクセラレータがサポートされています。このドキュメントでは、Android（OpenCL または OpenGL ES 3.1 以上）および iOS（iOS 8 以上）で TensorFlow Lite デリゲート API を使用して GPU バックエンドを使用する方法について説明します。

## GPU アクセラレーションの利点

### 速度

GPU は、大規模に実行する並列化可能なワークロードで高い処理能力を実現するように設計されています。そのため、これは多数の演算で構成されるディープニューラルネットに適しています。各演算は、より小さなワークロードに簡単に分割でき、並列に実行する入力テンソルで機能するため、通常レイテンシが低くなります。現在、最良のシナリオでは、GPU での推論は以前は利用できなかったリアルタイムアプリケーションで十分に速く実行できます。

### 精度

GPU は、16 ビットまたは 32 ビットの浮動小数点数を使用して計算を行い、（CPU とは異なり）最適なパフォーマンスを得るために量子化を必要としません。精度の低下によりモデルでの量子化が不可能になる場合、GPU でニューラルネットワークを実行すると、この問題が解消される場合があります。

### 電力効率

GPU の推論のもう 1 つの利点は、電力効率です。GPU は非常に効率的かつ最適化された方法で計算を実行するため、同じタスクを CPU で実行する場合よりも消費電力と発熱が少なくなります。

## サポートされている演算

GPU では TensorFlow Lite は、16 ビットおよび 32 ビットの浮動小数点精度で次の演算をサポートします。

- `ADD`
- `AVERAGE_POOL_2D`
- `CONCATENATION`
- `CONV_2D`
- `DEPTHWISE_CONV_2D v1-2`
- `EXP`
- `FULLY_CONNECTED`
- `LOGISTIC`
- `LSTM v2 (Basic LSTM のみ)`
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
&lt;/div&gt;
&lt;pre data-md-type="block_code" data-md-language=""&gt;&lt;code&gt;GL_CODE_30&lt;/code&gt;</pre>
<div data-md-type="block_html">
</div>
</section></devsite-selector>
</div>
<h3 data-md-type="header" data-md-header-level="3">Android (C/C++)</h3>
<p data-md-type="paragraph">Android C/C++ 向け TensorFlow Lite GPU を使用する場合、GPU デリゲートは<code data-md-type="codespan">TfLiteGpuDelegateV2Create()</code>で作成し、<code data-md-type="codespan">TfLiteGpuDelegateV2Delete()</code>で破棄できます。</p>
<pre data-md-type="block_code" data-md-language="c++"><code class="language-c++">// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr&lt;Interpreter&gt; interpreter;
InterpreterBuilder(*model, op_resolver)(&amp;interpreter);

// NEW: Prepare GPU delegate.
auto* delegate = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
if (interpreter-&gt;ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter-&gt;typed_input_tensor&lt;float&gt;(0));
if (interpreter-&gt;Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter-&gt;typed_output_tensor&lt;float&gt;(0));

// NEW: Clean up.
TfLiteGpuDelegateV2Delete(delegate);</code></pre>
<p data-md-type="paragraph"><code data-md-type="codespan">TfLiteGpuDelegateOptionsV2</code>を見て、カスタムオプションを使用してデリゲートインスタンスを作成します。<code data-md-type="codespan">TfLiteGpuDelegateOptionsV2Default()</code>でデフォルトオプションを初期化し、必要に応じて変更します。</p>
<p data-md-type="paragraph">Android C/C++ 向け TFLite GPU では、<a href="https://bazel.io" data-md-type="link">Bazel</a> ビルドシステムを使用します。デリゲートは、次のコマンドなどを使用して構築できます。</p>
<pre data-md-type="block_code" data-md-language="sh"><code class="language-sh">bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:delegate                           # for static library
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so  # for dynamic library</code></pre>
<p data-md-type="paragraph">注意: <code data-md-type="codespan">Interpreter::ModifyGraphWithDelegate()</code>または<code data-md-type="codespan">Interpreter::Invoke()</code>を呼び出す場合、呼び出し元はその時点のスレッドに<code data-md-type="codespan">EGLContext</code>を持ち、<code data-md-type="codespan">Interpreter::Invoke()</code>は、同じ<code data-md-type="codespan">EGLContext</code>から呼び出す必要があります。<code data-md-type="codespan">EGLContext</code>が存在しない場合、デリゲートは内部的に作成しますが、開発者は<code data-md-type="codespan">Interpreter::Invoke()</code>が常に<code data-md-type="codespan">Interpreter::ModifyGraphWithDelegate()</code>を呼び出すスレッドと同じスレッドから呼び出されるようにする必要があります。</p>
<h3 data-md-type="header" data-md-header-level="3">iOS (C++)</h3>
<p data-md-type="paragraph">注意：Swift/Objective-C/C のユースケースについては、<a href="gpu#ios" data-md-type="link">GPU デリゲートガイド</a>を参照してください。</p>
<p data-md-type="paragraph">注意：これは、bazel を使用している場合、または TensorFlow Lite を自分でビルドしている場合にのみ使用できます。C++ API は CocoaPods では使用できません。</p>
<p data-md-type="paragraph">GPU で TensorFlow Lite を使用するには、<code data-md-type="codespan">TFLGpuDelegateCreate()</code>を介して GPU デリゲートを取得し、（<code data-md-type="codespan">Interpreter::AllocateTensors()</code>を呼び出す代わりに）それを<code data-md-type="codespan">Interpreter::ModifyGraphWithDelegate()</code>に渡します。</p>
<pre data-md-type="block_code" data-md-language="c++"><code class="language-c++">// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
tflite::ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr&lt;Interpreter&gt; interpreter;
InterpreterBuilder(*model, op_resolver)(&amp;interpreter);

// NEW: Prepare GPU delegate.

auto* delegate = TFLGpuDelegateCreate(/*default options=*/nullptr);
if (interpreter-&gt;ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter-&gt;typed_input_tensor&lt;float&gt;(0));
if (interpreter-&gt;Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter-&gt;typed_output_tensor&lt;float&gt;(0));

// Clean up.
TFLGpuDelegateDelete(delegate);
</code></pre>
<h2 data-md-type="header" data-md-header-level="2">高度な利用法</h2>
<h3 data-md-type="header" data-md-header-level="3">iOS のデリゲートオプション</h3>
<p data-md-type="paragraph">GPU デリゲートのコンストラクタは、オプションの<code data-md-type="codespan">struct</code>を受け入れます。(<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/swift/Sources/MetalDelegate.swift" data-md-type="link">Swift API</a>、<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/objc/apis/TFLMetalDelegate.h" data-md-type="link">Objective-C API</a>、<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/metal_delegate.h" data-md-type="link">C API</a>)</p>
<p data-md-type="paragraph"><code data-md-type="codespan">nullptr</code>（C API）を初期化子に渡すと、または初期化子に何も渡さないと（Objective-C と Swift API）、デフォルトのオプションが設定されます（上記の基本的な使用例で説明されています）。</p>
<div data-md-type="block_html">
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
&lt;/div&gt;
&lt;pre data-md-type="block_code" data-md-language=""&gt;&lt;code&gt;GL_CODE_51&lt;/code&gt;</pre>
<div data-md-type="block_html">
</div>
</section></devsite-selector>
</div>
<p data-md-type="paragraph"><code data-md-type="codespan">nullptr</code>またはデフォルトのコンストラクタを使用すると便利ですが、オプションを明示的に設定して、将来デフォルト値が変更された場合の予期しない動作を回避することをお勧めします。</p>
<h3 data-md-type="header" data-md-header-level="3">GPU で量子化モデルを実行する</h3>
<p data-md-type="paragraph">このセクションでは、GPU デリゲートが 8 ビットの量子化モデルを高速化する方法について説明します。以下のようなあらゆる種類の量子化が対象となります。</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">
<a href="https://www.tensorflow.org/lite/convert/quantization" data-md-type="link">量子化認識トレーニング</a>でトレーニングされたモデル</li>
<li data-md-type="list_item" data-md-list-type="unordered"><a href="https://www.tensorflow.org/lite/performance/post_training_quant" data-md-type="link">トレーニング後のダイナミックレンジ量子化</a></li>
<li data-md-type="list_item" data-md-list-type="unordered"><a href="https://www.tensorflow.org/lite/performance/post_training_integer_quant" data-md-type="link">トレーニング後の完全整数量子化</a></li>
</ul>
<p data-md-type="paragraph">パフォーマンスを最適化するには、浮動小数点入出力テンソルを持つモデルを使用します。</p>
<h4 data-md-type="header" data-md-header-level="4">仕組み</h4>
<p data-md-type="paragraph">GPU バックエンドは浮動小数点の実行のみをサポートするため、元のモデルの「浮動小数点ビュー」を与えて量子化モデルを実行します。上位レベルで、次のような手順が含まれます。</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="false">
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph"><em data-md-type="emphasis">定数テンソル</em>（重み/バイアスなど）は、GPU メモリに一度逆量子化されます。これは、デリゲートが TFLite Interpreter に適用されるときに発生します。</p>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">8 ビット量子化されている場合、GPU プログラムへの<em data-md-type="emphasis">入出力</em>は、推論ごとにそれぞれ逆量子化および量子化されます。これは、TFLite の最適化されたカーネルを使用して CPU 上で行われます。</p>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">GPU プログラムは、演算の間に<em data-md-type="emphasis">量子化シミュレータ</em>を挿入することにより、量子化された動作を模倣するように変更されます。これは、演算時にアクティベーションが量子化中に学習された境界に従うことが期待されるモデルに必要です。</p>
</li>
</ul>
<p data-md-type="paragraph">この機能は、次のデリゲートオプションを使用して有効にできます。</p>
<h4 data-md-type="header" data-md-header-level="4">Android</h4>
<p data-md-type="paragraph">Android API は、デフォルトで量子化モデルをサポートしています。無効にするには、次の手順に従います。</p>
<p data-md-type="paragraph"><strong data-md-type="double_emphasis">C++ API</strong></p>
<pre data-md-type="block_code" data-md-language="c++"><code class="language-c++">TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;

auto* delegate = TfLiteGpuDelegateV2Create(options);
if (interpreter-&gt;ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;
</code></pre>
<p data-md-type="paragraph"><strong data-md-type="double_emphasis">Java API</strong></p>
<pre data-md-type="block_code" data-md-language="java"><code class="language-java">GpuDelegate delegate = new GpuDelegate(new GpuDelegate.Options().setQuantizedModelsAllowed(false));

Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
</code></pre>
<h4 data-md-type="header" data-md-header-level="4">iOS</h4>
<p data-md-type="paragraph">iOS API は、デフォルトで量子化モデルをサポートしています。無効にするには、次の手順に従います。</p>
<div data-md-type="block_html">
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
&lt;/div&gt;
&lt;pre data-md-type="block_code" data-md-language=""&gt;&lt;code&gt;GL_CODE_55&lt;/code&gt;</pre>
<div data-md-type="block_html">
</div>
</section></devsite-selector>
</div>
<h3 data-md-type="header" data-md-header-level="3">入出力バッファ（iOS、C++ API のみ）</h3>
<p data-md-type="paragraph">注意：これは、bazel を使用している場合、または TensorFlow Lite を自分でビルドしている場合にのみ使用できます。C++ API は CocoaPods では使用できません。</p>
<p data-md-type="paragraph">GPU で計算を実行するには、データを GPU で使用できるようにする必要があり、多くの場合、メモリコピーの実行が必要になります。これにはかなり時間がかかる可能性があるため、可能であれば CPU/GPU のメモリ境界を超えないようにしてください。通常、このような交差は避けられませんが、一部の特殊なケースでは、どちらか一方を省略できます。</p>
<p data-md-type="paragraph">ネットワークの入力が GPU メモリに既に読み込まれている画像（たとえば、カメラフィードを含む GPU テクスチャ）である場合、CPU メモリに読み込むことなく、GPU メモリに保持できます。また、ネットワークの出力がレンダリング可能な画像（たとえば、<a href="https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf" data-md-type="link">画像スタイルの転送</a>）の形式である場合は、画面に直接表示できます。</p>
<p data-md-type="paragraph">TensorFlow Lite では、最高のパフォーマンスを実現するために、TensorFlow ハードウェアバッファから直接読み書きできるので、回避可能なメモリコピーをバイパスできます。</p>
<p data-md-type="paragraph">画像入力が GPU メモリにある場合、最初に Metal の<code data-md-type="codespan">MTLBuffer</code>オブジェクトに変換する必要があります。TfLiteTensor をユーザーが準備した<code data-md-type="codespan">MTLBuffer</code>に<code data-md-type="codespan">TFLGpuDelegateBindMetalBufferToTensor()</code>を関連付けることができます。<code data-md-type="codespan">TFLGpuDelegateBindMetalBufferToTensor()</code>は、<code data-md-type="codespan">Interpreter::ModifyGraphWithDelegate()</code>の後に呼び出す必要があることに注意してください。さらに、推論出力はデフォルトで、GPU メモリから CPU メモリにコピーされます。この動作は、初期化中に<code data-md-type="codespan">Interpreter::SetAllowBufferHandleOutput(true)</code>を呼び出すことで無効にできます。</p>
<pre data-md-type="block_code" data-md-language="c++"><code class="language-c++">#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
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
if (interpreter-&gt;Invoke() != kTfLiteOk) return false;</code></pre>
<p data-md-type="paragraph">注意: デフォルトの動作が無効になっている場合、GPU メモリから CPU メモリに推論出力をコピーするには、各出力テンソルに対して<code data-md-type="codespan">Interpreter::EnsureTensorDataIsReadable()</code>を明示的に呼び出す必要があります。</p>
<p data-md-type="paragraph">注意: これは量子化モデルでも機能しますが、バッファは内部の逆量子化バッファにバインドされるため、<strong data-md-type="double_emphasis">float32 データを含む float32 サイズのバッファ</strong>が必要です。</p>
<h2 data-md-type="header" data-md-header-level="2">ヒントとコツ</h2>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="false">
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">演算によっては CPU では簡単で GPU ではコストが高くなる可能性があります。このような演算の 1 つのクラスは、<code data-md-type="codespan">BATCH_TO_SPACE</code>、<code data-md-type="codespan">SPACE_TO_BATCH</code>、<code data-md-type="codespan">SPACE_TO_DEPTH</code>など、さまざまな形の変形演算です。ネットワークアーキテクトの論理的思考のためだけにこれらの演算がネットワークに挿入されている場合、パフォーマンスのためにそれらを削除することをお勧めします。</p>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">GPU では、テンソルデータは 4 チャネルにスライスされます。したがって、形状<code data-md-type="codespan">[B,H,W,5]</code>のテンソルに対する計算は、形状<code data-md-type="codespan">[B,H,W,8]</code>のテンソルに対する計算とほぼ同じように実行されますが、パフォーマンスは<code data-md-type="codespan">[B,H,W,4]</code>と比べて大幅に低下します。</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">たとえば、カメラハードウェアが RGBA の画像フレームをサポートしている場合、メモリコピー (3 チャネル RGB から 4 チャネル RGBX へ) を回避できるため、4 チャネル入力のフィードは大幅に速くなります。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">最高のパフォーマンスを得るには、モバイル向けに最適化されたネットワークアーキテクチャで分類器を再トレーニングします。これは、デバイス上の推論の最適化の重要な部分です。</p>
</li>
</ul>
</div>
</div>
