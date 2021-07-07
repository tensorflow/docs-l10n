# GPU 上的 TensorFlow Lite

[TensorFlow Lite](https://tensorflow.google.cn/mobile/tflite/) 支持多种硬件加速器。本文档介绍如何通过 TensorFlow Lite 委托 API 在 Android（要求 OpenCL 或者 OpenGL ES 3.1 及更高版本）和 iOS（要求 iOS 8 或更高版本）上使用 GPU 后端。

## GPU 加速的优势

### 速度

GPU 采用高吞吐量式设计，可处理大规模可并行化的工作负载。因此，它们非常适合包含大量算子的深度神经网络，每个算子都会处理一个或多个输入张量，可以轻松地划分为较小的工作负载且并行执行，这通常可以降低延迟。在最佳情况下，GPU 上的推断速度已足够快，适用于以前无法实现的实时应用。

### 准确率

GPU 使用 16 位或 32 位浮点数进行计算，并且（与 CPU 不同）不需要量化即可获得最佳性能。如果准确率降低使模型的量化无法达到要求，那么在 GPU 上运行神经网络可以消除这种担忧。

### 能效

GPU 推断的另一个优势是其功效。GPU 以非常高效且经优化的方式执行计算，因此与在 CPU 上执行相同任务时相比，功耗和产生的热量更低。

## 支持的运算

GPU 上的 TensorFlow Lite 支持 16 位和 32 位浮点精度的以下运算：

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

默认情况下，只有版本 1 支持所有运算。启用[实验性量化支持](gpu_advanced.md#running-quantized-models-experimental-android-only)可以允许相应的版本，例如 ADD v2。

## 基本用法

您可以通过两种方式调用模型加速，具体取决于您使用的是 [Android Studio 机器学习模型绑定](../inference_with_metadata/codegen#acceleration)还是 TensorFlow Lite 解释器。

### 通过 TensorFlow Lite 解释器

将 `tensorflow-lite-gpu` 软件包与现有 `tensorflow-lite` 软件包一起添加到现有 `dependencies` 块中。

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite:2.3.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.3.0'
}
```

然后，使用 `TfLiteDelegate` 在 GPU 上运行 TensorFlow Lite。在 Java 中，您可以通过 `Interpreter.Options` 指定 `GpuDelegate`。

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
<p data-md-type="paragraph">对于 Android 上 TensorFlow Lite GPU 的 C/C++ 用法，可以使用 <code data-md-type="codespan">TfLiteGpuDelegateV2Create()</code> 创建 GPU 委托，使用 <code data-md-type="codespan">TfLiteGpuDelegateV2Delete()</code> 销毁 GPU 委托。</p>
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
<p data-md-type="paragraph">查看 <code data-md-type="codespan">TfLiteGpuDelegateOptionsV2</code> 以使用自定义选项创建一个委托实例。您可以使用 <code data-md-type="codespan">TfLiteGpuDelegateOptionsV2Default()</code> 初始化默认选项，然后根据需要对其进行修改。</p>
<p data-md-type="paragraph">TFLite GPU for Android C/C++ 使用 <a href="https://bazel.io" data-md-type="link">Bazel</a> 构建系统。例如，可以使用以下命令构建委托：</p>
<pre data-md-type="block_code" data-md-language="sh"><code class="language-sh">bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:delegate                           # for static library
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so  # for dynamic library</code></pre>
<p data-md-type="paragraph">注：调用 <code data-md-type="codespan">Interpreter::ModifyGraphWithDelegate()</code> 或 <code data-md-type="codespan">Interpreter::Invoke()</code> 时，调用者在当前线程中必须具有 <code data-md-type="codespan">EGLContext</code>，并且 <code data-md-type="codespan">Interpreter::Invoke()</code> 必须从相同的 <code data-md-type="codespan">EGLContext</code> 调用。如果 <code data-md-type="codespan">EGLContext</code> 不存在，委托将在内部创建一个，但开发者随后必须确保该 <code data-md-type="codespan">Interpreter::Invoke()</code> 始终从调用 <code data-md-type="codespan">Interpreter::ModifyGraphWithDelegate()</code> 的同一个线程调用。</p>
<h3 data-md-type="header" data-md-header-level="3">iOS (C++)</h3>
<p data-md-type="paragraph">注：有关 Swift/Objective-C/C 用例，请参阅 <a href="gpu#ios" data-md-type="link">GPU 委托指南</a></p>
<p data-md-type="paragraph">注：仅当您使用 Bazel 或自行构建 TensorFlow Lite 时，此功能才可用。C++ API 不能与 CocoaPods 一起使用。</p>
<p data-md-type="paragraph">要在 GPU 上使用 TensorFlow Lite，请通过 <code data-md-type="codespan">TFLGpuDelegateCreate()</code> 获取 GPU 委托，然后将其传递给 <code data-md-type="codespan">Interpreter::ModifyGraphWithDelegate()</code>（而不是调用 <code data-md-type="codespan">Interpreter::AllocateTensors()</code>）。</p>
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
TFLGpuDelegateDelete(delegate);</code></pre>
<h2 data-md-type="header" data-md-header-level="2">高级用法</h2>
<h3 data-md-type="header" data-md-header-level="3">iOS 委托选项</h3>
<p data-md-type="paragraph">GPU 委托的构造函数接受选项的 <code data-md-type="codespan">struct</code>。（<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/swift/Sources/MetalDelegate.swift" data-md-type="link">Swift API</a>、<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/objc/apis/TFLMetalDelegate.h" data-md-type="link">Objective-C API</a>、<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/metal_delegate.h" data-md-type="link">C API</a>）</p>
<p data-md-type="paragraph">向初始值设定项传递 <code data-md-type="codespan">nullptr</code> (C API) 或不传递任何内容（Objective-C 和 Swift API）即可设置默认选项（上方“基本用法”示例已作说明）。</p>
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
<p data-md-type="paragraph">尽管使用 <code data-md-type="codespan">nullptr</code> 或默认构造函数十分方便，但建议您显式设置选项，以避免将来因更改默认值而发生任何意外行为。</p>
<h3 data-md-type="header" data-md-header-level="3">在 GPU 上运行量化模型</h3>
<p data-md-type="paragraph">本部分将说明 GPU 委托如何加速 8 位量化模型。这包括所有量化方式，包括：</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">使用<a href="https://www.tensorflow.org/lite/convert/quantization" data-md-type="link">量化感知训练</a>训练的模型</li>
<li data-md-type="list_item" data-md-list-type="unordered"><a href="https://www.tensorflow.org/lite/performance/post_training_quant" data-md-type="link">训练后动态范围量化</a></li>
<li data-md-type="list_item" data-md-list-type="unordered"><a href="https://www.tensorflow.org/lite/performance/post_training_integer_quant" data-md-type="link">训练后全整数量化</a></li>
</ul>
<p data-md-type="paragraph">为了优化性能，请使用具有浮点输入和输出张量的模型。</p>
<h4 data-md-type="header" data-md-header-level="4">运作方式</h4>
<p data-md-type="paragraph">由于 GPU 后端仅支持浮点执行，因此我们通过为其提供原始模型的“浮点视图”来运行量化模型。在较高层面上讲，这需要执行以下步骤：</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="false">
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph"><em data-md-type="emphasis">常量张量</em>（例如权重/偏置）进入 GPU 内存后会立即去量化。将委托应用于 TFLite 解释器时，就会发生这种情况。</p>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">如果为 8 位量化，则 GPU 程序的<em data-md-type="emphasis">输入和输出</em> 将分别针对每个推断进行去量化和量化。此操作在 CPU 上使用 TFLite 的优化内核完成。</p>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">通过在运算之间插入<em data-md-type="emphasis">量化模拟器</em>来修改 GPU 程序以模仿量化行为。如果运算期望激活函数遵循在量化过程中学习的边界，则对于这种模型而言，这是必需步骤。</p>
</li>
</ul>
<p data-md-type="paragraph">可以使用委托选项来启用此功能，如下所示：</p>
<h4 data-md-type="header" data-md-header-level="4">Android</h4>
<p data-md-type="paragraph">Android API 默认支持量化模型。要停用，请执行以下操作：</p>
<p data-md-type="paragraph"><strong data-md-type="double_emphasis">C++ API</strong></p>
<pre data-md-type="block_code" data-md-language="c++"><code class="language-c++">TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;

auto* delegate = TfLiteGpuDelegateV2Create(options);
if (interpreter-&gt;ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;</code></pre>
<p data-md-type="paragraph"><strong data-md-type="double_emphasis">Java API</strong></p>
<pre data-md-type="block_code" data-md-language="java"><code class="language-java">GpuDelegate delegate = new GpuDelegate(new GpuDelegate.Options().setQuantizedModelsAllowed(false));

Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);</code></pre>
<h4 data-md-type="header" data-md-header-level="4">iOS</h4>
<p data-md-type="paragraph">iOS API 默认支持量化模型。要停用，请执行以下操作：</p>
<div data-md-type="block_html">
<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    var options = MetalDelegate.Options()
    options.isQuantizationEnabled = false
    let delegate = MetalDelegate(options: options)</pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">    TFLMetalDelegateOptions* options = [[TFLMetalDelegateOptions alloc] init];
    options.quantizationEnabled = false;</pre>
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
<h3 data-md-type="header" data-md-header-level="3">输入/输出缓冲区（仅适用于 iOS，C++ API）</h3>
<p data-md-type="paragraph">注：仅当您使用 Bazel 或自行构建 TensorFlow Lite 时，此功能才可用。C++ API 不能与 CocoaPods 一起使用。</p>
<p data-md-type="paragraph">要在 GPU 上执行计算，则必须使数据可用于 GPU。这通常需要执行内存复制。如果可能，最好不要越过 CPU/GPU 内存边界，因为这会占用大量时间。通常，这种越界是不可避免的，但在某些特殊情况下却可以忽略其中一种内存。</p>
<p data-md-type="paragraph">如果网络的输入为 GPU 内存中已加载的图像（例如，包含摄像头feed 的 GPU 纹理），则它可以驻留在 GPU 内存中而无需进入 CPU 内存。同样，如果网络的输出为可渲染图像形式（例如，<a href="https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf" data-md-type="link">图像风格转换</a>），则可以直接在屏幕上显示。</p>
<p data-md-type="paragraph">为了获得最佳性能，TensorFlow Lite 使用户可以直接从 TensorFlow 硬件缓冲区读取和写入数据，并绕过可避免的内存复制过程。</p>
<p data-md-type="paragraph">假设图像输入位于 GPU 内存中，则必须首先将其转换为 Metal 的 <code data-md-type="codespan">MTLBuffer</code> 对象。您可以使用 <code data-md-type="codespan">TFLGpuDelegateBindMetalBufferToTensor()</code> 将 TfLiteTensor 关联至用户准备的 <code data-md-type="codespan">MTLBuffer</code>。请注意，必须在 <code data-md-type="codespan">Interpreter::ModifyGraphWithDelegate()</code> 之后调用 <code data-md-type="codespan">TFLGpuDelegateBindMetalBufferToTensor()</code>。此外，在默认情况下，推断输出会从 GPU 内存复制到 CPU 内存。可以通过在初始化期间调用 <code data-md-type="codespan">Interpreter::SetAllowBufferHandleOutput(true)</code> 来关闭此行为。</p>
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
<p data-md-type="paragraph">注：关闭默认行为后，要将推断输出从 GPU 内存复制到 CPU 内存，则需要对每个输出张量显式调用 <code data-md-type="codespan">Interpreter::EnsureTensorDataIsReadable()</code>。</p>
<p data-md-type="paragraph">注：这也适用于量化模型，但是您仍然需要<strong data-md-type="double_emphasis">带有 float32 数据的 float32 大小的缓冲区</strong>，因为该缓冲区将绑定到内部去量化缓冲区。</p>
<h2 data-md-type="header" data-md-header-level="2">提示和技巧</h2>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="false">
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">CPU 上一些琐碎的运算对于 GPU 而言可能会造成高昂的成本。各种形式的整形运算就是此类运算中的一类（包括 <code data-md-type="codespan">BATCH_TO_SPACE</code>、<code data-md-type="codespan">SPACE_TO_BATCH</code>、<code data-md-type="codespan">SPACE_TO_DEPTH</code> 以及类似运算）。如果并不需要这些运算（例如，插入它们只为帮助网络架构师分析系统，但不会影响输出），则有必要移除它们以提高性能。</p>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">在 GPU 上，张量数据会被切分成 4 个通道。因此，对形状为 <code data-md-type="codespan">[B, H, W, 5]</code> 的张量执行计算将与对形状为 <code data-md-type="codespan">[B, H, W, 8]</code> 的张量执行计算大致相同，但与 <code data-md-type="codespan">[B, H, W, 4]</code> 会有显著差距。</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">例如，如果相机硬件支持 RGBA 图像帧，那么馈送这种 4 通道输入的速度会显著提升，因为这避免了内存复制过程（从 3 通道 RGB 到 4 通道 RGBX）。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">为了获得最佳性能，请立即使用针对移动设备进行优化的网络架构来重新训练您的分类器。这是优化设备端推断的重要部分。</p>
</li>
</ul>
</div>
</div>
