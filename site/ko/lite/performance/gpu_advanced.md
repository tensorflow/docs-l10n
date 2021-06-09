# GPU의 TensorFlow Lite

[TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/)는 여러 하드웨어 가속기를 지원합니다. 이 설명서에서는 Android(OpenCL 또는 OpenGL ES 3.1 이상 필요) 및 iOS(iOS 8 이상 필요)에서 TensorFlow Lite 대리자 API를 사용하여 GPU 백엔드를 사용하는 방법을 설명합니다.

## GPU 가속의 이점

### 속도

GPU는 대규모 병렬 처리가 가능한 워크로드를 위해 높은 처리량을 갖도록 설계되었습니다. 따라서 GPU는 각각 작은 워크로드로 쉽게 분할되고 병렬로 수행될 수 있는 일부 입력 텐서(들)에서 작업하는 수많은 연산자로 구성된 심층 신경망에 적합합니다. 보통은 병렬 처리를 통해 지연 시간이 크게 줍니다. 최상의 시나리오에서 GPU에 대한 추론은 이전에는 가능하지 않았던 실시간 애플리케이션에 적합하도록 충분히 빠르게 실행될 수 있습니다.

### 정확성

GPU는 16bit 또는 32bit 부동 소수점 숫자로 계산을 수행하며 CPU와 달리 최적의 성능을 위해 양자화가 필요하지 않습니다. 정확성이 낮아 모델에 대해 양자화가 불가능한 경우 GPU에서 신경망을 실행하면 이 문제를 해결할 수 있습니다.

### 에너지 효율성

GPU 추론과 함께 제공되는 또 다른 이점은 전력 효율성입니다. GPU는 매우 효율적이고 최적화된 방식으로 계산을 수행하여 CPU에서 실행되는 같은 작업보다 전력을 덜 소비하고 열을 덜 발생시킵니다.

## 지원되는 연산

GPU의 TensorFlow Lite는 16bit 및 32bit 부동 소수점 정밀도에서 다음 연산을 지원합니다.

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

기본적으로 모든 연산은 버전 1에서만 지원됩니다. [실험적 양자화 지원](gpu_advanced.md#running-quantized-models-experimental-android-only)을 활성화하면 적절한 버전이 허용됩니다. 예: ADD v2.

## 기본 사용법

[Android Studio ML 모델 바인딩](../inference_with_metadata/codegen#acceleration) 또는 TensorFlow Lite 인터프리터를 사용하는 지에 따라 Android에서 모델 가속을 호출하는 두 가지 방법이 있습니다.

### TensorFlow Lite 인터프리터를 통한 Android

기존 `dependencies` 블록의 기존 `tensorflow-lite` 패키지와 함께 `tensorflow-lite-gpu` 패키지를 추가합니다.

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite:2.3.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.3.0'
}
```

`TfLiteDelegate`로 GPU에서 TensorFlow Lite를 실행합니다. Java에서는 `Interpreter.Options`를 통해 `GpuDelegate`를 지정할 수 있습니다.

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
<h3 data-md-type="header" data-md-header-level="3">Android(C/C++)</h3>
<p data-md-type="paragraph">Android에서 TensorFlow Lite GPU의 C/C++를 사용하는 경우, GPU 대리자는 <code data-md-type="codespan">TfLiteGpuDelegateV2Create()</code>로 만들고 <code data-md-type="codespan">TfLiteGpuDelegateV2Delete()</code>로 제거할 수 있습니다.</p>
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
TfLiteGpuDelegateV2Delete(delegate);
</code></pre>
<p data-md-type="paragraph"><code data-md-type="codespan">TfLiteGpuDelegateOptionsV2</code>를 살펴보고 사용자 정의 옵션이 있는 대리자 인스턴스를 만듭니다. <code data-md-type="codespan">TfLiteGpuDelegateOptionsV2Default()</code>를 사용하여 기본 옵션을 초기화한 다음 필요에 따라 수정할 수 있습니다.</p>
<p data-md-type="paragraph">Android C/C++용 TFLite GPU는 <a href="https://bazel.io" data-md-type="link">Bazel</a> 빌드 시스템을 사용합니다. 예를 들어 다음 명령을 사용하여 대리자를 만들 수 있습니다.</p>
<pre data-md-type="block_code" data-md-language="sh"><code class="language-sh">bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:delegate                           # for static library
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so  # for dynamic library
</code></pre>
<p data-md-type="paragraph">참고: <code data-md-type="codespan">Interpreter::ModifyGraphWithDelegate()</code> 또는<code data-md-type="codespan">Interpreter::Invoke()</code>를 호출할 때 호출자는 현재 스레드에 <code data-md-type="codespan">EGLContext</code>가 있어야 하며 <code data-md-type="codespan">Interpreter::Invoke()</code>는 같은 <code data-md-type="codespan">EGLContext</code>에서 호출되어야 합니다. <code data-md-type="codespan">EGLContext</code>가 없는 경우 대리자는 내부에서 하나를 만들지만, 개발자는 <code data-md-type="codespan">Interpreter:: Invoke()</code>가 항상 같은 스레드에서 호출되도록 해야 합니다. 그리고 이 스레드에서는 <code data-md-type="codespan">Interpreter::ModifyGraphWithDelegate()</code>가 호출되었습니다.</p>
<h3 data-md-type="header" data-md-header-level="3">iOS (C++)</h3>
<p data-md-type="paragraph">참고: Swift/Objective-C/C 사용 사례는 <a href="gpu#ios" data-md-type="link">GPU 대리자 가이드</a>를 참조하세요.</p>
<p data-md-type="paragraph">참고: 이는 베젤을 사용하거나 TensorFlow Lite를 직접 빌드하는 경우에만 사용할 수 있습니다. C++ API는 CocoaPods와 함께 사용할 수 없습니다.</p>
<p data-md-type="paragraph">GPU에서 TensorFlow Lite를 사용하려면 <code data-md-type="codespan">TFLGpuDelegateCreate()</code>를 통해 GPU 대리자를 가져온 다음 (<code data-md-type="codespan">Interpreter::AllocateTensors()</code>를 호출하는 대신) <code data-md-type="codespan">Interpreter::ModifyGraphWithDelegate()</code>에 전달합니다.</p>
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
<h2 data-md-type="header" data-md-header-level="2">고급 사용법</h2>
<h3 data-md-type="header" data-md-header-level="3">iOS용 대리자 옵션</h3>
<p data-md-type="paragraph">GPU 대리자의 생성자는 옵션의 <code data-md-type="codespan">struct</code>를 수락합니다(<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/swift/Sources/MetalDelegate.swift" data-md-type="link">Swift API</a>, <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/objc/apis/TFLMetalDelegate.h" data-md-type="link">Objective-C API</a>, <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/metal_delegate.h" data-md-type="link">C API</a>).</p>
<p data-md-type="paragraph">이니셜라이저로 <code data-md-type="codespan">nullptr</code>(C API) 또는 아무것도 전달하지 않으면(Objective-C 및 Swift API) 기본 옵션이 설정됩니다(위의 기본 사용 예제에서 설명됨).</p>
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
<p data-md-type="paragraph"><code data-md-type="codespan">nullptr</code> 또는 기본 생성자를 사용하는 것이 편리하지만 나중에 기본값이 변경될 경우 예기치 않은 동작이 발생하지 않도록 옵션을 명시적으로 설정하는 것이 좋습니다.</p>
<h3 data-md-type="header" data-md-header-level="3">GPU에서 양자화된 모델 실행하기</h3>
<p data-md-type="paragraph">이 섹션에서는 GPU 대리자가 8bit 양자화된 모델을 가속화하는 방법을 설명합니다. 여기에는 다음을 포함한 모든 종류의 양자화가 포함됩니다.</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">
<a href="https://www.tensorflow.org/lite/convert/quantization" data-md-type="link">양자화 인식 훈련으로</a> 훈련된 모델</li>
<li data-md-type="list_item" data-md-list-type="unordered"><a href="https://www.tensorflow.org/lite/performance/post_training_quant" data-md-type="link">훈련 후 동적 범위 양자화</a></li>
<li data-md-type="list_item" data-md-list-type="unordered"><a href="https://www.tensorflow.org/lite/performance/post_training_integer_quant" data-md-type="link">훈련 후 전체 정수 양자화</a></li>
</ul>
<p data-md-type="paragraph">성능을 최적화하려면 부동 소수점 입력 및 출력 텐서를 가진 모델을 사용하세요.</p>
<h4 data-md-type="header" data-md-header-level="4">어떻게 동작합니까?</h4>
<p data-md-type="paragraph">GPU 백엔드는 부동 소수점 실행만 지원하므로 원래 모델의 '부동 소수점 보기'를 제공하여 양자화된 모델을 실행합니다. 높은 수준에서 모델을 실행하려면 다음 단계가 수반됩니다.</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="false">
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph"><em data-md-type="emphasis">상수 텐서</em>(예: 가중치/바이어스)는 GPU 메모리로 한 번 역양자화되는데, 이는 대리자가 TFLite 인터프리터에 적용될 때 발생합니다.</p>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">8bit 양자화된 경우 GPU 프로그램에 대한 <em data-md-type="emphasis">입력 및 출력</em>은 각 추론에 대해 (각각) 역양자화 및 양자화됩니다. 이것은 TFLite의 최적화된 커널을 사용하여 CPU에서 수행됩니다.</p>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">GPU 프로그램은 연산 사이에 <em data-md-type="emphasis">양자화 시뮬레이터</em>를 삽입하여 양자화된 동작을 모방하도록 수정되었습니다. 이는 연산이 양자화 중에 학습한 경계를 따라 활성화될 것으로 예상되는 모델에 필요합니다.</p>
</li>
</ul>
<p data-md-type="paragraph">이 특성은 다음과 같은 대리자 옵션을 사용하여 활성화할 수 있습니다.</p>
<h4 data-md-type="header" data-md-header-level="4">Android</h4>
<p data-md-type="paragraph">Android API는 기본적으로 양자화된 모델을 지원합니다. 비활성화하려면 다음을 수행합니다.</p>
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
<p data-md-type="paragraph">iOS API는 기본적으로 양자화된 모델을 지원합니다. 비활성화하려면 다음을 수행합니다.</p>
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
<h3 data-md-type="header" data-md-header-level="3">입력/출력 버퍼(iOS, C++ API 전용)</h3>
<p data-md-type="paragraph">참고: 이는 베젤을 사용하거나 TensorFlow Lite를 직접 빌드하는 경우에만 사용할 수 있습니다. C++ API는 CocoaPods와 함께 사용할 수 없습니다.</p>
<p data-md-type="paragraph">GPU에서 계산을 수행하려면 GPU에서 데이터를 사용할 수 있어야 합니다. 따라서 종종 메모리 사본을 수행해야 합니다. 상당한 시간이 소요될 수 있으므로 가능하면 CPU/GPU 메모리 경계를 넘지 않는 것이 좋습니다. 일반적으로 이러한 교차는 불가피하지만 일부 특수한 경우에는 둘 중 하나를 생략할 수 있습니다.</p>
<p data-md-type="paragraph">네트워크의 입력이 GPU 메모리에 이미 로드된 이미지(예: 카메라 피드를 포함하는 GPU 텍스처)인 경우 CPU 메모리에 들어가지 않고도 GPU 메모리에 남아있을 수 있습니다. 마찬가지로 네트워크의 출력이 렌더링 가능한 이미지 형식(예: <a href="https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf" data-md-type="link">이미지 스타일 전송</a>)인 경우 화면에 직접 표시할 수 있습니다.</p>
<p data-md-type="paragraph">최고의 성능을 내도록 TensorFlow Lite는 사용자가 TensorFlow 하드웨어 버퍼에서 직접 읽고 쓸 수 있게 하고, 피할 수 있는 메모리 사본을 우회할 수 있도록 합니다.</p>
<p data-md-type="paragraph">이미지 입력이 GPU 메모리에 있다고 가정하면 먼저 이 입력을 Metal용 <code data-md-type="codespan">MTLBuffer</code> 객체로 변환해야 합니다. TfLiteTensor를 <code data-md-type="codespan">TFLGpuDelegateBindMetalBufferToTensor()</code>로 사용자가 준비한 <code data-md-type="codespan">MTLBuffer</code>와 연결할 수 있습니다. <code data-md-type="codespan">TFLGpuDelegateBindMetalBufferToTensor()</code>는 <code data-md-type="codespan">Interpreter::ModifyGraphWithDelegate()</code> 이후에 호출되어야 합니다. 또한 추론 출력은 기본적으로 GPU 메모리에서 CPU 메모리로 복사됩니다. 이 동작은 초기화 중에 <code data-md-type="codespan">Interpreter::SetAllowBufferHandleOutput(true)</code>을 호출하여 끌 수 있습니다.</p>
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
if (interpreter-&gt;Invoke() != kTfLiteOk) return false;
</code></pre>
<p data-md-type="paragraph">참고: 기본 동작이 꺼지고 나서 GPU 메모리에서 CPU 메모리로 추론 출력을 복사하려면 각 출력 텐서에 대해 <code data-md-type="codespan">Interpreter::EnsureTensorDataIsReadable()</code>을 명시적으로 호출해야 합니다.</p>
<p data-md-type="paragraph">참고: 이는 양자화된 모델에서도 동작하지만 버퍼가 내부의 역양자화된 버퍼에 바인딩되므로 <strong data-md-type="double_emphasis">float32 데이터가 있는 float32 크기의 버퍼</strong>가 여전히 필요합니다.</p>
<h2 data-md-type="header" data-md-header-level="2">팁과 요령</h2>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="false">
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">CPU에서 사소한 일부 연산은 GPU에서는 높은 비용이 발생할 수 있습니다. 이러한 연산의 한 클래스에는 다양한 형태의 reshape 연산(<code data-md-type="codespan">BATCH_TO_SPACE</code>, <code data-md-type="codespan">SPACE_TO_BATCH</code>, <code data-md-type="codespan">SPACE_TO_DEPTH</code> 및 유사한 연산 포함)이 포함됩니다. 이러한 연산이 필요하지 않은 경우(예: 네트워크 설계자가 시스템에 대해 추론하는 데 도움을 주기 위해 삽입되었지만 출력에 영향을 주지 않는 경우) 성능을 고려해서 해당 연산을 제거하는 것이 좋습니다.</p>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">GPU에서 텐서 데이터는 4채널로 분할됩니다. 따라서 형상 <code data-md-type="codespan">[B, H, W, 5]</code> 텐서에 대한 계산은 형상 <code data-md-type="codespan">[B, H, W, 8]</code> 텐서와 거의 동일하게 수행되지만 <code data-md-type="codespan">[B, H, W, 4]</code>에 비해서는 성능이 훨씬 나쁩니다.</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">예를 들어 카메라 하드웨어가 RGBA의 이미지 프레임을 지원하는 경우 메모리 사본(3채널 RGB에서 4채널 RGBX로)을 피할 수 있으므로 해당 4채널 입력을 훨씬 빠르게 공급할 수 있습니다.</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">최상의 성능을 위해 모바일에 최적화된 네트워크 아키텍처로 분류자를 다시 훈련하는 것을 주저하지 마세요. 이는 온디바이스 추론을 위한 최적화에 있어 중요한 부분입니다.</p>
</li>
</ul>
</div>
</div>
