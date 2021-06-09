# Tensorflow Lite Core ML delegate

TensorFlow Lite Core ML 대리자를 사용하면 [Core ML 프레임워크](https://developer.apple.com/documentation/coreml)에서 TensorFlow Lite 모델을 실행할 수 있으므로 iOS 기기에서 모델 추론이 더 빨라집니다.

Note: This delegate is in experimental (beta) phase. It is available from TensorFlow Lite 2.4.0 and latest nightly releases.

참고: Core ML 대리자는 Core ML 버전 2 이상을 지원합니다.

**지원되는 iOS 버전 및 기기**

- iOS 12 and later. In the older iOS versions, Core ML delegate will automatically fallback to CPU.
- By default, Core ML delegate will only be enabled on devices with A12 SoC and later (iPhone Xs and later) to use Neural Engine for faster inference. If you want to use Core ML delegate also on the older devices, please see [best practices](#best-practices)

**Supported models**

Core ML 대리자는 현재 float(FP32 및 FP16) 모델을 지원합니다.

## Trying the Core ML delegate on your own model

The Core ML delegate is already included in nightly release of TensorFlow lite CocoaPods. To use Core ML delegate, change your TensorFlow lite pod to include subspec `CoreML` in your `Podfile`.

Note: If you want to use C API instead of Objective-C API, you can include `TensorFlowLiteC/CoreML` pod to do so.

```
target 'YourProjectName'
  pod 'TensorFlowLiteSwift/CoreML', '~> 2.4.0'  # Or TensorFlowLiteObjC/CoreML
```

OR

```
# Particularily useful when you also want to include 'Metal' subspec.
target 'YourProjectName'
  pod 'TensorFlowLiteSwift', '~> 2.4.0', :subspecs => ['CoreML']
```

Note: Core ML delegate can also use C API for Objective-C code. Prior to TensorFlow Lite 2.4.0 release, this was the only option.

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    let coreMLDelegate = CoreMLDelegate()
    var interpreter: Interpreter
&lt;/div&gt;
&lt;pre data-md-type="block_code" data-md-language=""&gt;&lt;code&gt;GL_CODE_5&lt;/code&gt;</pre>
<div data-md-type="block_html">
</div>
</section></devsite-selector>
</div>
<h2 data-md-type="header" data-md-header-level="2">Best practices</h2>
<h3 data-md-type="header" data-md-header-level="3">Neural Engine이 없는 기기에서 Core ML 대리자 사용하기</h3>
<p data-md-type="paragraph">By default, Core ML delegate will only be created if the device has Neural Engine, and will return <code data-md-type="codespan">null</code> if the delegate is not created. If you want to run Core ML delegate on other environments (for example, simulator), pass <code data-md-type="codespan">.all</code> as an option while creating delegate in Swift. On C++ (and Objective-C), you can pass <code data-md-type="codespan">TfLiteCoreMlDelegateAllDevices</code>. Following example shows how to do this:</p>
<div data-md-type="block_html">
<div>
  <devsite-selector>
    <section>
      <h3> Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    var options = CoreMLDelegate.Options()
    options.enabledDevices = .all
    let coreMLDelegate = CoreMLDelegate(options: options)!
    let interpreter = try Interpreter(modelPath: modelPath,
                                      delegates: [coreMLDelegate])
      </pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">    TFLCoreMLDelegateOptions* coreMLOptions = [[TFLCoreMLDelegateOptions alloc] init];
    coreMLOptions.enabledDevices = TFLCoreMLDelegateEnabledDevicesAll;
    TFLCoreMLDelegate* coreMLDelegate = [[TFLCoreMLDelegate alloc]
                                          initWithOptions:coreMLOptions];
&lt;/div&gt;
&lt;pre data-md-type="block_code" data-md-language=""&gt;&lt;code&gt;GL_CODE_9&lt;/code&gt;</pre>
<div data-md-type="block_html">
</div>
</section></devsite-selector>
</div>
<h3 data-md-type="header" data-md-header-level="3">Metal(GPU) 대리자를 폴백으로 사용하기</h3>
<p data-md-type="paragraph">When the Core ML delegate is not created, alternatively you can still use <a href="https://www.tensorflow.org/lite/performance/gpu#ios" data-md-type="link">Metal delegate</a> to get performance benefits. Following example shows how to do this:</p>
<div data-md-type="block_html">
<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    var delegate = CoreMLDelegate()
    if delegate == nil {
      delegate = MetalDelegate()  // Add Metal delegate options if necessary.
    }
&lt;/div&gt;
&lt;pre data-md-type="block_code" data-md-language=""&gt;&lt;code&gt;GL_CODE_10&lt;/code&gt;</pre>
<div data-md-type="block_html">
</div>
</section></devsite-selector>
</div>
<p data-md-type="paragraph">The delegate creation logic reads device's machine id (e.g. iPhone11,1) to determine its Neural Engine availability. See the <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/coreml/coreml_delegate.mm" data-md-type="link">code</a> for more detail. Alternatively, you can implement your own set of denylist devices using other libraries such as <a href="https://github.com/devicekit/DeviceKit" data-md-type="link">DeviceKit</a>.</p>
<h3 data-md-type="header" data-md-header-level="3">이전 Core ML 버전 사용하기</h3>
<p data-md-type="paragraph">Although iOS 13 supports Core ML 3, the model might work better when it is converted with Core ML 2 model specification. The target conversion version is set to the latest version by default, but you can change this by setting <code data-md-type="codespan">coreMLVersion</code> (in Swift, <code data-md-type="codespan">coreml_version</code> in C API) in the delegate option to older version.</p>
<h2 data-md-type="header" data-md-header-level="2">지원되는 연산</h2>
<p data-md-type="paragraph">Core ML 대리자는 다음 연산을 지원합니다.</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Add</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">Only certain shapes are broadcastable. In Core ML tensor layout, following tensor shapes are broadcastable. <code data-md-type="codespan">[B, C, H, W]</code>, <code data-md-type="codespan">[B, C, 1, 1]</code>, <code data-md-type="codespan">[B, 1, H, W]</code>, <code data-md-type="codespan">[B, 1, 1, 1]</code>.</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">AveragePool2D</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Concat</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">채널 축을 따라 연결해야 합니다.</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Conv2D</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">가중치와 바이어스는 일정해야 합니다.</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">DepthwiseConv2D</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">가중치와 바이어스는 일정해야 합니다.</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">FullyConnected(일명 Dense 또는 InnerProduct)</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">가중치와 바이어스(있는 경우)는 일정해야 합니다.</li>
<li data-md-type="list_item" data-md-list-type="unordered">Only supports single-batch case. Input dimensions should be 1, except the last dimension.</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">Hardswish</li>
<li data-md-type="list_item" data-md-list-type="unordered">Logistic(일명 Sigmoid)</li>
<li data-md-type="list_item" data-md-list-type="unordered">MaxPool2D</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">MirrorPad</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">
<code data-md-type="codespan">REFLECT</code> 모드의 4D 입력만 지원됩니다. 패딩은 일정해야 하며 H 및 W 차원에만 허용됩니다.</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Mul</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">특정 형상만 브로드캐스팅할 수 있습니다. Core ML 텐서 레이아웃에서 다음 텐서 형상을 브로드캐스팅할 수 있습니다. <code data-md-type="codespan">[B, C, H, W]</code>, <code data-md-type="codespan">[B, C, 1, 1]</code>, <code data-md-type="codespan">[B, 1, H, W]</code>, <code data-md-type="codespan">[B, 1, 1, 1]</code>.</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Pad 및 PadV2</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">4D 입력만 지원됩니다. 패딩은 일정해야 하며 H 및 W 차원에만 허용됩니다.</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">Relu</li>
<li data-md-type="list_item" data-md-list-type="unordered">ReluN1To1</li>
<li data-md-type="list_item" data-md-list-type="unordered">Relu6</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Reshape</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">대상 Core ML 버전이 2인 경우에만 지원되고 Core ML 3을 대상으로 하는 경우 지원되지 않습니다.</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">ResizeBilinear</li>
<li data-md-type="list_item" data-md-list-type="unordered">SoftMax</li>
<li data-md-type="list_item" data-md-list-type="unordered">Tanh</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">TransposeConv</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">가중치는 일정해야 합니다.</li>
</ul>
</li>
</ul>
<h2 data-md-type="header" data-md-header-level="2">Feedback</h2>
<p data-md-type="paragraph">For issues, please create a <a href="https://github.com/tensorflow/tensorflow/issues/new?template=50-other-issues.md" data-md-type="link">GitHub</a> issue with all the necessary details to reproduce.</p>
<h2 data-md-type="header" data-md-header-level="2">FAQ</h2>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Does CoreML delegate support fallback to CPU if a graph contains unsupported ops?</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">Yes</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">CoreML 대리자는 iOS 시뮬레이터에서 동작합니까?</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">Yes. The library includes x86 and x86_64 targets so it can run on a simulator, but you will not see performance boost over CPU.</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">TensorFlow Lite 및 CoreML 대리자는 MacOS를 지원하나요?</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">TensorFlow Lite is only tested on iOS but not MacOS.</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">사용자 정의 TF Lite 연산이 지원되나요?</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">No, CoreML delegate does not support custom ops and they will fallback to CPU.</li>
</ul>
</li>
</ul>
<h2 data-md-type="header" data-md-header-level="2">API</h2>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered"><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/swift/Sources/CoreMLDelegate.swift" data-md-type="link">Core ML delegate Swift API</a></li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph"><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/coreml/coreml_delegate.h" data-md-type="link">Core ML delegate C API</a></p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">This can be used for Objective-C codes. ~~~</li>
</ul>
</li>
</ul>
</div>
</div>
