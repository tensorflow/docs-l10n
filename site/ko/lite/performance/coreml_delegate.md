# Tensorflow Lite Core ML 대리자

TensorFlow Lite Core ML 대리자를 사용하면 [Core ML 프레임워크](https://developer.apple.com/documentation/coreml)에서 TensorFlow Lite 모델을 실행할 수 있으므로 iOS 기기에서 모델 추론이 더 빨라집니다.

참고: 이 대리자는 실험 (베타) 단계입니다. TensorFlow Lite 2.4.0 및 최신 야간 릴리스에서 사용할 수 있습니다.

참고: Core ML 대리자는 Core ML 버전 2 이상을 지원합니다.

**지원되는 iOS 버전 및 기기**

- iOS 12 이상. 이전 iOS 버전에서 Core ML 대리자는 자동으로 CPU로 대체됩니다.
- 기본적으로 Core ML 대리자는 A12 SoC 이상 (iPhone Xs 이상)이 있는 기기에서만 활성화되어 더 빠른 추론을 위해 Neural Engine을 사용합니다. Core ML 대리자를 이전 기기에서도 사용하려면 [모범 사례](#best-practices)를 참조하세요.

**지원되는 모델**

Core ML 대리자는 현재 float(FP32 및 FP16) 모델을 지원합니다.

## 자신의 모델에서 Core ML 대리자 시도

Core ML 대리자는 TensorFlow lite CocoaPods의 야간 릴리스에 이미 포함되어 있습니다. Core ML 대리자를 사용하려면 TensorFlow lite 포드를 변경하고 하위 스펙`CoreML`를 `Podfile`에 포함하십시오.

참고: Objective-C API 대신 C API를 사용하고자 하는 경우,  `TensorFlowLiteC/CoreML` 포드를 포함시킬 수 있습니다.

```
target 'YourProjectName'
  pod 'TensorFlowLiteSwift/CoreML', '~> 2.4.0'  # Or TensorFlowLiteObjC/CoreML
```

또는

```
# Particularily useful when you also want to include 'Metal' subspec.
target 'YourProjectName'
  pod 'TensorFlowLiteSwift', '~> 2.4.0', :subspecs => ['CoreML']
```

참고 : Core ML 대리자는 Objective-C 코드에 C API를 사용할 수도 있습니다. TensorFlow Lite 2.4.0 릴리스 이전에는 이것이 유일한 옵션이었습니다.

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
<h2 data-md-type="header" data-md-header-level="2">모범 사례</h2>
<h3 data-md-type="header" data-md-header-level="3">Neural Engine이 없는 기기에서 Core ML 대리자 사용하기</h3>
<p data-md-type="paragraph">기본적으로 Core ML 대리자는 기기에 Neural Engine이 있는 경우에만 생성되고 대리자가 생성되지 않은 경우 <code data-md-type="codespan">null</code>을 반환합니다. 다른 환경(예: 시뮬레이터)에서 Core ML 대리자를 실행하려면 Swift에서 대리자를 생성하는 동안 <code data-md-type="codespan">.all</code>을 옵션으로 전달하십시오. C++(및 Objective-C)에서는 <code data-md-type="codespan">TfLiteCoreMlDelegateAllDevices</code>를 전달할 수 있습니다. 다음 예에서는 이를 수행하는 방법을 보여줍니다.</p>
<div data-md-type="block_html">
<div>
  <devsite-selector>
    <section>
      <h3> Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">var options = CoreMLDelegate.Options()
options.enabledDevices = .all
let coreMLDelegate = CoreMLDelegate(options: options)!
let interpreter = try Interpreter(modelPath: modelPath,
                                  delegates: [coreMLDelegate])</pre>
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
<p data-md-type="paragraph">Core ML 대리자가 생성되지 않은 경우에도 여전히 <a href="https://www.tensorflow.org/lite/performance/gpu#ios" data-md-type="link">Metal 대리자</a>를 사용하여 성능 이점을 얻을 수 있습니다. 다음 예에서는 이를 수행하는 방법을 보여줍니다.</p>
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
<p data-md-type="paragraph">대리자 생성 로직은 기기의 머신 ID(예: iPhone11,1)를 읽어 Neural Engine 가용성을 결정합니다. 자세한 내용은 <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/delegates/coreml/coreml_delegate.mm" data-md-type="link">코드</a>를 참조하세요. 또는 <a href="https://github.com/devicekit/DeviceKit" data-md-type="link">DeviceKit</a>와 같은 다른 라이브러리를 사용하여 자체 거부 목록 기기 세트를 구현할 수 있습니다.</p>
<h3 data-md-type="header" data-md-header-level="3">이전 Core ML 버전 사용하기</h3>
<p data-md-type="paragraph">iOS 13은 Core ML 3을 지원하지만 Core ML 2 모델 사양으로 변환하면 모델이 더 잘 동작할 수 있습니다. 대상 변환 버전은 기본적으로 최신 버전으로 설정되어 있지만 대리자 옵션에서 <code data-md-type="codespan">coreMLVersion</code>(Swift에서는 C API의 <code data-md-type="codespan">coreml_version</code>)을 이전 버전으로 설정하여 변경할 수 있습니다.</p>
<h2 data-md-type="header" data-md-header-level="2">지원되는 연산</h2>
<p data-md-type="paragraph">Core ML 대리자는 다음 연산을 지원합니다.</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Add</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">특정 형상만 브로드캐스팅할 수 있습니다. Core ML 텐서 레이아웃에서 다음 텐서 형상을 브로드캐스팅할 수 있습니다. <code data-md-type="codespan">[B, C, H, W]</code>, <code data-md-type="codespan">[B, C, 1, 1]</code>, <code data-md-type="codespan">[B, 1, H, W]</code>, <code data-md-type="codespan">[B, 1, 1, 1]</code> .</li>
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
<li data-md-type="list_item" data-md-list-type="unordered">단일 배치 케이스만 지원합니다. 입력 차원은 마지막 차원을 제외하고 1이어야 합니다.</li>
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
<h2 data-md-type="header" data-md-header-level="2">피드백</h2>
<p data-md-type="paragraph">문제가 발생한 경우 재현하는 데 필요한 모든 세부 정보가 포함된 <a href="https://github.com/tensorflow/tensorflow/issues/new?template=50-other-issues.md" data-md-type="link">GitHub</a> 문제를 생성하십시오.</p>
<h2 data-md-type="header" data-md-header-level="2">자주하는 질문</h2>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">그래프에 지원되지 않는 연산이 포함된 경우 CoreML 대리자가 CPU로 폴백을 지원합니까?</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">예</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">CoreML 대리자는 iOS 시뮬레이터에서 동작합니까?</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">예. 라이브러리에는 x86 및 x86_64 대상이 포함되어 있으므로 시뮬레이터에서 실행할 수 있지만 CPU에 비해 성능이 향상되지는 않습니다.</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">TensorFlow Lite 및 CoreML 대리자는 MacOS를 지원하나요?</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">TensorFlow Lite는 iOS에서만 테스트되고 MacOS에서는 테스트되지 않습니다.</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">사용자 정의 TF Lite 연산이 지원되나요?</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">아니요, CoreML 대리자는 사용자 정의 연산을 지원하지 않으며 CPU로 대체됩니다.</li>
</ul>
</li>
</ul>
<h2 data-md-type="header" data-md-header-level="2">API</h2>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered"><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/swift/Sources/CoreMLDelegate.swift" data-md-type="link">Core ML delegate Swift API</a></li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph"><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/coreml/coreml_delegate.h" data-md-type="link">Core ML delegate C API</a></p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">Objective-C 코드에 사용될 수 있습니다. ~~~</li>
</ul>
</li>
</ul>
</div>
</div>
