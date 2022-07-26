# Tensorflow Lite Core ML 대리자

TensorFlow Lite Core ML 대리자를 사용하면 [Core ML 프레임워크](https://developer.apple.com/documentation/coreml)에서 TensorFlow Lite 모델을 실행할 수 있으므로 iOS 기기에서 모델 추론이 더 빨라집니다.

Note: This delegate is in experimental (beta) phase. It is available from TensorFlow Lite 2.4.0 and latest nightly releases.

참고: Core ML 대리자는 Core ML 버전 2 이상을 지원합니다.

**지원되는 iOS 버전 및 기기**

- iOS 12 and later. In the older iOS versions, Core ML delegate will automatically fallback to CPU.
- By default, Core ML delegate will only be enabled on devices with A12 SoC and later (iPhone Xs and later) to use Neural Engine for faster inference. If you want to use Core ML delegate also on the older devices, please see [best practices](#best-practices)

**지원되는 모델**

Core ML 대리자는 현재 float(FP32 및 FP16) 모델을 지원합니다.

## Trying the Core ML delegate on your own model

The Core ML delegate is already included in nightly release of TensorFlow lite CocoaPods. To use Core ML delegate, change your TensorFlow lite pod to include subspec `CoreML` in your `Podfile`.

Note: If you want to use C API instead of Objective-C API, you can include `TensorFlowLiteC/CoreML` pod to do so.

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

Note: Core ML delegate can also use C API for Objective-C code. Prior to TensorFlow Lite 2.4.0 release, this was the only option.

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    let coreMLDelegate = CoreMLDelegate()
    var interpreter: Interpreter

    // Core ML delegate will only be created for devices with Neural Engine
    if coreMLDelegate != nil {
      interpreter = try Interpreter(modelPath: modelPath,
                                    delegates: [coreMLDelegate!])
    } else {
      interpreter = try Interpreter(modelPath: modelPath)
    }
      </pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">
    // Import module when using CocoaPods with module support
    @import TFLTensorFlowLite;

    // Or import following headers manually
    # import "tensorflow/lite/objc/apis/TFLCoreMLDelegate.h"
    # import "tensorflow/lite/objc/apis/TFLTensorFlowLite.h"

    // Initialize Core ML delegate
    TFLCoreMLDelegate* coreMLDelegate = [[TFLCoreMLDelegate alloc] init];

    // Initialize interpreter with model path and Core ML delegate
    TFLInterpreterOptions* options = [[TFLInterpreterOptions alloc] init];
    NSError* error = nil;
    TFLInterpreter* interpreter = [[TFLInterpreter alloc]
                                    initWithModelPath:modelPath
                                              options:options
                                            delegates:@[ coreMLDelegate ]
                                                error:&amp;error];
    if (error != nil) { /* Error handling... */ }

    if (![interpreter allocateTensorsWithError:&amp;error]) { /* Error handling... */ }
    if (error != nil) { /* Error handling... */ }

    // Run inference ...
      </pre>
    </section>
    <section>
      <h3>C(2.3.0까지)</h3>
      <p></p>
<pre class="prettyprint lang-c">    #include "tensorflow/lite/delegates/coreml/coreml_delegate.h"

    // Initialize interpreter with model
    TfLiteModel* model = TfLiteModelCreateFromFile(model_path);

    // Initialize interpreter with Core ML delegate
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteDelegate* delegate = TfLiteCoreMlDelegateCreate(NULL);  // default config
    TfLiteInterpreterOptionsAddDelegate(options, delegate);
    TfLiteInterpreterOptionsDelete(options);

    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

    TfLiteInterpreterAllocateTensors(interpreter);

    // Run inference ...

    /* ... */

    // Dispose resources when it is no longer used.
    // Add following code to the section where you dispose of the delegate
    // (e.g. `dealloc` of class).

    TfLiteInterpreterDelete(interpreter);
    TfLiteCoreMlDelegateDelete(delegate);
    TfLiteModelDelete(model);
      </pre>
    </section>
  </devsite-selector>
</div>

## 모범 사례

### Neural Engine이 없는 기기에서 Core ML 대리자 사용하기

기본적으로 Core ML 대리자는 기기에 Neural Engine이 있는 경우에만 생성되고 대리자가 생성되지 않은 경우 `null`을 반환합니다. 다른 환경(예: 시뮬레이터)에서 Core ML 대리자를 실행하려면 Swift에서 대리자를 생성하는 동안 `.all`을 옵션으로 전달하세요. C++(및 Objective-C)에서는 `TfLiteCoreMlDelegateAllDevices`를 전달할 수 있습니다. 다음 예에서는 이를 수행하는 방법을 보여줍니다.

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

    // Initialize interpreter with delegate
      </pre>
    </section>
    <section>
      <h3>C</h3>
      <p></p>
<pre class="prettyprint lang-c">    TfLiteCoreMlDelegateOptions options;
    options.enabled_devices = TfLiteCoreMlDelegateAllDevices;
    TfLiteDelegate* delegate = TfLiteCoreMlDelegateCreate(&amp;options);
    // Initialize interpreter with delegate
      </pre>
    </section>
  </devsite-selector>
</div>

### Metal(GPU) 대리자를 폴백으로 사용하기

Core ML 대리자가 생성되지 않은 경우에도 여전히 [Metal 대리자](https://www.tensorflow.org/lite/performance/gpu#ios)를 사용하여 성능 이점을 얻을 수 있습니다. 다음 예에서는 이를 수행하는 방법을 보여줍니다.

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    var delegate = CoreMLDelegate()
    if delegate == nil {
      delegate = MetalDelegate()  // Add Metal delegate options if necessary.
    }

    let interpreter = try Interpreter(modelPath: modelPath,
                                      delegates: [delegate!])
      </pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">    TFLDelegate* delegate = [[TFLCoreMLDelegate alloc] init];
    if (!delegate) {
      // Add Metal delegate options if necessary
      delegate = [[TFLMetalDelegate alloc] init];
    }
    // Initialize interpreter with delegate
      </pre>
    </section>
    <section>
      <h3>C</h3>
      <p></p>
<pre class="prettyprint lang-c">    TfLiteCoreMlDelegateOptions options = {};
    delegate = TfLiteCoreMlDelegateCreate(&amp;options);
    if (delegate == NULL) {
      // Add Metal delegate options if necessary
      delegate = TFLGpuDelegateCreate(NULL);
    }
    // Initialize interpreter with delegate
      </pre>
    </section>
  </devsite-selector>
</div>

대리자 생성 로직은 기기의 머신 ID(예: iPhone11,1)를 읽어 Neural Engine 가용성을 결정합니다. 자세한 내용은 [코드](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/coreml/coreml_delegate.mm)를 참조하세요. 또는 [DeviceKit](https://github.com/devicekit/DeviceKit)와 같은 다른 라이브러리를 사용하여 자체 거부 목록 기기 세트를 구현할 수 있습니다.

### 이전 Core ML 버전 사용하기

iOS 13은 Core ML 3을 지원하지만 Core ML 2 모델 사양으로 변환하면 모델이 더 잘 동작할 수 있습니다. 대상 변환 버전은 기본적으로 최신 버전으로 설정되어 있지만 대리자 옵션에서 `coreMLVersion`(Swift에서는 C API의 `coreml_version`)을 이전 버전으로 설정하여 변경할 수 있습니다.

## 지원되는 연산

Core ML 대리자는 다음 연산을 지원합니다.

- Add
    - 특정 형상만 브로드캐스팅할 수 있습니다. Core ML 텐서 레이아웃에서 다음 텐서 형상을 브로드캐스팅할 수 있습니다. `[B, C, H, W]`, `[B, C, 1, 1]`, `[B, 1, H, W]`, `[B, 1, 1, 1]`.
- AveragePool2D
- Concat
    - 채널 축을 따라 연결해야 합니다.
- Conv2D
    - 가중치와 바이어스는 일정해야 합니다.
- DepthwiseConv2D
    - 가중치와 바이어스는 일정해야 합니다.
- FullyConnected(일명 Dense 또는 InnerProduct)
    - 가중치와 바이어스(있는 경우)는 일정해야 합니다.
    - 단일 배치 케이스만 지원합니다. 입력 차원은 마지막 차원을 제외하고 1이어야 합니다.
- Hardswish
- Logistic(일명 Sigmoid)
- MaxPool2D
- MirrorPad
    -  `REFLECT` 모드의 4D 입력만 지원됩니다. 패딩은 일정해야 하며 H 및 W 차원에만 허용됩니다.
- Mul
    - 특정 형상만 브로드캐스팅할 수 있습니다. Core ML 텐서 레이아웃에서 다음 텐서 형상을 브로드캐스팅할 수 있습니다. `[B, C, H, W]`, `[B, C, 1, 1]`, `[B, 1, H, W]`, `[B, 1, 1, 1]`.
- Pad 및 PadV2
    - 4D 입력만 지원됩니다. 패딩은 일정해야 하며 H 및 W 차원에만 허용됩니다.
- Relu
- ReluN1To1
- Relu6
- Reshape
    - 대상 Core ML 버전이 2인 경우에만 지원되고 Core ML 3을 대상으로 하는 경우 지원되지 않습니다.
- ResizeBilinear
- SoftMax
- Tanh
- TransposeConv
    - 가중치는 일정해야 합니다.

## 피드백

문제가 발생한 경우 재현하는 데 필요한 모든 세부 정보가 포함된 [GitHub](https://github.com/tensorflow/tensorflow/issues/new?template=50-other-issues.md) 문제를 만드세요.

## 자주하는 질문

- 그래프에 지원되지 않는 연산이 포함된 경우 CoreML 대리자가 CPU로 폴백을 지원합니까?
    - 예
- CoreML 대리자는 iOS 시뮬레이터에서 동작합니까?
    - 예. 라이브러리에는 x86 및 x86_64 대상이 포함되어 있으므로 시뮬레이터에서 실행할 수 있지만 CPU에 비해 성능이 향상되지는 않습니다.
- TensorFlow Lite 및 CoreML 대리자는 MacOS를 지원하나요?
    - TensorFlow Lite는 iOS에서만 테스트되고 MacOS에서는 테스트되지 않습니다.
- 사용자 정의 TF Lite 연산이 지원되나요?
    - 아니요, CoreML 대리자는 사용자 정의 ops를 지원하지 않으며 CPU로 폴백됩니다.

## API

- [Core ML delegate Swift API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/swift/Sources/CoreMLDelegate.swift)
- [Core ML delegate C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/coreml/coreml_delegate.h)
    - Objective-C 코드에 사용될 수 있습니다. ~~~
