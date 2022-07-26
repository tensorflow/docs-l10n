# TensorFlow Lite GPU 대리자

[TensorFlow Lite](https://www.tensorflow.org/lite)는 여러 하드웨어 가속기를 지원합니다. 이 문서는 Android 및 iOS에서 TensorFlow Lite 대리자 API를 사용하여 GPU 백엔드를 사용하는 방법을 설명합니다.

GPU는 대규모 병렬 처리가 가능한 워크로드를 위해 높은 처리량을 갖도록 설계되었습니다. 따라서 GPU는 많은 수의 연산자로 구성된 심층 신경망에 적합하며, 일부 입력 텐서에서 더 작은 워크로드로 쉽게 분할되고 병렬로 수행될 수 있는 각 작업은 일반적으로 지연 시간이 더 짧습니다. 최상의 시나리오에서, GPU에 대한 추론은 이전에는 사용할 수 없었던 실시간 애플리케이션에 대해 충분히 빠르게 실행될 수 있습니다.

CPU와 달리 GPU는 16비트 또는 32비트 부동 소수점 숫자로 계산하며 최적의 성능을 위해 양자화가 필요하지 않습니다. 대리자는 8비트 양자화된 모델을 허용하지만 계산은 부동 소수점 숫자로 수행됩니다. 자세한 내용은 [고급 문서](gpu_advanced.md)를 참조하세요.

GPU 추론의 또 다른 이점은 전력 효율성입니다. GPU는 매우 효율적이고 최적화된 방식으로 계산을 수행하므로 같은 작업이 CPU에서 실행될 때보다 전력을 덜 소비하고 열을 덜 발생시킵니다.

## 데모 앱 튜토리얼

GPU 대리자를 시험해보는 가장 쉬운 방법은 GPU 지원을 통해 분류 데모 애플리케이션을 빌드하는 아래 튜토리얼을 따르는 것입니다. GPU 코드는 현재로서는 바이너리일 뿐이며 곧 오픈 소스가 될 것입니다. 데모를 동작시키는 방법을 이해한 후에는 자체 사용자 정의 모델에서 이를 시도해 볼 수 있습니다.

### Android(Android Studio 포함)

단계별 가이드는 [Android용 GPU 대리자](https://youtu.be/Xkhgre8r5G0) 동영상을 시청하세요.

참고: OpenCL 또는 OpenGL ES(3.1 이상)가 필요합니다.

#### 1단계: TensorFlow 소스 코드를 복제하고 Android Studio에서 엽니다.

```sh
git clone https://github.com/tensorflow/tensorflow
```

#### 2단계: 야간 GPU AAR을 사용하도록 `app/build.gradle`을 편집합니다.

참고: 이제 매니페스트에 `targetSdkVersion="S"`을 적용하거나 Gradle `defaultConfig`(API 수준 미정)에 `targetSdkVersion "S"`를 적용해 **Android S+**를 대상으로 지정할 수 있습니다. 이 경우 [`AndroidManifestGpu.xml`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/AndroidManifestGpu.xml)의 콘텐츠를 Android 애플리케이션의 매니페스트에 병합해야 합니다. 이 변경이 없으면 GPU 대리자가 가속을 위해 OpenCL 라이브러리에 액세스할 수 없습니다. *이 작업을 수행하려면 AGP 4.2.0 이상이 필요합니다.*

기존 `dependencies` 블록의 기존 `tensorflow-lite` 패키지와 함께 `tensorflow-lite-gpu` 패키지를 추가합니다.

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite:2.3.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.3.0'
}
```

#### 3단계: 빌드하고 실행합니다.

실행 → '앱'을 실행합니다. 애플리케이션을 실행하면 GPU 활성화 버튼이 표시됩니다. 양자화에서 부동 모델로 변경한 다음 GPU를 클릭하여 GPU에서 실행합니다.

![Android GPU 데모 실행 및 GPU로 전환](images/android_gpu_demo.gif)

### iOS(XCode 포함)

단계별 가이드는 [GPU Delegate for iOS](https://youtu.be/a5H4Zwjp49c) 동영상을 시청하세요.

참고: XCode v10.1 이상이 필요합니다.

#### 1단계: 데모 소스 코드를 얻고 컴파일되었는지 확인합니다.

iOS 데모 앱 [튜토리얼](https://www.tensorflow.org/lite/guide/ios)을 따라 진행합니다. 이를 통해 수정되지 않은 iOS 카메라 데모가 휴대전화에서 동작하는 지점으로 이동합니다.

#### 2단계: TensorFlow Lite GPU CocoaPod를 사용하도록 Podfile을 수정합니다.

2.3.0 릴리스부터 바이너리 크기를 줄이기 위해 기본적으로 GPU 대리자가 포드에서 제외됩니다. 하위 사양을 지정하여 이를 포함할 수 있습니다. `TensorFlowLiteSwift` 포드의 경우:

```ruby
pod 'TensorFlowLiteSwift/Metal', '~> 0.0.1-nightly',
```

또는

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly', :subspecs => ['Metal']
```

Objective-C(2.4.0 릴리스부터) 또는 C API를 사용하려는 경우 `TensorFlowLiteObjC` 또는 `TensorFlowLitC`에 대해서도 유사하게 수행할 수 있습니다.

<div>
  <devsite-expandable>
    <h4 class="showalways">2.3.0 릴리스 이전</h4>
    <h4>TensorFlow Lite 2.0.0까지</h4>
    <p>GPU 대리자를 포함하는 바이너리 CocoaPod를 구축했습니다. 이를 사용하도록 프로젝트를 전환하려면 `TensorFlowLite` 대신`TensorFlowLiteGpuExperimental` 포드를 사용하도록 `tensorflow/tensorflow/lite/examples/ios/camera/Podfile` 파일을 수정하세요.</p>
    <pre class="prettyprint lang-ruby notranslate" translate="no"><code>
    target 'YourProjectName'
      # pod 'TensorFlowLite', '1.12.0'
      pod 'TensorFlowLiteGpuExperimental'
    </code></pre>
    <h4>TensorFlow Lite 2.2.0까지</h4>
    <p>TensorFlow Lite 2.1.0에서 2.2.0까지 GPU 대리자는 `TensorFlowLiteC` 포드에 포함됩니다. 언어에 따라 `TensorFlowLiteC`와 `TensorFlowLiteSwift` 중에서 선택할 수 있습니다.</p>
  </devsite-expandable>
</div>

#### 3단계: GPU 대리자를 활성화합니다.

GPU 대리자를 사용할 코드를 활성화하려면 `CameraExampleViewController.h`에서 `TFLITE_USE_GPU_DELEGATE`를 0에서 1로 변경해야 합니다.

```c
#define TFLITE_USE_GPU_DELEGATE 1
```

#### 4단계: 데모 앱을 빌드하고 실행합니다.

이전 단계를 수행한 후 앱을 실행할 수 있습니다.

#### 5단계: 모드를 릴리스합니다.

4단계에서 디버그 모드로 실행하는 동안 성능을 향상하려면 적절한 최적의 Metal 설정을 사용하여 릴리스 빌드로 변경해야 합니다. 특히 해당 설정을 편집하려면 `Product> Scheme> Edit Scheme...`으로 이동합니다. `Run`을 선택합니다. `Info` 탭에서 `Build Configuration`을 `Debug`에서 `Release`로 변경하고 `Debug executable`을 선택 취소합니다.

![금속 옵션 설정](images/iosmetal.png)

그런 다음 `Options` 탭을 클릭하고 `GPU Frame Capture`를 `Disabled`로 변경하고 `Metal API Validation`을 `Disabled`로 바꿉니다.

![릴리스 설정](images/iosdebug.png)

마지막으로 64bit 아키텍처에서 릴리스 전용 빌드를 선택해야 합니다. `Project navigator -> tflite_camera_example -> PROJECT -> tflite_camera_example -> Build Settings`에서 `Build Active Architecture Only > Release`를 Yes로 설정합니다.

![릴리스 옵션 설정](images/iosrelease.png)

## 자체 모델에서 GPU 대리자 시도하기

### Android

참고: TensorFlow Lite 인터프리터는 실행될 때와 같은 스레드에서 생성되어야 합니다. 그렇지 않으면, `TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized.`가 발생할 수 있습니다.

[Android Studio ML 모델 바인딩](../inference_with_metadata/codegen#acceleration) 또는 TensorFlow Lite 인터프리터를 사용하는 지에 따라 모델 가속을 호출하는 두 가지 방법이 있습니다.

#### TensorFlow Lite 인터프리터

대리자를 추가하는 방법을 보려면 데모를 보세요. 애플리케이션에서 위와 같이 AAR을 추가하고 `org.tensorflow.lite.gpu.GpuDelegate` 모듈을 가져온 다음 `addDelegate` 함수를 사용하여 GPU 대리자를 인터프리터에 등록합니다.

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

### iOS

참고: GPU 대리자는 Objective-C 코드에 C API를 사용할 수도 있습니다. TensorFlow Lite 2.4.0 릴리스 이전에는 이것이 유일한 옵션이었습니다.

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    import TensorFlowLite

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
<pre class="prettyprint lang-objc">    // Import module when using CocoaPods with module support
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
    ```
      </pre>
    </section>
    <section>
      <h3>C(2.3.0까지)</h3>
      <p></p>
<pre class="prettyprint lang-c">    #include "tensorflow/lite/c/c_api.h"
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

## 지원되는 모델 및 연산

릴리스된 GPU 대리자에 백엔드에서 실행할 수 있는 몇 가지 모델이 포함되었습니다.

- [MobileNet v1(224x224) 이미지 분류](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html) [[다운로드]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobilenet_v1_1.0_224.tflite) <br><i>(모바일 및 임베디드 기반 비전 애플리케이션을 위해 설계된 이미지 분류 모델)</i>
- [DeepLab 분할(257x257)](https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html) [[다운로드]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite) <br><i>(입력 이미지의 모든 픽셀에 의미론적 레이블(예: 개, 고양이, 자동차)을 할당하는 이미지 분할 모델)</i>
- [MobileNet SSD 개체 감지](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html) [[다운로드]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite) <br><i>(경계 상자가 있는 여러 개체를 감지하는 이미지 분류 모델)</i>
- [포즈 추정을 위한 PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/posenet) [[다운로드]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite) <br><i>(이미지 또는 동영상에서 사람의 포즈를 추정하는 비전 모델)</i>

지원되는 연산의 전체 목록을 보려면 [고급 설명서](gpu_advanced.md)를 참조하세요.

## 지원되지 않는 모델 및 작업

일부 작업이 GPU 대리자에서 지원되지 않는 경우 프레임워크는 GPU에서 그래프의 일부만 실행하고 CPU에서 나머지 부분을 실행합니다. CPU/GPU 동기화 비용이 높기 때문에 이와 같은 분할 실행 모드는 전체 네트워크가 CPU에서만 실행될 때보다 성능이 저하되는 경우가 많습니다. 이 경우 사용자는 다음과 같은 경고를 받게 됩니다.

```none
WARNING: op code #42 cannot be handled by this delegate.
```

실패에 대한 콜백이 제공되지 않았습니다. 이 실패는 진정한 런타임 실패가 아니기 때문에 개발자가 대리자에서 네트워크를 실행하는 동안 관찰할 수 있습니다.

## 최적화를 위한 팁

### 모바일 장치를 위한 최적화

<br>CPU에서는 사소한 작업이 모바일 장치의 GPU에서는 높은 비용을 지급해야 하는 작업일 수 있습니다. `BATCH_TO_SPACE`, `SPACE_TO_BATCH`, `SPACE_TO_DEPTH` 등의 형상 변경 작업은 실행 비용이 많이 듭니다. 형상 변경 작업의 사용을 자세히 조사하고 데이터 탐색이나 초기에 모델을 반복하는 작업에만 적용되었을 수 있음을 고려해야 합니다. 형상 변경 작업을 제거하면 성능이 크게 향상될 수 있습니다.

GPU에서 텐서 데이터는 4채널로 분할됩니다. 따라서 `[B,H,W,5]` 형상의 텐서에 대한 계산은 `[B,H,W,8]` 형상의 텐서에 대한 계산과 거의 동일하게 수행되지만 `[B,H,W,4]`에 대한 계산보다 훨씬 좋지 않습니다. 그런 의미에서 카메라 하드웨어가 RGBA의 이미지 프레임을 지원하는 경우 메모리 복사(3채널 RGB에서 4채널 RGBX로)를 피할 수 있으므로 해당 4채널 입력을 공급하는 것이 훨씬 더 빠릅니다.

최상의 성능을 위해 모바일에 최적화된 네트워크 아키텍처를 사용하여 분류자를 다시 훈련하는 것을 고려해야 합니다. 온디바이스 최적화를 진행하면 모바일 하드웨어 기능의 이점을 활용하여 대기 시간과 전력 소비를 크게 줄일 수 있습니다.

### 직렬화로 초기화 시간 단축하기

GPU 대리자 기능을 사용하여 사전에 컴파일된 커널 코드로부터 로드하고, 직렬화되고 디스크에 저장된 모델 데이터를 이전 실행으로부터 로드할 수 있습니다. 이 접근 방식은 다시 컴파일을 수행하는 것을 방지하고 시작 시간을 최대 90%까지 단축합니다. 프로젝트에 직렬화를 적용하는 방법에 대한 지침은 [GPU 대리자 직렬화](gpu_advanced.md#gpu_delegate_serialization)를 참조하세요.
