# TensorFlow Lite GPU 대리자

[TensorFlow Lite](https://www.tensorflow.org/lite)는 여러 하드웨어 가속기를 지원합니다. 이 문서는 Android 및 iOS에서 TensorFlow Lite 대리자 API를 사용하여 GPU 백엔드를 사용하는 방법을 설명합니다.

GPU는 대규모 병렬 처리가 가능한 워크로드를 위해 높은 처리량을 갖도록 설계되었습니다. 따라서 GPU는 많은 수의 연산자로 구성된 심층 신경망에 적합하며, 일부 입력 텐서에서 더 작은 워크로드로 쉽게 분할되고 병렬로 수행될 수 있는 각 작업은 일반적으로 지연 시간이 더 짧습니다. 최상의 시나리오에서, GPU에 대한 추론은 이전에는 사용할 수 없었던 실시간 애플리케이션에 대해 충분히 빠르게 실행될 수 있습니다.

CPU와 달리 GPU는 16비트 또는 32비트 부동 소수점 숫자로 계산하며 최적의 성능을 위해 양자화가 필요하지 않습니다. 대리자는 8비트 양자화된 모델을 허용하지만 계산은 부동 소수점 숫자로 수행됩니다. 자세한 내용은 [고급 문서](gpu_advanced.md)를 참조하세요.

GPU 추론의 또 다른 이점은 전력 효율성입니다. GPU는 매우 효율적이고 최적화된 방식으로 계산을 수행하므로 같은 작업이 CPU에서 실행될 때보다 전력을 덜 소비하고 열을 덜 발생시킵니다.

## Demo app tutorials

The easiest way to try out the GPU delegate is to follow the below tutorials, which go through building our classification demo applications with GPU support. The GPU code is only binary for now; it will be open-sourced soon. Once you understand how to get our demos working, you can try this out on your own custom models.

### Android (with Android Studio)

For a step-by-step tutorial, watch the [GPU Delegate for Android](https://youtu.be/Xkhgre8r5G0) video.

Note: This requires OpenCL or OpenGL ES (3.1 or higher).

#### Step 1. Clone the TensorFlow source code and open it in Android Studio

```sh
git clone https://github.com/tensorflow/tensorflow
```

#### Step 2. Edit `app/build.gradle` to use the nightly GPU AAR

Add the `tensorflow-lite-gpu` package alongside the existing `tensorflow-lite` package in the existing `dependencies` block.

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite:2.3.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.3.0'
}
```

#### Step 3. Build and run

Run → Run ‘app’. When you run the application you will see a button for enabling the GPU. Change from quantized to a float model and then click GPU to run on the GPU.

![running android gpu demo and switch to gpu](images/android_gpu_demo.gif)

### iOS (with XCode)

For a step-by-step tutorial, watch the [GPU Delegate for iOS](https://youtu.be/a5H4Zwjp49c) video.

Note: This requires XCode v10.1 or later.

#### Step 1. Get the demo source code and make sure it compiles.

Follow our iOS Demo App [tutorial](https://www.tensorflow.org/lite/demo_ios). This will get you to a point where the unmodified iOS camera demo is working on your phone.

#### Step 2. Modify the Podfile to use the TensorFlow Lite GPU CocoaPod

2.3.0 릴리스부터 바이너리 크기를 줄이기 위해 기본적으로 GPU 대리자가 포드에서 제외됩니다. 하위 사양을 지정하여 이를 포함할 수 있습니다. `TensorFlowLiteSwift` 포드의 경우:

```ruby
pod 'TensorFlowLiteSwift/Metal', '~> 0.0.1-nightly',
```

OR

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly', :subspecs => ['Metal']
```

Objective-C(2.4.0 릴리스부터) 또는 C API를 사용하려는 경우 `TensorFlowLiteObjC` 또는 `TensorFlowLitC`에 대해서도 유사하게 수행할 수 있습니다.

<div>
  <devsite-expandable>
    <h4 class="showalways">2.3.0 릴리스 이전</h4>
    <h4>Until TensorFlow Lite 2.0.0</h4>
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

#### Step 3. Enable the GPU delegate

To enable the code that will use the GPU delegate, you will need to change `TFLITE_USE_GPU_DELEGATE` from 0 to 1 in `CameraExampleViewController.h`.

```c
#define TFLITE_USE_GPU_DELEGATE 1
```

#### Step 4. Build and run the demo app

After following the previous step, you should be able to run the app.

#### Step 5. Release mode

While in Step 4 you ran in debug mode, to get better performance, you should change to a release build with the appropriate optimal Metal settings. In particular, To edit these settings go to the `Product > Scheme > Edit Scheme...`. Select `Run`. On the `Info` tab, change `Build Configuration`, from `Debug` to `Release`, uncheck `Debug executable`.

![setting up release](images/iosdebug.png)

Then click the `Options` tab and change `GPU Frame Capture` to `Disabled` and `Metal API Validation` to `Disabled`.

![setting up metal options](images/iosmetal.png)

Lastly make sure to select Release-only builds on 64-bit architecture. Under `Project navigator -> tflite_camera_example -> PROJECT -> tflite_camera_example -> Build Settings` set `Build Active Architecture Only > Release` to Yes.

![setting up release options](images/iosrelease.png)

## Trying the GPU delegate on your own model

### Android

참고: TensorFlow Lite 인터프리터는 실행될 때와 같은 스레드에서 생성되어야 합니다. 그렇지 않으면, `TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized.`가 발생할 수 있습니다.

[Android Studio ML 모델 바인딩](../inference_with_metadata/codegen#acceleration) 또는 TensorFlow Lite 인터프리터를 사용하는 지에 따라 모델 가속을 호출하는 두 가지 방법이 있습니다.

#### TensorFlow Lite 인터프리터

Look at the demo to see how to add the delegate. In your application, add the AAR as above, import `org.tensorflow.lite.gpu.GpuDelegate` module, and use the`addDelegate` function to register the GPU delegate to the interpreter:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">    import org.tensorflow.lite.Interpreter
    import org.tensorflow.lite.gpu.CompatibilityList
    import org.tensorflow.lite.gpu.GpuDelegate
&lt;/div&gt;
&lt;pre data-md-type="block_code" data-md-language=""&gt;&lt;code&gt;GL_CODE_32&lt;/code&gt;</pre>
<div data-md-type="block_html">
</div>
</section></devsite-selector>
</div>
<h3 data-md-type="header" data-md-header-level="3">iOS</h3>
<p data-md-type="paragraph">참고: GPU 대리자는 Objective-C 코드에 C API를 사용할 수도 있습니다. TensorFlow Lite 2.4.0 릴리스 이전에는 이것이 유일한 옵션이었습니다.</p>
<div data-md-type="block_html">
<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    import TensorFlowLite
&lt;/div&gt;
&lt;pre data-md-type="block_code" data-md-language=""&gt;&lt;code&gt;GL_CODE_33&lt;/code&gt;</pre>
<div data-md-type="block_html">
</div>
</section></devsite-selector>
</div>
<h2 data-md-type="header" data-md-header-level="2">Supported Models and Ops</h2>
<p data-md-type="paragraph">With the release of the GPU delegate, we included a handful of models that can be run on the backend:</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">
<a href="https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html" data-md-type="link">MobileNet v1 (224x224) 이미지 분류</a> <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobilenet_v1_1.0_224.tflite" data-md-type="link">[다운로드]</a> <br><i data-md-type="raw_html">(모바일 및 임베디드 기반 비전 애플리케이션을 위해 설계된 이미지 분류 모델)</i>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<a href="https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html" data-md-type="link">DeepLab 분할(257x257)</a> <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite" data-md-type="link">[다운로드]</a> <br><i data-md-type="raw_html">(입력 이미지의 모든 픽셀에 의미론적 레이블(예: 개, 고양이, 자동차)을 할당하는 이미지 분할 모델)</i>
</li>
<li data-md-type="list_item" data-md-list-type="unordered"> <a href="https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html" data-md-type="link">MobileNet SSD 객체 감지</a> <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite" data-md-type="link">[다운로드]</a> <br><i data-md-type="raw_html">(경계 상자가 있는 여러 객체를 감지하는 이미지 분류 모델)</i>
</li>
<li data-md-type="list_item" data-md-list-type="unordered"> <a href="https://github.com/tensorflow/tfjs-models/tree/master/posenet" data-md-type="link">포즈 추정을 위한 PoseNet</a> <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite" data-md-type="link">[다운로드]</a> <br><i data-md-type="raw_html">(이미지 또는 비디오에서 사람의 포즈를 추정하는 비전 모델)</i>
</li>
</ul>
<p data-md-type="paragraph">To see a full list of supported ops, please see the <a href="gpu_advanced.md" data-md-type="link">advanced documentation</a>.</p>
<h2 data-md-type="header" data-md-header-level="2">Non-supported models and ops</h2>
<p data-md-type="paragraph">If some of the ops are not supported by the GPU delegate, the framework will only run a part of the graph on the GPU and the remaining part on the CPU. Due to the high cost of CPU/GPU synchronization, a split execution mode like this will often result in slower performance than when the whole network is run on the CPU alone. In this case, the user will get a warning like:</p>
<pre data-md-type="block_code" data-md-language="none"><code class="language-none">WARNING: op code #42 cannot be handled by this delegate.
</code></pre>
<p data-md-type="paragraph">We did not provide a callback for this failure, as this is not a true run-time failure, but something that the developer can observe while trying to get the network to run on the delegate.</p>
<h2 data-md-type="header" data-md-header-level="2">Tips for optimization</h2>
<p data-md-type="paragraph">CPU에서 사소한 일부 연산은 GPU에서는 높은 비용이 발생할 수 있습니다. 이러한 연산의 한 클래스는 <code data-md-type="codespan">BATCH_TO_SPACE</code>, <code data-md-type="codespan">SPACE_TO_BATCH</code>, <code data-md-type="codespan">SPACE_TO_DEPTH</code> 등 다양한 형태의 reshape 연산입니다. 네트워크 설계자의 논리적 사고만을 위해 연산을 네트워크에 삽입한 경우, 성능을 고려해서 해당 연산을 제거하는 것이 좋습니다.</p>
<p data-md-type="paragraph">GPU에서 텐서 데이터는 4채널로 분할됩니다. 따라서 형상 <code data-md-type="codespan">[B, H, W, 5]</code> 텐서에 대한 계산은 형상 <code data-md-type="codespan">[B, H, W, 8]</code> 텐서와 거의 동일하게 수행되지만 <code data-md-type="codespan">[B, H, W, 4]</code>에 비해서는 성능이 훨씬 나쁩니다.</p>
<p data-md-type="paragraph">그런 의미에서 카메라 하드웨어가 RGBA의 이미지 프레임을 지원하는 경우 메모리 사본(3채널 RGB에서 4채널 RGBX로)을 피할 수 있으므로 해당 4채널 입력을 훨씬 빠르게 공급할 수 있습니다.</p>
<p data-md-type="paragraph">For best performance, do not hesitate to retrain your classifier with a mobile-optimized network architecture. That is a significant part of optimization for on-device inference.</p>
</div>
