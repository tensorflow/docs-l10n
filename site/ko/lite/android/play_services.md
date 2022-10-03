# TensorFlow Lite in Google Play services

TensorFlow Lite is available in Google Play services runtime for all Android devices running the current version of Play services. This runtime allows you to run machine learning (ML) models without statically bundling TensorFlow Lite libraries into your app.

With the Google Play services API, you can reduce the size of your apps and gain improved performance from the latest stable version of the libraries. TensorFlow Lite in Google Play services is the recommended way to use TensorFlow Lite on Android.

You can get started with the Play services runtime with the [Quickstart](../android/quickstart), which provides a step-by-step guide to implement a sample application. If you are already using stand-alone TensorFlow Lite in your app, refer to the [Migrating from stand-alone TensorFlow Lite](#migrating) section to update an existing app to use the Play services runtime. For more information about Google Play services, see the [Google Play services](https://developers.google.com/android/guides/overview) website.

<aside class="note"> <b>Terms:</b> By accessing or using TensorFlow Lite in Google Play services APIs, you agree to the <a href="#tos">Terms of Service</a>. Please read and understand all applicable terms and policies before accessing the APIs. </aside>

## Using the Play services runtime

TensorFlow Lite in Google Play services is available through the [TensorFlow Lite Task API](../api_docs/java/org/tensorflow/lite/task/core/package-summary) and [TensorFlow Lite Interpreter API](../api_docs/java/org/tensorflow/lite/InterpreterApi). The Task Library provides optimized out-of-box model interfaces for common machine learning tasks using visual, audio, and text data. The TensorFlow Lite Interpreter API, provided by the TensorFlow runtime and support libraries, provides a more general-purpose interface for building and running ML models.

The following sections provide instructions on how to implement the Interpreter and Task Library APIs in Google Play services. While it is possible for an app to use both the Interpreter APIs and Task Library APIs, most apps should only use one set of APIs.

### Using the Task Library APIs

The TensorFlow Lite Task API wraps the Interpreter API and provides a high-level programming interface for common machine learning tasks that use visual, audio, and text data. You should use the Task API if your application requires one of the [supported tasks](../inference_with_metadata/task_library/overview#supported_tasks).

#### 1. 프로젝트 종속성 추가

Your project dependency depends on your machine learning use case. The Task APIs contain the following libraries:

- Vision library: `org.tensorflow:tensorflow-lite-task-vision-play-services`
- Audio library: `org.tensorflow:tensorflow-lite-task-audio-play-services`
- Text library: `org.tensorflow:tensorflow-lite-task-text-play-services`

Add one of the dependencies to your app project code to access the Play services API for TensorFlow Lite. For example, use the following to implement a vision task:

```
dependencies {
...
    implementation 'org.tensorflow:tensorflow-lite-task-vision-play-services:0.4.2'
...
}
```

Caution: The TensorFlow Lite Tasks Audio library version 0.4.2 maven repository is incomplete. Use version 0.4.2.1 for this library instead: `org.tensorflow:tensorflow-lite-task-audio-play-services:0.4.2.1`.

#### 2. TensorFlow Lite 초기화 추가

Initialize the TensorFlow Lite component of the Google Play services API *before* using the TensorFlow Lite APIs. The following example initializes the vision library:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">init {
  TfLiteVision.initialize(context)
    }
  }
</pre>
    </section>
  </devsite-selector>
</div>

Important: Make sure the `TfLite.initialize` task completes before executing code that accesses TensorFlow Lite APIs.

Tip: The TensorFlow Lite modules are installed at the same time your application is installed or updated from the Play Store. You can check the availability of the modules by using `ModuleInstallClient` from the Google Play services APIs. For more information on checking module availability, see [Ensuring API availability with ModuleInstallClient](https://developers.google.com/android/guides/module-install-apis).

#### 3. Run inferences

After initializing the TensorFlow Lite component, call the `detect()` method to generate inferences. The exact code within the `detect()` method varies depending on the library and use case. The following is for a simple object detection use case with the `TfLiteVision` library:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">fun detect(...) {
  if (!TfLiteVision.isInitialized()) {
    Log.e(TAG, "detect: TfLiteVision is not initialized yet")
    return
  }

  if (objectDetector == null) {
    setupObjectDetector()
  }

  ...

}
</pre>
    </section>
  </devsite-selector>
</div>

Depending on the data format, you may also need to preprocess and convert your data within the `detect()` method before generating inferences. For example, image data for an object detector requires the following:

```kotlin
val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()
val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
val results = objectDetector?.detect(tensorImage)
```

### Using the Interpreter APIs

The Interpreter APIs offer more control and flexibility than the Task Library APIs. You should use the Interpreter APIs if your machine learning task is not supported by the Task library, or if you require a more general-purpose interface for building and running ML models.

#### 1. Add project dependencies

Add the following dependencies to your app project code to access the Play services API for TensorFlow Lite:

```
dependencies {
...
    // Tensorflow Lite dependencies for Google Play services
    implementation 'com.google.android.gms:play-services-tflite-java:16.0.0'
    // Optional: include Tensorflow Lite Support Library
    implementation 'com.google.android.gms:play-services-tflite-support:16.0.0'
...
}
```

#### 2. Add initialization of TensorFlow Lite

Initialize the TensorFlow Lite component of the Google Play services API *before* using the TensorFlow Lite APIs:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">val initializeTask: Task&lt;Void&gt; by lazy { TfLite.initialize(this) }
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">Task&lt;Void&gt; initializeTask = TfLite.initialize(context);
</pre>
    </section>
  </devsite-selector>
</div>

참고: TensorFlow Lite API에 액세스하는 코드를 실행하기 전에 `TfLite.initialize` 작업이 완료되었는지 확인하세요. 다음 섹션에 표시된 대로 `addOnSuccessListener()` 메서드를 사용하세요.

#### 3. Create an Interpreter and set runtime option {:#step_3_interpreter}

Create an interpreter using `InterpreterApi.create()` and configure it to use Google Play services runtime, by calling `InterpreterApi.Options.setRuntime()`, as shown in the following example code:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
...
private lateinit var interpreter: InterpreterApi
...
initializeTask.addOnSuccessListener {
  val interpreterOption =
    InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
  interpreter = InterpreterApi.create(
    modelBuffer,
    interpreterOption
  )}
  .addOnFailureListener { e -&gt;
    Log.e("Interpreter", "Cannot initialize interpreter", e)
  }
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
...
private InterpreterApi interpreter;
...
initializeTask.addOnSuccessListener(a -&gt; {
    interpreter = InterpreterApi.create(modelBuffer,
      new InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY));
  })
  .addOnFailureListener(e -&gt; {
    Log.e("Interpreter", String.format("Cannot initialize interpreter: %s",
          e.getMessage()));
  });
</pre>
    </section>
  </devsite-selector>
</div>

Android 사용자 인터페이스 스레드를 차단하지 않기 때문에 위의 구현을 사용해야 합니다. 스레드 실행을 더 면밀하게 관리해야 하는 경우, 인터프리터 생성에 `Tasks.await()` 호출을 추가할 수 있습니다.

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">import androidx.lifecycle.lifecycleScope
...
lifecycleScope.launchWhenStarted { // uses coroutine
  initializeTask.await()
}
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">@BackgroundThread
InterpreterApi initializeInterpreter() {
    Tasks.await(initializeTask);
    return InterpreterApi.create(...);
}
</pre>
    </section>
  </devsite-selector>
</div>

경고: 포그라운드 사용자 인터페이스 스레드에서 `.await()`를 호출하지 마세요. 사용자 인터페이스 요소의 표시를 방해하고 사용자 경험이 나빠지기 때문입니다.

#### 4. 추론 실행

생성한 `interpreter` 객체로 `run()` 메서드를 호출하여 추론을 생성합니다.

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">interpreter.run(inputBuffer, outputBuffer)
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">interpreter.run(inputBuffer, outputBuffer);
</pre>
    </section>
  </devsite-selector>
</div>

## Hardware acceleration {:#hardware-acceleration}

TensorFlow Lite allows you to accelerate the performance of your model using specialized hardware processors, such as graphics processing units (GPUs). You can take advantage of these specialized processors using hardware drivers called [*delegates*](https://www.tensorflow.org/lite/performance/delegates). You can use the following hardware acceleration delegates with TensorFlow Lite in Google Play services:

- *[GPU delegate](https://www.tensorflow.org/lite/performance/gpu) (recommended)* - This delegate is provided through Google Play services and is dynamically loaded, just like the Play services versions of the Task API and Interpreter API.

- [*NNAPI delegate*](https://www.tensorflow.org/lite/android/delegates/nnapi) - This delegate is available as an included library dependency in your Android development project, and is bundled into your app.

For more information about hardware acceleration with TensorFlow Lite, see the [TensorFlow Lite Delegates](https://www.tensorflow.org/lite/performance/delegates) page.

### Checking device compatibility

Not all devices support GPU hardware acceleration with TFLite. In order to mitigate errors and potential crashes, use the `TfLiteGpu.isGpuDelegateAvailable` method to check whether a device is compatible with the GPU delegate.

Use this method to confirm whether a device is compatible with GPU, and use CPU or the NNAPI delegate as a fallback for when GPU is not supported.

```
useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context)
```

Once you have a variable like `useGpuTask`, you can use it to determine whether devices use the GPU delegate. The following examples show how this can be done with both the Task Library and Interpreter APIs.

**With the Task Api**

<div>
<devsite-selector>
<section>
  <h3>Kotlin</h3>
    <pre class="prettyprint">lateinit val optionsTask = useGpuTask.continueWith { task -&gt;
  val baseOptionsBuilder = BaseOptions.builder()
  if (task.result) {
    baseOptionsBuilder.useGpu()
  }
 ObjectDetectorOptions.builder()
          .setBaseOptions(baseOptionsBuilder.build())
          .setMaxResults(1)
          .build()
}
    </pre>
</section>
<section>
  <h3>Java</h3>
    <pre class="prettyprint">Task&lt;ObjectDetectorOptions&gt; optionsTask = useGpuTask.continueWith({ task -&gt;
  BaseOptions baseOptionsBuilder = BaseOptions.builder();
  if (task.getResult()) {
    baseOptionsBuilder.useGpu();
  }
  return ObjectDetectorOptions.builder()
          .setBaseOptions(baseOptionsBuilder.build())
          .setMaxResults(1)
          .build()
});
    </pre>
</section>
</devsite-selector>
</div>

**With the Interpreter Api**

<div>
<devsite-selector>
<section>
  <h3>Kotlin</h3>
    <pre class="prettyprint">val interpreterTask = useGpuTask.continueWith { task -&gt;
  val interpreterOptions = InterpreterApi.Options()
      .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
  if (task.result) {
      interpreterOptions.addDelegateFactory(GpuDelegateFactory())
  }
  InterpreterApi.create(FileUtil.loadMappedFile(context, MODEL_PATH), interpreterOptions)
}
    </pre>
</section>
<section>
  <h3>Java</h3>
    <pre class="prettyprint">Task&lt;InterpreterApi.Options&gt; interpreterOptionsTask = useGpuTask.continueWith({ task -&gt;
  InterpreterApi.Options options =
      new InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY);
  if (task.getResult()) {
     options.addDelegateFactory(new GpuDelegateFactory());
  }
  return options;
});
    </pre>
</section>
</devsite-selector>
</div>

### GPU with Task Library APIs

To use the GPU delegate with the Task APIs:

1. Update the project dependencies to use the GPU delegate from Play services:

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0'
    ```

2. Initialize the GPU delegate with `setEnableGpuDelegateSupport`. For example, you can initialize the GPU delegate for `TfLiteVision` with the following:

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">        TfLiteVision.initialize(context, TfLiteInitializationOptions.builder().setEnableGpuDelegateSupport(true).build())
            </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">        TfLiteVision.initialize(context, TfLiteInitializationOptions.builder().setEnableGpuDelegateSupport(true).build());
            </pre>
    </section>
    </devsite-selector>
    </div>

3. Enable the GPU delegate option with [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder):

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">        val baseOptions = BaseOptions.builder().useGpu().build()
            </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">        BaseOptions baseOptions = BaseOptions.builder().useGpu().build();
            </pre>
    </section>
    </devsite-selector>
    </div>

4. Configure the options using `.setBaseOptions`. For example, you can set up GPU in `ObjectDetector` with the following:

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">        val options =
                ObjectDetectorOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setMaxResults(1)
                    .build()
            </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">        ObjectDetectorOptions options =
                ObjectDetectorOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setMaxResults(1)
                    .build();
            </pre>
    </section>
    </devsite-selector>
    </div>

### GPU with Interpreter APIs

To use the GPU delegate with the Interpreter APIs:

1. Update the project dependencies to use the GPU delegate from Play services:

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0'
    ```

2. Enable the GPU delegate option in the TFlite initialization:

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">        TfLite.initialize(context,
              TfLiteInitializationOptions.builder()
               .setEnableGpuDelegateSupport(true)
               .build())
            </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">        TfLite.initialize(context,
              TfLiteInitializationOptions.builder()
               .setEnableGpuDelegateSupport(true)
               .build());
            </pre>
    </section>
    </devsite-selector>
    </div>

3. Set GPU delegate in interpreter options to use `DelegateFactory` by calling `addDelegateFactory()` within `InterpreterApi.Options()`:

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">        val interpreterOption = InterpreterApi.Options()
             .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
             .addDelegateFactory(GpuDelegateFactory())
            </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">        Options interpreterOption = InterpreterApi.Options()
              .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
              .addDelegateFactory(new GpuDelegateFactory());
            </pre>
    </section>
    </devsite-selector>
    </div>

## Migrating from stand-alone TensorFlow Lite {:#migrating}

앱을 독립형 TensorFlow Lite에서 Play Services API로 마이그레이션하려는 경우, 앱 프로젝트 코드 업데이트에 대한 다음 추가 지침을 검토하세요.

1. 이 페이지의 [제한 사항](#limitations) 섹션을 검토하여 해당 사례가 지원되는지 확인합니다.
2. 코드를 업데이트하기 전에 특히 TensorFlow Lite 버전 2.1 이전 버전을 사용하는 경우 모델에 대한 성능 및 정확도 검사를 수행하여 새 구현과 비교할 기준선을 확보하세요.
3. TensorFlow Lite용 Play Services API를 사용하도록 모든 코드를 마이그레이션한 경우 build.gradle 파일에서 기존 TensorFlow Lite *런타임 라이브러리* 종속성(<code>org.tensorflow:**tensorflow-lite**:*</code> 항목)을 제거해야 합니다. 그러면 파일 크기를 줄일 수 있습니다.
4. Identify all occurrences of `new Interpreter` object creation in your code, and modify it so that it uses the InterpreterApi.create() call. This new API is asynchronous, which means in most cases it's not a drop-in replacement, and you must register a listener for when the call completes. Refer to the code snippet in [Step 3](#step_3_interpreter) code.
5. `org.tensorflow.lite.Interpreter` 또는 `org.tensorflow.lite.InterpreterApi` 클래스를 사용하여 모든 소스 파일에 `import org.tensorflow.lite.InterpreterApi;` 및 `import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;`를 추가합니다.
6. `InterpreterApi.create()`에 대한 결과적인 호출에 단일 인수만 있는 경우, 인수 목록에 `new InterpreterApi.Options()`를 추가합니다.
7. `InterpreterApi.create()`에 대한 호출의 마지막 인수에 `.setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)`을 추가합니다.
8. `org.tensorflow.lite.Interpreter` 클래스의 다른 모든 항목을 `org.tensorflow.lite.InterpreterApi` 로 교체하세요.

독립형 TensorFlow Lite와 Play Services API를 나란히 사용하려면 TensorFlow Lite 2.9(또는 그 이상)를 사용해야 합니다. TensorFlow Lite 2.8 및 이전 버전은 Play Services API 버전과 호환되지 않습니다.

## 한계

TensorFlow Lite in Google Play services has the following limitations:

- Support for hardware acceleration delegates is limited to the delegates listed in the [Hardware acceleration](#hardware-acceleration) section. No other acceleration delegates are supported.
- [기본 API](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c)를 통한 TensorFlow Lite 액세스는 지원되지 않습니다. TensorFlow Lite Java API만 Google Play Services를 통해 사용할 수 있습니다.
- 사용자 지정 작업을 포함하여 실험적이거나 더 이상 사용되지 않는 TensorFlow Lite API는 지원되지 않습니다.

## 지원 및 피드백 {:#support}

You can provide feedback and get support through the TensorFlow Issue Tracker. Please report issues and support requests using the [Issue template](https://github.com/tensorflow/tensorflow/issues/new?title=TensorFlow+Lite+in+Play+Services+issue&template=tflite-in-play-services.md) for TensorFlow Lite in Google Play services.

## Terms of service {:#tos}

Use of TensorFlow Lite in Google Play services APIs is subject to the [Google APIs Terms of Service](https://developers.google.com/terms/).

### Privacy and data collection

When you use TensorFlow Lite in Google Play services APIs, processing of the input data, such as images, video, text, fully happens on-device, and TensorFlow Lite in Google Play services APIs does not send that data to Google servers. As a result, you can use our APIs for processing data that should not leave the device.

Google Play Services API의 TensorFlow Lite는 버그 수정, 업데이트된 모델 및 하드웨어 가속기 호환성 정보와 같은 정보를 수신하기 위해 때때로 Google 서버에 연결할 수 있습니다. Google Play Services API의 TensorFlow Lite는 앱에서 API의 성능 및 활용도에 대한 메트릭도 Google에 보냅니다. Google은 [개인정보 보호정책](https://policies.google.com/privacy)에 자세히 설명된 대로 이 메트릭 데이터를 사용하여 성능을 측정하고, API를 디버그, 관리 및 개선하며, 오용 또는 남용을 감시합니다.

**You are responsible for informing users of your app about Google's processing of TensorFlow Lite in Google Play services APIs metrics data as required by applicable law.**

수집되는 데이터에는 다음이 포함됩니다.

- 장치 정보(예: 제조업체, 모델, OS 버전 및 빌드) 및 사용 가능한 ML 하드웨어 가속기(GPU 및 DSP). 진단 및 사용 분석에 사용됩니다.
- 진단 및 사용 분석에 사용되는 장치 식별자
- 앱 정보(패키지 이름, 앱 버전). 진단 및 사용 분석에 사용됩니다.
- API 구성(예: 사용 중인 대리자). 진단 및 사용 분석에 사용됩니다.
- 이벤트 유형(예: 인터프리터 생성, 추론). 진단 및 사용 분석에 사용됩니다.
- 오류 코드. 진단에 사용됩니다.
- 성능 메트릭. 진단에 사용됩니다.

## 다음 단계

TensorFlow Lite를 사용하여 모바일 애플리케이션에서 머신 러닝을 구현하는 방법에 대한 자세한 내용은 [TensorFlow Lite 개발자 가이드](https://www.tensorflow.org/lite/guide)를 참조하세요. [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite)에서 이미지 분류, 객체 감지 및 기타 애플리케이션을 위한 추가 TensorFlow Lite 모델을 찾을 수 있습니다.
