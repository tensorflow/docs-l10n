# Google Play 서비스의 TensorFlow Lite

TensorFlow Lite는 현재 버전의 Play 서비스를 실행하는 모든 Android 기기에서 Google Play 서비스 런타임에서 사용할 수 있습니다. 이 런타임을 사용하면 TensorFlow Lite 라이브러리를 앱에 정적으로 번들링하지 않고도 머신러닝(ML) 모델을 실행할 수 있습니다.

Google Play 서비스 API를 사용하여 앱의 규모를 줄이고 라이브러리의 안정적인 최신 버전에서 개선된 성능을 얻을 수 있습니다. Google Play 서비스의 TensorFlow Lite는 Android에서 TensorFlow Lite를 사용하는 데 권장되는 방법입니다.

샘플 애플리케이션을 구현하는 데 단계별 가이드를 제공하는 [Quickstart](../android/quickstart)를 통해 Play 서비스 런타임으로 시작할 수 있습니다. 이미 독립형 TensorFlow Lite를 앱에서 사용하고 있다면 [독립형 TensorFlow Lite에서 마이그레이션](#migrating) 섹션을 참조하여 기존 앱을 업데이트해 Play 서비스 런타임을 사용하세요. Google Play 서비스에 대한 더 자세한 정보는 [Google Play 서비스](https://developers.google.com/android/guides/overview) 웹사이트를 확인하세요.

<aside class="note"><b>약관:</b> Google Play 서비스 API에서 TensorFlow Lite에 액세스하거나 사용함으로써, <a href="#tos">서비스 약관</a>에 동의하게 됩니다. API에 액세스하기 전에 모든 해당 약관 및 정책을 읽고 이해해 주시길 바랍니다.</aside>

## Play 서비스 런타임 사용하기

Google Play 서비스의 TensorFlow Lite는 [TensorFlow Lite Task API](../api_docs/java/org/tensorflow/lite/task/core/package-summary) 및  [TensorFlow Lite Interpreter API](../api_docs/java/org/tensorflow/lite/InterpreterApi)를 통해 사용할 수 있습니다. 작업 라이브러리는 비전, 오디오 및 텍스트 데이터를 사용해 일반적인 머신러닝 작업을 위한 최적화된 바로 사용할 수 있는 모델 인터페이스를 제공합니다. TensorFlow 런타임과 지원 라이브러리가 제공하는 TensorFlow Lite Interpreter API는 ML 모델을 구축하고 실행하기 위한 더욱 일반적인 목적의 인터페이스를 제공합니다.

다음 섹션은 Google Play 서비스에서 Interpreter 및 Task Library API를 구현하는 방법에 대한 지침을 제공합니다. Interpreter API 및 Task Library API 모두를 앱이 사용하는 것은 가능하지만 대부분의 앱은 하나의 API 세트만 사용해야 합니다.

### Task Library API 사용하기

TensorFlow Lite Task API는 Interpreter API를 래핑하며 비전, 오디오 및 텍스트 데이터를 사용하는 일반적인 머신러닝 작업을 위한 고급 프로그래밍 인터페이스를 제공합니다. 애플리케이션에 [지원되는 작업](../inference_with_metadata/task_library/overview#supported_tasks) 중 하나가 필요하다면 Task API를 사용해야 합니다.

#### 1. 프로젝트 종속성 추가

프로젝트 종속성은 머신러닝 사용 사례에 따라 다릅니다. Task API는 다음 라이브러리를 포함합니다.

- 비전 라이브러리: `org.tensorflow:tensorflow-lite-task-vision-play-services`
- 오디오 라이브러리: `org.tensorflow:tensorflow-lite-task-audio-play-services`
- 텍스트 라이브러리: `org.tensorflow:tensorflow-lite-task-text-play-services`

앱 프로젝트 코드에 종속성 중 하나를 추가하여 TensorFlow Lite용 Play 서비스 API에 액세스합니다. 예를 들어, 다음을 사용하여 비전 작업을 구현하세요.

```
dependencies {
...
    implementation 'org.tensorflow:tensorflow-lite-task-vision-play-services:0.4.2'
...
}
```

주의: TensorFlow Lite Tasks Audio 라이브러리 버전 0.4.2 전문가 리포지토리는 불완전합니다. 이 라이브러리 대신 0.4.2.1 버전(`org.tensorflow:tensorflow-lite-task-audio-play-services:0.4.2.1`)을 사용하세요.

#### 2. TensorFlow Lite 초기화 추가

TensorFlow Lite API를 사용하기 *전에* Google Play 서비스 API의 TensorFlow Lite 구성요소를 초기화합니다. 다음 예시는 비전 라이브러리를 초기화합니다.

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

중요: TensorFlow Lite API에 액세스하는 코드를 실행하기 전에 `TfLite.initialize` 작업을 완료해야 합니다.

팁: TensorFlow Lite 모듈은 Play Store에서 애플리케이션이 설치되거나 업데이트될 때와 동시에 설치됩니다. Google Play 서비스 API의 `ModuleInstallClient`를 사용하여 모듈 가용성을 확인할 수 있습니다. 모듈 가용성에 관한 더 자세한 정보는 [ModuleInstallClient를 통한 API 가용성 보장](https://developers.google.com/android/guides/module-install-apis)을 확인하세요.

#### 3. 추론 실행

TensorFlow Lite 구성요소를 초기화한 후, `detect()` 메서드를 호출하여 추론을 생성합니다. `detect()` 메서드의 정확한 코드는 라이브러리와 사용 사례에 따라 다양합니다. 다음은 `TfLiteVision` 라이브러리를 통한 단순한 객체 감지 사용 사례입니다.

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

데이터 유형에 따라, 추론을 생성하기 전 `detect()` 메서드 내의 데이터를 사전 처리 및 변환해야 할 수도 있습니다. 예를 들어, 객체 감지기에 대한 이미지 데이터는 다음이 필요합니다.

```kotlin
val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()
val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
val results = objectDetector?.detect(tensorImage)
```

### Interpreter API 사용하기

Interpreter API는 Task Library API보다 더 많은 제어 기능과 유연성을 제공합니다. 머신러닝 작업을 작업 라이브러리가 지원하지 않거나 ML 모델을 구축하고 실행하는 데 더 많은 일반적인 목적의 추론이 필요하다면 Interpreter API를 사용해야 합니다.

#### 1. 프로젝트 종속성 추가

TensorFlow Lite용 Play 서비스 API에 액세스하려면 앱 프로젝트 코드에 다음 종속성을 추가하세요.

```
dependencies {
...
    // Tensorflow Lite dependencies for Google Play services
    implementation 'com.google.android.gms:play-services-tflite-java:16.0.1'
    // Optional: include Tensorflow Lite Support Library
    implementation 'com.google.android.gms:play-services-tflite-support:16.0.1'
...
}
```

#### 2. TensorFlow Lite 초기화 추가

TensorFlow Lite API를 사용하기 *전에* Google Play 서비스 API의 TensorFlow Lite 구성요소를 초기화합니다.

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

#### 3. 인터프리터 생성 및 런타임 옵션 설정 {:#step_3_interpreter}

`InterpreterApi.create()`를 사용하여 인터프리터를 만들고 다음 예제 코드와 같이 `InterpreterApi.Options.setRuntime()`을 호출하여 Google Play 서비스 런타임을 사용하도록 이를 구성합니다.

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

## 하드웨어 가속 {:#hardware-acceleration}

TensorFlow Lite를 사용하면 그래픽 처리 장치(GPU)와 같은 전문적인 하드웨어 프로세서를 사용하여 모델의 성능을 가속할 수 있습니다. [*대리자*](https://www.tensorflow.org/lite/performance/delegates)라고 불리는 하드웨어 드라이버를 사용하여 이러한 전문적인 프로세서를 활용할 수 있습니다. Google Play 서비스에서 TensorFlow Lite를 통해 다음 하드웨어 가속 대리자를 사용할 수 있습니다.

- *[GPU 대리자](https://www.tensorflow.org/lite/performance/gpu)(권장)* - 이 대리자는 Task API 및 Interpreter API의 Play 서비스 버전과 같이 Google Play 서비스를 통해 제공되며 동적으로 로딩됩니다.

- [*NNAPI 대리자*](https://www.tensorflow.org/lite/android/delegates/nnapi) - 이 대리자는 Android 개발 프로젝트에서 내장된 라이브러리 종속성으로 사용할 수 있으며 앱에 번들링됩니다.

TensorFlow Lite를 통한 하드웨어 가속 대리자에 대한 자세한 내용은 [TensorFlow Lite 대리자](https://www.tensorflow.org/lite/performance/delegates) 페이지를 참조하세요.

### 기기 호환성 확인

모든 기기가 TFLite를 통한 GPU 하드웨어 가속을 지원하지 않습니다. 오류와 잠재적인 충돌을 완화하기 위해 `TfLiteGpu.isGpuDelegateAvailable` 메서드를 사용해 기기가 GPU 대리자와 호환되는지 확인하세요.

이 메서드를 사용하여 기기가 GPU와 호환되는지 확인하고 GPU가 지원되지 않는 경우 GPU 또는 NNAPI 대리자를 폴백으로 사용하세요.

```
useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context)
```

`useGpuTask`와 같은 변수를 갖추면 이를 사용하여 기기가 GPU 대리자를 사용하는지 확인할 수 있습니다. 다음 예시는 Task Library 및 Interpreter API 모두를 통해 이를 어떻게 할 수 있는지 보여줍니다.

**Task Api 사용**

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

**Interpreter Api 사용**

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

### Task Library API가 있는 GPU

Task API가 있는 GPU 대리자를 사용하려면 다음을 수행합니다.

1. 프로젝트 종속성을 업데이트하여 Play 서비스에서 GPU 대리자를 사용하세요.

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.1.0'
    ```

2. `setEnableGpuDelegateSupport`로 GPU 대리자를 초기화합니다. 예를 들어, 다음을 통해 `TfLiteVision`의 GPU 대리자를 초기화할 수 있습니다.

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

3. 다음과 같이 [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder)를 통해 GPU 대리자 옵션을 활성화합니다.

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

4. `.setBaseOptions`를 사용해 옵션을 구성합니다. 예를 들어, 다음을 통해 `ObjectDetector`에서 GPU를 설정할 수 있습니다.

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

### Interpreter API가 있는 GPU

Interpreter API가 있는 GPU 대리자를 사용하려면 다음을 수행합니다.

1. 프로젝트 종속성을 업데이트하여 Play 서비스에서 GPU 대리자를 사용하세요.

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.1.0'
    ```

2. TFlite 초기화의 GPU 대리자 옵션을 활성화합니다.

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

3. `InterpreterApi.Options()` 내의 `addDelegateFactory()`를 호출하여 인터프리터 옵션의 GPU 대리자를 설정해 `DelegateFactory`를 사용합니다.

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

## 독립형 TensorFlow Lite에서 마이그레이션 {:#migrating}

앱을 독립형 TensorFlow Lite에서 Play Services API로 마이그레이션하려는 경우, 앱 프로젝트 코드 업데이트에 대한 다음 추가 지침을 검토하세요.

1. 이 페이지의 [제한 사항](#limitations) 섹션을 검토하여 해당 사례가 지원되는지 확인합니다.
2. 코드를 업데이트하기 전에 특히 TensorFlow Lite 버전 2.1 이전 버전을 사용하는 경우 모델에 대한 성능 및 정확도 검사를 수행하여 새 구현과 비교할 기준선을 확보하세요.
3. TensorFlow Lite용 Play Services API를 사용하도록 모든 코드를 마이그레이션한 경우 build.gradle 파일에서 기존 TensorFlow Lite *런타임 라이브러리* 종속성(<code>org.tensorflow:**tensorflow-lite**:*</code> 항목)을 제거해야 합니다. 그러면 파일 크기를 줄일 수 있습니다.
4. 코드에서 모든 `new Interpreter` 객체 생성 항목을 식별하고 InterpreterApi.create() 호출을 사용하도록 이를 수정합니다. 이 새로운 API는 비동기식입니다. 즉, 대부분의 경우 드롭인 대체가 아니며 호출이 완료될 때를 수신할 리스너를 등록해야 합니다. [3단계](#step_3_interpreter) 코드의 코드 스니펫을 참조하세요.
5. `org.tensorflow.lite.Interpreter` 또는 `org.tensorflow.lite.InterpreterApi` 클래스를 사용하여 모든 소스 파일에 `import org.tensorflow.lite.InterpreterApi;` 및 `import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;`를 추가합니다.
6. `InterpreterApi.create()`에 대한 결과적인 호출에 단일 인수만 있는 경우, 인수 목록에 `new InterpreterApi.Options()`를 추가합니다.
7. `InterpreterApi.create()`에 대한 호출의 마지막 인수에 `.setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)`을 추가합니다.
8. `org.tensorflow.lite.Interpreter` 클래스의 다른 모든 항목을 `org.tensorflow.lite.InterpreterApi` 로 교체하세요.

독립형 TensorFlow Lite와 Play Services API를 나란히 사용하려면 TensorFlow Lite 2.9(또는 그 이상)를 사용해야 합니다. TensorFlow Lite 2.8 및 이전 버전은 Play Services API 버전과 호환되지 않습니다.

## 한계

Google Play 서비스의 TensorFlow Lite는 다음과 같은 제한 사항이 있습니다.

- 하드웨어 가속 대리자에 대한 지원은 [하드웨어 가속](#hardware-acceleration) 섹션에 나열된 대리자로 한정됩니다. 다른 가속 대리자는 지원되지 않습니다.
- [기본 API](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c)를 통한 TensorFlow Lite 액세스는 지원되지 않습니다. TensorFlow Lite Java API만 Google Play Services를 통해 사용할 수 있습니다.
- 사용자 지정 작업을 포함하여 실험적이거나 더 이상 사용되지 않는 TensorFlow Lite API는 지원되지 않습니다.

## 지원 및 피드백 {:#support}

TensorFlow Issue Tracker를 통해 피드백을 제공하고 지원을 받을 수 있습니다. Google Play 서비스의 TensorFlow Lite용 [이슈 템플릿](https://github.com/tensorflow/tensorflow/issues/new?title=TensorFlow+Lite+in+Play+Services+issue&template=tflite-in-play-services.md)을 사용하여 문제 및 지원 요청을 보고하세요.

## 서비스 약관 {:#tos}

Google Play 서비스 API에서 TensorFlow Lite를 사용하는 것은 [Google API 서비스 약관](https://developers.google.com/terms/)의 대상이 됩니다.

### 개인 정보 및 데이터 수집

Google Play Services API에서 TensorFlow Lite를 사용하면 이미지, 비디오, 텍스트와 같은 입력 데이터 처리가 기기에서 완전하게 이루어지고 Google Play Services의 TensorFlow Lite는 해당 데이터를 Google 서버로 보내지 않습니다. 결과적으로 기기를 나가지 않아야 하는 데이터를 처리하는 데 API를 사용할 수 있습니다.

Google Play Services API의 TensorFlow Lite는 버그 수정, 업데이트된 모델 및 하드웨어 가속기 호환성 정보와 같은 정보를 수신하기 위해 때때로 Google 서버에 연결할 수 있습니다. Google Play Services API의 TensorFlow Lite는 앱에서 API의 성능 및 활용도에 대한 메트릭도 Google에 보냅니다. Google은 [개인정보 보호정책](https://policies.google.com/privacy)에 자세히 설명된 대로 이 메트릭 데이터를 사용하여 성능을 측정하고, API를 디버그, 관리 및 개선하며, 오용 또는 남용을 감시합니다.

**귀하는 관련 법률에서 요구하는 대로 Google Play Services 메트릭 데이터의 TensorFlow Lite 처리에 대해 앱 사용자에게 알릴 책임이 있습니다.**

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
