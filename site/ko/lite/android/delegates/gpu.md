# 인터프리터 API를 사용한 GPU 가속 대리자

GPU(그래픽 처리 장치)를 사용하여 머신러닝(ML) 모델을 실행하면 ML 지원 애플리케이션의 성능과 사용자 경험을 크게 향상시킬 수 있습니다. Android 기기에서는 다음을 활성화할 수 있습니다.

[*대리자*](../../performance/delegates) 및 다음 API 중 하나:

- 인터프리터 API - 이 가이드
- 작업 라이브러리 API - [가이드](./gpu_task)
- 네이티브 (C/C++) API - [가이드](./gpu_native)

이 페이지는 인터프리터 API를 사용하여 Android 앱에서 TensorFlow Lite 모델용 GPU 가속을 활성화하는 방법을 설명합니다. 모범 사례와 고급 기법을 포함한 TensorFlow Lite용 GPU 대리자 사용에 대한 자세한 내용은 [GPU 대리자](../../performance/gpu) 페이지를 참조하세요.

## Google Play 서비스와 TensorFlow Lite로 GPU 사용하기

TensorFlow Lite [인터프리터 API](https://tensorflow.org/lite/api_docs/java/org/tensorflow/lite/InterpreterApi)는 머신러닝 애플리케이션을 빌드하는 목적을 가진 일반 API 세트를 제공합니다. 이 섹션은 Google Play 서비스와 TensorFlow Lite로 이러한 API와 함께 GPU 가속기 대리자를 사용하는 방법에 대해 설명합니다.

[Google Play 서비스를 사용하는 TensorFlow Lite](../play_services)는 Android에서 TensorFlow Lite를 사용하기 위한 권장 경로입니다. 애플리케이션이 Google Play를 실행하지 않는 기기를 대상으로 하는 경우 [인터프리터 API 및 독립형 TensorFlow Lite를 사용하는 GPU](#standalone) 섹션을 참조하세요.

### 프로젝트 종속성 추가

GPU 대리자에 대한 액세스를 활성화하려면 앱의 `build.gradle` 파일에 `com.google.android.gms:play-services-tflite-gpu`를 추가하세요.

```
dependencies {
    ...
    implementation 'com.google.android.gms:play-services-tflite-java:16.0.1'
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.1.0'
}
```

### GPU 가속화 사용

그런 다음 GPU를 지원하는 Google Play 서비스로 TensorFlow Lite를 초기화합니다:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">    val useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context)

    val interpreterTask = useGpuTask.continueWith { useGpuTask -&gt;
      TfLite.initialize(context,
          TfLiteInitializationOptions.builder()
          .setEnableGpuDelegateSupport(useGpuTask.result)
          .build())
      }
      </pre>
    </section>
    <section>
      <h3> Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    Task&lt;boolean&gt; useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context);

    Task&lt;Options&gt; interpreterOptionsTask = useGpuTask.continueWith({ task -&gt;
      TfLite.initialize(context,
      TfLiteInitializationOptions.builder()
        .setEnableGpuDelegateSupport(true)
        .build());
    });
      </pre>
    </section>
  </devsite-selector>
</div>

마지막으로 `InterpreterApi.Options`를 통해 `GpuDelegateFactory`를 전달하는 인터프리터를 초기화할 수 있습니다.

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">
    val options = InterpreterApi.Options()
      .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
      .addDelegateFactory(GpuDelegateFactory())

    val interpreter = InterpreterApi(model, options)

    // Run inference
    writeToInput(input)
    interpreter.run(input, output)
    readFromOutput(output)
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">
    Options options = InterpreterApi.Options()
      .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
      .addDelegateFactory(new GpuDelegateFactory());

    Interpreter interpreter = new InterpreterApi(model, options);

    // Run inference
    writeToInput(input);
    interpreter.run(input, output);
    readFromOutput(output);
      </pre>
    </section>
  </devsite-selector>
</div>

참고: GPU 대리자는 실행되는 것과 같은 스레드에서 생성되어야 합니다. 그렇지 않으면 오류(`TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized`)가 발생할 수 있습니다.

GPU 대리자는 또한 Android Studio에서 연결된 ML 모델과 사용할 수 있습니다. 더 자세한 정보는 [메타데이터를 사용하는 일반 모델 인터페이스](../../inference_with_metadata/codegen#acceleration)를 참조하세요.

## 독립형 TensorFlow Lite로 GPU 사용하기{:#standalone}

애플리케이션이 Google Play를 실행하지 않는 기기를 대상으로 하는 경우, 애플리케이션에 GPU 대리자를 번들로 제공하여 독립형 버전의 TensorFlow Lite와 함께 사용할 수 있습니다.

### 프로젝트 종속성 추가

GPU 대리자에 대한 액세스를 활성화하려면 `org.tensorflow:tensorflow-lite-gpu-delegate-plugin`을 앱의 `build.gradle` 파일에 추가하세요.

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite'
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

### GPU 가속화 사용

`TfLiteDelegate`로 GPU에서 TensorFlow Lite를 실행합니다. Java에서는 `Interpreter.Options`를 통해 `GpuDelegate`를 지정할 수 있습니다.

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

### 양자화 모델 {:#quantized-models}

Android GPU 대리자 라이브러리는 기본으로 양자화 모델을 지원합니다. GPU 대리자를 통해 양자화 모델을 사용해 모드를 변경할 필요가 없습니다. 다음 섹션은 테스트 또는 실험 목적으로 양자화 지원을 비활성화하는 방법을 설명합니다.

#### 양자화 모델 지원 비활성화

다음 코드는 양자화 모델에 대한 지원을 ***비활성화***하는 방법을 보여줍니다.

<div>
  <devsite-selector>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">GpuDelegate delegate = new GpuDelegate(new GpuDelegate.Options().setQuantizedModelsAllowed(false));

Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
      </pre>
    </section>
  </devsite-selector>
</div>

GPU 가속화로 양자화 모델을 실행하는 데 대한 더 자세한 정보는 [GPU 대리자](../../performance/gpu#quantized-models) 개요를 참조하세요.
