# 작업 라이브러리를 사용하는 GPU 가속 대리자

그래픽 처리 장치(GPU)를 사용하여 머신러닝(ML) 모델을 실행하면 ML 지원 애플리케이션의 성능과 사용자 경험을 크게 향상할 수 있습니다. Android 기기에서는 [*대리자*](../../performance/delegates) 및 다음 API 중 하나를 사용하여 모델의 GPU 가속 실행 사용을 활성화할 수 있습니다.

- 인터프리터 API - [가이드](./gpu)
- 작업 라이브러리 API - <a>이 가이드</a>
- 네이티브 (C/C++) API - 이 [가이드](./gpu_native)

이 페이지는 작업 라이브러리를 사용하여 Android 앱에서 TensorFlow Lite 모델용 GPU 가속을 활성화하는 방법을 설명합니다. 모범 사례와 고급 기법을 포함한 TensorFlow Lite용 GPU 대리자 사용에 대한 자세한 내용은 [GPU 대리자](../../performance/gpu) 페이지를 참조하세요.

## Google Play 서비스와 TensorFlow Lite로 GPU 사용하기

TensorFlow Lite [작업 라이브러리](../../inference_with_metadata/task_library/overview)는 머신러닝 애플리케이션을 빌드하는 목적을 가진 작업별 API 세트를 제공합니다. 이 섹션은 Google Play 서비스와 TensorFlow Lite로 이러한 API와 함께 GPU 가속기 대리자를 사용하는 방법에 대해 설명합니다.

[TGoogle Play 서비스를 사용하는 TensorFlow Lite](../play_services)는 Android에서 TensorFlow Lite를 사용하기 위한 권장 경로입니다. 애플리케이션이 Google Play를 실행하지 않는 기기를 대상으로 하는 경우 [작업 라이브러리 및 독립형 TensorFlow Lite를 사용하는 GPU](#standalone) 섹션을 참조하세요.

### 프로젝트 종속성 추가하기

Google Play 서비스를 사용하여 TensorFlow Lite 작업 라이브러리로 GPU 대리자에 대한 액세스를 활성화하려면 앱의 `build.gradle` 파일 종속성에 `com.google.android.gms:play-services-tflite-gpu`를 추가하세요.

```
dependencies {
  ...
  implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0'
}
```

### GPU 가속 사용하기

그런 다음 [`TfLiteGpu`](https://developers.google.com/android/reference/com/google/android/gms/tflite/gpu/support/TfLiteGpu) 클래스를 사용하여 장치에 GPU 대리자를 사용할 수 있는지 비동기식으로 확인하고 [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder) 클래스로 작업 API 모델 클래스용 GPU 대리자 옵션을 활성화합니다. 예를 들어, 다음 코드 예시와 같이 `ObjectDetector`에서 GPU를 설정할 수 있습니다.

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">        val useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context)

        lateinit val optionsTask = useGpuTask.continueWith { task -&gt;
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
      <p></p>
<pre class="prettyprint lang-java">      Task&lt;Boolean&gt; useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context);

      Task&lt;ObjectDetectorOptions&gt; optionsTask = useGpuTask.continueWith({ task -&gt;
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

## 독립형 TensorFlow Lite로 GPU 사용하기{:#standalone}

애플리케이션이 Google Play를 실행하지 않는 기기를 대상으로 하는 경우, 애플리케이션에 GPU 대리자를 번들로 제공하여 독립형 버전의 TensorFlow Lite와 함께 사용할 수 있습니다.

### 프로젝트 종속성 추가하기

독립형 버전의 TensorFlow Lite를 사용하여 TensorFlow Lite 작업 라이브러리로 GPU 대리자에 대한 액세스를 활성화하려면 앱의 `build.gradle` 파일 종속성에 `org.tensorflow:tensorflow-lite-gpu-delegate-plugin`을 추가하세요.

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite'
  implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

### GPU 가속 사용하기

그런 다음 [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder) 클래스를 사용하여 작업 API 모델 클래스용 GPU 대리자 옵션을 활성화합니다. 예를 들어, 다음 코드 예시와 같이 `ObjectDetector`에서 GPU를 설정할 수 있습니다:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">    import org.tensorflow.lite.task.core.BaseOptions
    import org.tensorflow.lite.task.gms.vision.detector.ObjectDetector

    val baseOptions = BaseOptions.builder().useGpu().build()

    val options =
        ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setMaxResults(1)
            .build()

    val objectDetector = ObjectDetector.createFromFileAndOptions(
      context, model, options)
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    import org.tensorflow.lite.task.core.BaseOptions
    import org.tensorflow.lite.task.gms.vision.detector.ObjectDetector

    BaseOptions baseOptions = BaseOptions.builder().useGpu().build();

    ObjectDetectorOptions options =
        ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setMaxResults(1)
            .build();

    val objectDetector = ObjectDetector.createFromFileAndOptions(
      context, model, options);
      </pre>
    </section>
  </devsite-selector>
</div>
