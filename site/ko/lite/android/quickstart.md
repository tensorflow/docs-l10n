# Android 빠른 시작

이 페이지는 TensorFlow Lite를 통해 Android 앱을 구축하여 라이브 카메라 피드를 분석하고 객체를 식별하는 방법을 보여줍니다. 이 머신러닝 사용 사례는 *객체 감지*라고 합니다. 예제 앱은 [Google Play 서비스](./play_services)를 통해 TensorFlow Lite [비전용 작업 라이브러리](../inference_with_metadata/task_library/overview#supported_tasks)를 사용하여 TensorFlow Lite를 통해 ML 애플리케이션을 구축하는 데 권장되는 접근 방법인 객체 감지 머신러닝 모델의 실행을 활성화합니다.

<aside class="note"><b>약관:</b> Google Play 서비스 API에서 TensorFlow Lite에 액세스하거나 사용함으로써, <a href="./play_services#tos">서비스 약관</a>에 동의하게 됩니다. API에 액세스하기 전에 모든 해당 약관 및 정책을 읽고 이해해 주시길 바랍니다.</aside>

![객체 감지 애니메이션 데모](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/obj_detection_cat.gif){: .attempt-right width="250px"}

## 설치 및 예제 실행

이 연습의 첫 번째 부분에서는 GitHub에서 [예시 코드](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android_play_services)를 다운로드하고 [Android Studio](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android_play_services)를 사용하여 실행합니다. 이 문서의 다음 섹션에서는 코드 예제의 관련 섹션을 살펴볼 것이므로 이를 자신의 Android 앱에 적용할 수 있습니다. 이러한 도구의 다음 버전이 설치되어 있어야 합니다.

- Android Studio 4.2 이상
- Android SDK 버전 21 이상

참고: 이 예에서는 카메라를 사용하므로 실제 Android 기기에서 실행해야 합니다.

### 예제 코드 가져오기

예제 코드를 구축하고 실행할 수 있도록 예제 코드의 로컬 사본을 생성합니다.

예제 코드를 복제하고 설정하려면:

1. git 리포지토리를 복제합니다.
    <pre class="devsite-click-to-copy">    git clone https://github.com/tensorflow/examples.git
        </pre>
2. 희소 체크아웃을 사용하도록 git 인스턴스를 구성하면 객체 감지 예제 앱에 대한 파일만 남게 됩니다.
    <pre class="devsite-click-to-copy">    cd examples
        git sparse-checkout init --cone
        git sparse-checkout set lite/examples/object_detection/android_play_services
        </pre>

### 프로젝트 가져오기 및 실행하기

Android Studio를 사용해 다운로드한 예제 코드에서 프로젝트를 생성하고 프로젝트를 구축하고 실행합니다.

예제 코드 프로젝트를 가져오고 빌드하려면:

1. [Android Studio](https://developer.android.com/studio)를 시작합니다.
2. Android Studio의 **시작** 페이지에서 **Import Project(프로젝트 가져오기)**를 선택하거나 **File(파일) &gt; New(새로 만들기) &gt; Import Project(프로젝트 가져오기)**를 선택합니다.
3. build.gradle 파일(`...examples/lite/examples/object_detection/android_play_services/build.gradle`)이 포함된 예제 코드 디렉터리로 이동하여 해당 디렉터리를 선택합니다.

이 디렉터리를 선택한 후, Android Studio는 새 프로젝트를 생성하고 이를 구축합니다. 구축이 완료되면 Android Studio는 **Build Output** 상태 패널에 `BUILD SUCCESSFUL` 메시지를 표시합니다.

프로젝트를 실행하려면:

1. Android Studio에서 **Run(실행) &gt; Run(실행)…** 및 **MainActivity**를 선택하여 프로젝트를 실행합니다.
2. 카메라가 있는 연결 Android 기기를 선택하여 앱을 테스트합니다.

## 예제 앱 작동 방식

예제 앱은 TensorFlow Lite 형식의 [mobilenetv1.tflite](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2?lite-format=tflite)과 같은 사전 훈련된 객체 감지 모델을 사용해 Android 기기의 카메라에서 라이브 비디오 스트림의 객체를 찾습니다. 이 기능을 위한 코드는 주로 다음과 같은 파일에 있습니다.

- [ObjectDetectorHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android_play_services/app/src/main/java/org/tensorflow/lite/examples/objectdetection/ObjectDetectorHelper.kt) - 런타임 환경을 초기화하고 하드웨어 가속을 활성화하며 객체 감지 ML 모델을 실행합니다.
- [CameraFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android_play_services/app/src/main/java/org/tensorflow/lite/examples/objectdetection/fragments/CameraFragment.kt) - 카메라 이미지 데이트 스트림을 구축하고 모델을 위한 데이터를 준비하며 객체 감지 결과를 표시합니다.

참여: 이 예제 앱은 일반 머신러닝 연산을 수행하는 데 사용하기 쉬운, 작업별 API를 제공하는 TensorFlow Lite [작업 라이브러리](../inference_with_metadata/task_library/overview#supported_tasks)를 사용합니다. 더욱 구체적인 요구와 사용자 정의된 ML 함수가 있는 앱의 경우, [Interpreter API](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/InterpreterApi)를 사용하는 것을 고려하세요.

다음 섹션은 Android 앱을 수정하여 이 기능을 추가할 수 있도록 이러한 코드 파일의 주요 구성요소를 보여줍니다.

## 앱 구축 {:#build_app}

다음 섹션은 자신만의 Android 앱을 구축하고 예제 앱에 표시된 모델을 실행하는 주요 단계를 설명합니다. 이러한 지침은 앞서 보여드린 예시 앱을 참조 포인트로 사용합니다.

참고: 이러한 지침을 따르고 자신만의 앱을 구축하려면 Android Studio를 사용하여 [기본 Android 프로젝트](https://developer.android.com/studio/projects/create-project)를 생성하세요.

### 프로젝트 종속성 추가 {:#add_dependencies}

기본 Android 앱에서, TensorFlow Lite 머신러닝 모델을 실행하고 ML 데이터 유틸리티 기능에 액세스하기 위해 프로젝트 종속성을 추가하세요. 이러한 유틸리티 기능은 이미지와 같은 데이터를 모델이 처리할 수 있는 텐서 데이터 형식으로 전환합니다.

예제 앱은 [Google Play 서비스](./play_services)의 TensorFlow Lite [비전용 작업 라이브러리](../inference_with_metadata/task_library/overview#supported_tasks)를 사용하여 객체 감지 머신러닝 모델을 실행할 수 있도록 합니다. 다음 지침은 필요한 라이브러리 종속성을 자체 Android 앱 프로젝트에 추가하는 방법을 설명합니다.

모듈 종속성을 추가하려면:

1. TensorFlow Lite를 사용하는 모듈에서 다음 종속성을 포함하도록 모듈의 `build.gradle` 파일을 업데이트합니다. 예제 코드에서 이 파일은 `...examples/lite/examples/object_detection/android_play_services/app/build.gradle`에 있습니다.
    ```
    ...
    dependencies {
    ...
        // Tensorflow Lite dependencies
        implementation 'org.tensorflow:tensorflow-lite-task-vision-play-services:0.4.2'
        implementation 'com.google.android.gms:play-services-tflite-gpu:16.1.0'
    ...
    }
    ```
2. Android Studio에서 **File(파일) &gt; Sync Project with Gradle Files(프로젝트를 Gradle 파일과 동기화)**를 선택하여 프로젝트 종속성을 동기화합니다.

### Google Play 서비스 초기화

[Google Play 서비스](./play_services)를 사용하여 TensorFlow Lite 모델을 실행하는 경우, 서비스를 초기화해야 이를 사용할 수 있습니다. 서비스를 통해 GPU 가속과 같은 하드웨어 가속 지원을 사용하고자 하는 경우, 이 초기화의 일환으로 해당 지원도 활성화해야 합니다.

Google Play 서비스로 TensorFlow Lite를 초기화하려면 다음을 수행합니다.

1. `TfLiteInitializationOptions` 객체를 생성하고 수정하여 GPU 지원을 활성화합니다.

    ```
    val options = TfLiteInitializationOptions.builder()
        .setEnableGpuDelegateSupport(true)
        .build()
    ```

2. `TfLiteVision.initialize()` 메서드를 사용하여 Play 서비스 런타임 사용을 활성화하고 리스너를 설정하여 성공적으로 로드되었는지 확인합니다.

    ```
    TfLiteVision.initialize(context, options).addOnSuccessListener {
        objectDetectorListener.onInitialized()
    }.addOnFailureListener {
        // Called if the GPU Delegate is not supported on the device
        TfLiteVision.initialize(context).addOnSuccessListener {
            objectDetectorListener.onInitialized()
        }.addOnFailureListener{
            objectDetectorListener.onError("TfLiteVision failed to initialize: "
                    + it.message)
        }
    }
    ```

### ML 모델 인터프리터 초기화하기

모델 파일을 로드하고 모델 매개변수를 설정하여 TensorFlow Lite 머신러닝 모델 인터프리터를 초기화합니다. TensorFlow Lite 모델은 모델 코드가 있는 `.tflite` 파일을 포함합니다. 예를 들어 다음과 같이 개발 프로젝트의 `src/main/assets` 디렉터리에 모델을 저장해야 합니다.

```
.../src/main/assets/mobilenetv1.tflite`
```

팁: 작업 라이브러리 인터프리터 코드는 파일 경로를 지정하지 않으면 `src/main/assets` 디렉터리에서 모델을 자동으로 찾습니다.

모델을 초기화하려면 다음을 수행합니다.

1. `.tflite` 모델 파일을 개발 프로젝트의 `src/main/assets` 디렉터리에 추가합니다(예: [ssd_mobilenet_v1](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2)).
2. `modelName` 변수를 설정하여 ML 모델의 파일 이름을 지정합니다.
    ```
    val modelName = "mobilenetv1.tflite"
    ```
3. 예측 임계값 및 결과 세트 크기와 같은 모델에 대한 옵션을 설정합니다.
    ```
    val optionsBuilder =
        ObjectDetector.ObjectDetectorOptions.builder()
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults)
    ```
4. 옵션이 있는 GPU 가속을 활성화하고 가속이 기기에서 지원되지 않는 경우 코드가 정상적으로 실패하도록 허용합니다.
    ```
    try {
        optionsBuilder.useGpu()
    } catch(e: Exception) {
        objectDetectorListener.onError("GPU is not supported on this device")
    }

    ```
5. 이 객체의 설정을 사용하여 모델이 포함된 TensorFlow Lite [`ObjectDetector`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/vision/detector/ObjectDetector#createFromFile(Context,%20java.lang.String)) 객체를 구성합니다.
    ```
    objectDetector =
        ObjectDetector.createFromFileAndOptions(
            context, modelName, optionsBuilder.build())
    ```

TensorFlow Lite와 함께 하드웨어 가속 대리자를 사용하는 방법에 대한 자세한 내용은 [TensorFlow Lite 대리자](../performance/delegates)를 참조하세요.

### 모델에 대한 데이터 준비하기

모델이 처리할 수 있도록 이미지와 같은 기존 데이터를 [텐서](../api_docs/java/org/tensorflow/lite/Tensor) 데이터 형식으로 변환하여 모델이 해석할 수 있도록 데이터를 준비합니다. 텐서의 데이터는 모델을 훈련하는 데 사용되는 데이터 형식과 일치하는 특정 치수나 형상이 있어야 합니다. 사용하는 모델에 따라, 모델이 기대하는 것에 맞도록 데이터를 변환해야 할 수 있습니다. 예제 앱은 [`ImageAnalysis`](https://developer.android.com/reference/androidx/camera/core/ImageAnalysis) 객체를 사용하여 카메라 하위 시스템에서 이미지 프레임을 추출합니다.

모델이 처리할 데이터를 준비하려면 다음을 수행합니다.

1. `ImageAnalysis` 객체를 빌드하여 필요한 형식으로 이미지를 추출합니다.
    ```
    imageAnalyzer =
        ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            ...
    ```
2. 분석기를 카메라 하위 시스템에 연결하고 카메라에서 받은 데이터를 포함할 비트맵 버퍼를 만듭니다.
    ```
            .also {
            it.setAnalyzer(cameraExecutor) { image ->
                if (!::bitmapBuffer.isInitialized) {
                    bitmapBuffer = Bitmap.createBitmap(
                        image.width,
                        image.height,
                        Bitmap.Config.ARGB_8888
                    )
                }
                detectObjects(image)
            }
        }
    ```
3. 모델에 필요한 특정 이미지 데이터를 추출하고 이미지 회전 정보를 전달합니다.
    ```
    private fun detectObjects(image: ImageProxy) {
        // Copy out RGB bits to the shared bitmap buffer
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
        val imageRotation = image.imageInfo.rotationDegrees
        objectDetectorHelper.detect(bitmapBuffer, imageRotation)
    }
    ```
4. 예제 앱의 `ObjectDetectorHelper.detect()` 메서드에 표시된 대로 최종 데이터 변환을 완료하고 이미지 데이터를 `TensorImage` 객체에 추가합니다.
    ```
    val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()

    // Preprocess the image and convert it into a TensorImage for detection.
    val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
    ```

### 예측 실행하기

올바른 형식의 이미지 데이터로 [TensorImage](../api_docs/java/org/tensorflow/lite/support/image/TensorImage) 객체를 생성하면 해당 데이터에 대해 모델을 실행하여 예측 또는 *추론*을 생성할 수 있습니다. 예제 앱에서, 이 코드는 `ObjectDetectorHelper.detect()` 메서드에 포함됩니다.

모델을 실행하고 이미지 데이터에서 예측을 생성하려면 다음을 수행합니다.

- 이미지 데이터를 예측 함수에 전달하여 예측을 실행합니다.
    ```
    val results = objectDetector?.detect(tensorImage)
    ```

### 모델 출력 처리하기

객체 감지 모델에 대해 이미지 데이터를 실행하면, 추가 비즈니스 로직을 실행하거나 사용자에게 결과를 표시하거나 다른 조치를 취하여 앱 코드가 다뤄야 하는 예측 결과 목록이 생성됩니다. 예제 앱의 객체 감지 모델은 예측 목록과 감지된 객체에 대한 바운딩 박스를 생성합니다. 예제 앱에서, 예측 결과는 추가 처리를 위해 리스너 객체로 전달되며 사용자에게 표시됩니다.

모델 예측 결과를 처리하려면:

1. 리스너 패턴을 사용하여 앱 코드 또는 사용자 인터페이스 객체에 결과를 전달합니다. 예제 앱은 이 패턴을 사용하여 `ObjectDetectorHelper` 객체의 감지 결과를 `CameraFragment` 객체로 전달합니다.
    ```
    objectDetectorListener.onResults( // instance of CameraFragment
        results,
        inferenceTime,
        tensorImage.height,
        tensorImage.width)
    ```
2. 사용자에게 예측을 표시하는 것과 같이 결과에 대해 작업을 수행합니다. 이 예에서는 `CameraPreview` 객체에 오버레이를 그려 결과를 표시합니다.
    ```
    override fun onResults(
      results: MutableList<Detection>?,
      inferenceTime: Long,
      imageHeight: Int,
      imageWidth: Int
    ) {
        activity?.runOnUiThread {
            fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
                String.format("%d ms", inferenceTime)

            // Pass necessary information to OverlayView for drawing on the canvas
            fragmentCameraBinding.overlay.setResults(
                results ?: LinkedList<Detection>(),
                imageHeight,
                imageWidth
            )

            // Force a redraw
            fragmentCameraBinding.overlay.invalidate()
        }
    }
    ```

## 다음 단계

- [Task Library API](../inference_with_metadata/task_library/overview#supported_tasks)에 대해 더 자세히 알아봅니다.
- [Interpreter API](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/InterpreterApi)에 대해 더 자세히 알아봅니다.
- [예제](../examples)에서 TensorFlow Lite의 용도를 살펴봅니다.
- [모델](../models) 섹션에서 TensorFlow Lite를 통해 머신러닝 모델을 사용하고 구축하는 것에 대해 더 자세히 알아봅니다.
- [TensorFlow Lite 개발자 가이드](../guide)에서 모바일 애플리케이션에서 머신러닝을 구현하는 방법에 대해 자세히 알아봅니다.
