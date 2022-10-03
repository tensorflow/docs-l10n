# Android를 사용한 객체 감지

이 튜토리얼에서는 TensorFlow Lite로 Android 앱을 빌드하여 장치 카메라로 캡처한 프레임에서 객체를 연속적으로 감지하는 방법을 보여줍니다. 이 애플리케이션은 실제 Android 장치용으로 설계되었습니다. 기존 프로젝트를 업데이트하는 경우 코드 샘플을 참조로 사용하고 [프로젝트 수정](#add_dependencies) 지침으로 건너뛸 수 있습니다.

![객체 감지 애니메이션 데모](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/obj_detection_cat.gif){: .attempt-right width="250px"}

## 객체 감지 개요

*객체 감지*는 이미지 내 여러 객체 클래스의 존재와 위치를 식별하는 머신 러닝 작업입니다. 객체 감지 모델은 알려진 객체 세트가 포함된 데이터세트에서 학습됩니다.

학습된 모델은 입력으로 이미지 프레임을 수신하고 인식하도록 학습된 알려진 클래스 집합에서 이미지의 항목을 분류하려고 시도합니다. 각 이미지 프레임에 대해 객체 감지 모델은 감지하는 객체의 목록, 각 객체에 대한 경계 상자의 위치 및 올바르게 분류되는 객체의 신뢰도를 나타내는 점수를 출력합니다.

## 모델 및 데이터세트

이 튜토리얼에서는 [COCO 데이터세트](http://cocodataset.org/)를 사용하여 학습된 모델을 사용합니다. COCO는 330K 이미지, 150만 객체 인스턴스 및 80개 객체 범주를 포함하는 대규모 객체 감지 데이터세트입니다.

다음 사전 학습된 모델 중 하나를 사용할 수 있습니다.

- [EfficientDet-Lite0](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1) *[권장]* - BiFPN 특징 추출기, 공유 상자 예측기 및 초점 손실이 있는 경량의 객체 감지 모델. COCO 2017 검증 데이터세트의 mAP(평균 평균 정밀도)는 25.69%입니다.

- [EfficientDet-Lite1](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite1/detection/metadata/1) - 중간 크기의 EfficientDet 객체 감지 모델. COCO 2017 검증 데이터세트의 mAP는 30.55%입니다.

- [EfficientDet-Lite2](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/metadata/1) - 더 큰 EfficientDet 객체 감지 모델. COCO 2017 검증 데이터세트의 mAP는 33.97%입니다.

- [MobileNetV1-SSD](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2) - 객체 감지를 위해 TensorFlow Lite와 함께 작동하도록 최적화된 초경량 모델. COCO 2017 검증 데이터세트의 mAP는 21%입니다.

이 튜토리얼에서 *EfficientDet-Lite0* 모델은 크기와 정확도 사이에서 적절한 균형을 유지합니다.

모델을 다운로드, 추출 및 자산 폴더에 배치하는 작업은 빌드 시 실행되는 `download.gradle` 파일에 의해 자동으로 관리됩니다. TFLite 모델을 프로젝트에 수동으로 다운로드할 필요가 없습니다.

## 예제 설정 및 실행

객체 감지 앱을 설정하려면 [GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android)에서 샘플을 다운로드하고 [Android Studio](https://developer.android.com/studio/)를 사용하여 실행합니다. 이 튜토리얼의 다음 섹션에서는 코드 예제의 관련 섹션을 탐색하므로 이를 자신의 Android 앱에 적용할 수 있습니다.

### 시스템 요구 사항

- **[Android Studio](https://developer.android.com/studio/index.html)** 버전 2021.1.1(Bumblebee) 이상
- Android SDK 버전 31 이상
- 개발자 모드가 활성화된 상태로 최소 OS 버전의 SDK 24(Android 7.0 - Nougat)가 설치된 Android 기기

참고: 이 예제에서는 카메라를 사용하므로 실제 Android 기기에서 실행하세요.

### 예제 코드 가져오기

예제 코드의 로컬 복사본을 만듭니다. 이 코드를 사용하여 Android Studio에서 프로젝트를 만들고 샘플 애플리케이션을 실행합니다.

예제 코드를 복제하고 설정하려면:

1. git 리포지토리를 복제합니다.
    <pre class="devsite-click-to-copy">    git clone https://github.com/tensorflow/examples.git
        </pre>
2. 선택적으로, 스파스 체크아웃을 사용하도록 git 인스턴스를 구성합니다. 그러면 객체 감지 예제 앱에 대한 파일만 남게 됩니다.
    <pre class="devsite-click-to-copy">    cd examples
        git sparse-checkout init --cone
        git sparse-checkout set lite/examples/object_detection/android
        </pre>

### 프로젝트 가져오기 및 실행

다운로드한 예제 코드에서 프로젝트를 생성하고 프로젝트를 빌드한 후 실행합니다.

예제 코드 프로젝트를 가져오고 빌드하려면:

1. [Android Studio](https://developer.android.com/studio)를 시작합니다.
2. Android 스튜디오에서 **File(파일) &gt; New(새로 만들기) &gt; Import Project(프로젝트 가져오기)**를 선택합니다.
3. build.gradle 파일(`.../examples/lite/examples/object_detection/android/build.gradle`)이 포함된 예제 코드 디렉터리로 이동하여 해당 디렉터리를 선택합니다.
4. Android Studio에서 Gradle 동기화를 요청하면 OK(확인)를 선택합니다.
5. Android 기기가 컴퓨터에 연결되어 있고 개발자 모드가 활성화되어 있는지 확인합니다. 녹색 `Run` 화살표를 클릭합니다.

올바른 디렉터리를 선택하면 Android Studio에서 새 프로젝트를 만들고 빌드합니다. 이 프로세스는 컴퓨터 속도와 다른 프로젝트에 Android Studio를 사용했는지 여부에 따라 몇 분이 소요될 수 있습니다. 빌드가 완료되면 Android Studio가 <strong>빌드 출력</strong> 상태 패널에 <code>BUILD SUCCESSFUL</code> 메시지를 표시합니다.

참고: 예제 코드는 Android Studio 4.2.2로 빌드되었지만 이전 버전의 Studio에서도 작동합니다. 이전 버전의 Android Studio를 사용하는 경우 Studio를 업그레이드하는 대신 빌드가 완료되도록 Android 플러그인의 버전 번호를 조정할 수 있습니다.

**선택 사항:** Android 플러그인 버전을 업데이트하여 빌드 오류를 수정하려면:

1. 프로젝트 디렉터리에서 build.gradle 파일을 엽니다.

2. Android 도구 버전을 다음과 같이 변경합니다.

    ```
    // from: classpath
    'com.android.tools.build:gradle:4.2.2'
    // to: classpath
    'com.android.tools.build:gradle:4.1.2'
    ```

3. **File(파일) &gt; Sync Project with Gradle Files(프로젝트를 Gradle 파일과 동기화)**를 선택하여 프로젝트를 동기화합니다.

프로젝트를 실행하려면:

1. Android Studio에서 **Run(실행) &gt; Run(실행)…**을 선택하여 프로젝트를 실행합니다.
2. 카메라가 있는 연결 Android 기기를 선택하여 앱을 테스트합니다.

다음 섹션에서는 이 예제 앱을 참조점으로 사용하여 이 기능을 자신의 앱에 추가하기 위해 기존 프로젝트에 수행해야 하는 변경 내용을 보여줍니다.

## 프로젝트 종속성 추가 {:#add_dependencies}

자신의 애플리케이션에서 TensorFlow Lite 머신 러닝 모델을 실행하고 이미지와 같은 데이터를 사용 중인 모델에서 처리할 수 있는 텐서 데이터 형식으로 변환하는 유틸리티 기능에 액세스하려면 특정 프로젝트 종속성을 추가해야 합니다.

예제 앱은 TensorFlow Lite의 [비전용 Task 라이브러리](../../inference_with_metadata/task_library/overview#supported_tasks)를 사용하여 객체 감지 머신 러닝 학습 모델을 실행할 수 있도록 합니다. 다음 지침은 필요한 라이브러리 종속성을 자체 Android 앱 프로젝트에 추가하는 방법을 설명합니다.

다음 지침은 필요한 프로젝트 및 모듈 종속성을 자체 Android 앱 프로젝트에 추가하는 방법을 설명합니다.

모듈 종속성을 추가하려면:

1. TensorFlow Lite를 사용하는 모듈에서 다음 종속성을 포함하도록 모듈의 `build.gradle` 파일을 업데이트합니다. 예제 코드에서 이 파일은 `...examples/lite/examples/object_detection/android/app/build.gradle`에 있습니다([코드 참조](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android/app/build.gradle)).

    ```
    dependencies {
      ...
      implementation 'org.tensorflow:tensorflow-lite-task-vision:0.4.0'
      // Import the GPU delegate plugin Library for GPU inference
      implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.0'
      implementation 'org.tensorflow:tensorflow-lite-gpu:2.9.0'
    }
    ```

    프로젝트에는 Vision 작업 라이브러리(`tensorflow-lite-task-vision`)가 포함되어야 합니다. 그래픽 처리 장치(GPU) 라이브러리(`tensorflow-lite-gpu-delegate-plugin`)는 GPU에서 앱을 실행하기 위한 인프라를 제공하고 대리자(`tensorflow-lite-gpu`)는 호환성 목록을 제공합니다.

2. Android Studio에서 **File(파일) &gt; Sync Project with Gradle Files(프로젝트를 Gradle 파일과 동기화)**를 선택하여 프로젝트 종속성을 동기화합니다.

## ML 모델 초기화

Android 앱에서 모델로 예측을 실행하기 전에 매개변수로 TensorFlow Lite 머신 러닝 모델을 초기화해야 합니다. 이러한 초기화 매개변수는 객체 감지 모델 전체에서 일관되며 예측을 위한 최소 정확도 임계값과 같은 설정을 포함할 수 있습니다.

TensorFlow Lite 모델에는 모델 코드가 포함된 `.tflite` 파일이 포함되어 있으며 모델에서 예측한 클래스 이름이 포함된 레이블 파일이 자주 포함됩니다. 객체 감지의 경우 클래스는 사람, 개, 고양이 또는 자동차와 같은 객체입니다.

이 예제는 `download_models.gradle`에 지정된 여러 모델을 다운로드하고 `ObjectDetectorHelper` 클래스는 모델에 대한 선택기를 제공합니다.

```
val modelName =
  when (currentModel) {
    MODEL_MOBILENETV1 -> "mobilenetv1.tflite"
    MODEL_EFFICIENTDETV0 -> "efficientdet-lite0.tflite"
    MODEL_EFFICIENTDETV1 -> "efficientdet-lite1.tflite"
    MODEL_EFFICIENTDETV2 -> "efficientdet-lite2.tflite"
    else -> "mobilenetv1.tflite"
  }
```

요점: 모델은 개발 프로젝트의 `src/main/assets` 디렉터리에 저장해야 합니다. TensorFlow Lite Task 라이브러리는 모델 파일 이름을 지정할 때 이 디렉터리를 자동으로 확인합니다.

앱에서 모델을 초기화하려면:

1. `.tflite` 모델 파일을 개발 프로젝트의 `src/main/assets` 디렉터리에 추가합니다(예: [EfficientDet-Lite0 )](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1).

2. 모델의 파일 이름에 대한 정적 변수를 설정합니다. 예제 앱에서는 EfficientDet-Lite0 탐지 모델을 사용하기 위해 `modelName` 변수를 `MODEL_EFFICIENTDETV0`으로 설정합니다.

3. 예측 임계값, 결과 집합 크기 및 선택적으로 하드웨어 가속 대리자와 같은 모델에 대한 옵션을 설정합니다.

    ```
    val optionsBuilder =
      ObjectDetector.ObjectDetectorOptions.builder()
        .setScoreThreshold(threshold)
        .setMaxResults(maxResults)
    ```

4. 이 객체의 설정을 사용하여 모델이 포함된 TensorFlow Lite [`ObjectDetector`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/vision/detector/ObjectDetector#createFromFile(Context,%20java.lang.String)) 객체를 구성합니다.

    ```
    objectDetector =
      ObjectDetector.createFromFileAndOptions(
        context, modelName, optionsBuilder.build())
    ```

`setupObjectDetector`는 다음 모델 매개변수를 설정합니다.

- 감지 임계값
- 최대 감지 결과 수
- 사용할 처리 스레드 수(`BaseOptions.builder().setNumThreads(numThreads)`)
- 실제 모델(`modelName`)
- ObjectDetector 객체(`objectDetector`)

### 하드웨어 가속기 구성하기

애플리케이션에서 TensorFlow Lite 모델을 초기화할 때 하드웨어 가속 기능을 사용하여 모델의 예측 계산 속도를 높일 수 있습니다.

TensorFlow Lite *대리자*는 그래픽 처리 장치(GPU), 텐서 처리 장치(TPU) 및 디지털 신호 프로세서(DSP)와 같은 모바일 장치의 특수 처리 하드웨어를 사용하여 머신 러닝 모델의 실행을 가속화하는 소프트웨어 모듈입니다. TensorFlow Lite 모델을 실행하기 위해 대리자를 사용하는 것이 권장되지만 필수는 아닙니다.

객체 감지기는 이를 사용 중인 스레드의 현재 설정을 사용하여 초기화됩니다. 메인 스레드에서 생성되고 백그라운드 스레드에서 사용되는 감지기와 함께 CPU 및 [NNAPI](../../android/delegates/nnapi) 대리자를 사용할 수 있지만 감지기를 초기화한 스레드는 GPU 대리자를 사용해야 합니다.

대리자는 `ObjectDetectionHelper.setupObjectDetector()` 함수 내에서 설정됩니다.

```
when (currentDelegate) {
    DELEGATE_CPU -> {
        // Default
    }
    DELEGATE_GPU -> {
        if (CompatibilityList().isDelegateSupportedOnThisDevice) {
            baseOptionsBuilder.useGpu()
        } else {
            objectDetectorListener?.onError("GPU is not supported on this device")
        }
    }
    DELEGATE_NNAPI -> {
        baseOptionsBuilder.useNnapi()
    }
}
```

TensorFlow Lite와 함께 하드웨어 가속 대리자를 사용하는 방법에 대한 자세한 내용은 [TensorFlow Lite 대리자](../../performance/delegates)를 참조하세요.

## 모델에 대한 데이터 준비하기

Android 앱에서 코드는 이미지 프레임과 같은 기존 데이터를 모델에서 처리할 수 있는 텐서 데이터 형식으로 변환하여 해석을 수행할 수 있게 모델에 데이터를 제공합니다. 모델로 전달되는 텐서의 데이터에는 모델 훈련에 사용되는 데이터 형식과 일치하는 특정 차원 또는 형상이 있어야 합니다.

이 코드 예제에 사용된 [EfficientDet-Lite0](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1) 모델은 픽셀당 3개의 채널(빨간색, 파란색 및 녹색)이 있는 320 x 320 크기의 이미지를 나타내는 텐서를 받아들입니다. 텐서의 각 값은 0에서 255 사이의 단일 바이트입니다. 따라서 새 이미지에 대한 예측을 실행하려면 앱에서 해당 이미지 데이터를 해당 크기와 모양의 텐서 데이터 객체로 변환해야 합니다. TensorFlow Lite Task Library Vision API는 데이터 변환을 자동으로 처리합니다.

앱은 [`ImageAnalysis`](https://developer.android.com/training/camerax/analyze) 객체를 사용하여 카메라에서 이미지를 가져옵니다. 이 객체는 카메라의 비트맵과 함께 `detectObject` 함수를 호출합니다. 데이터는 `ImageProcessor`에 의해 자동으로 크기가 조정되고 회전되어 모델의 이미지 데이터 요구 사항을 충족합니다. 그런 다음 이미지는 [`TensorImage`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/image/TensorImage) 객체로 변환됩니다.

ML 모델에서 처리할 카메라 하위 시스템의 데이터를 준비하려면:

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
      it.setAnalyzer(cameraExecutor) {
        image -> if (!::bitmapBuffer.isInitialized)
        { bitmapBuffer = Bitmap.createBitmap( image.width, image.height,
        Bitmap.Config.ARGB_8888 ) } detectObjects(image)
        }
      }
    ```

3. 모델에 필요한 특정 이미지 데이터를 추출하고 이미지 회전 정보를 전달합니다.

    ```
    private fun detectObjects(image: ImageProxy) {
      //Copy out RGB bits to the shared bitmap buffer
      image.use {bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
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

참고: Android 카메라 하위 시스템에서 이미지 정보를 추출할 때 이미지를 RGB 형식으로 가져와야 합니다. 이 형식은 모델에서 분석할 수 있게 이미지를 준비하는 데 사용되는 TensorFlow Lite <a>ImageProcessor</a> 클래스에 필요합니다. RGB 형식 이미지에 알파 채널이 포함된 경우, 해당 투명도 데이터는 무시됩니다.

## 예측 실행하기

Android 앱에서 올바른 형식의 이미지 데이터로 TensorImage 객체를 생성했으면 해당 데이터에 대해 모델을 실행하여 예측 또는 *추론*을 생성할 수 있습니다.

예제 앱의 `fragments/CameraFragment.kt` 클래스에서 `bindCameraUseCases` 함수 내의 `imageAnalyzer` 객체는 앱이 카메라에 연결될 때 예측을 위해 모델에 자동으로 데이터를 전달합니다.

앱은 `cameraProvider.bindToLifecycle()` 메서드를 사용하여 카메라 선택기, 미리보기 창 및 ML 모델 처리를 처리합니다. `ObjectDetectorHelper.kt` 클래스는 이미지 데이터를 모델로 전달하는 작업을 처리합니다. 모델을 실행하고 이미지 데이터에서 예측을 생성하려면:

- 이미지 데이터를 예측 함수에 전달하여 예측을 실행합니다.

    ```
    val results = objectDetector?.detect(tensorImage)
    ```

TensorFlow Lite Interpreter 객체는 이 데이터를 수신하여 모델에 대해 실행하고 예측 목록을 생성합니다. 모델에서 데이터를 연속적으로 처리하려면 각 예측 실행에 대해 Interpreter 객체가 생성된 다음 시스템에서 제거되지 않도록 `runForMultipleInputsOutputs()` 메서드를 사용하세요.

## 모델 출력 처리하기

Android 앱에서 객체 감지 모델에 대해 이미지 데이터를 실행하면 추가 비즈니스 로직을 실행하거나 사용자에게 결과를 표시하거나 다른 작업을 수행하여 앱 코드가 처리해야 하는 예측 목록을 생성합니다.

주어진 TensorFlow Lite 모델의 출력은 생성하는 예측 수(하나 또는 여러 개)와 각 예측에 대한 설명 정보에 따라 다릅니다. 객체 감지 모델의 경우 예측에는 일반적으로 이미지에서 객체가 감지된 위치를 나타내는 경계 상자에 대한 데이터가 포함됩니다. 예제 코드에서 결과는 객체 감지 프로세스에서 DetectorListener 역할을 하는 `CameraFragment.kt`의 `onResults` 함수로 전달됩니다.

```
interface DetectorListener {
  fun onError(error: String)
  fun onResults(
    results: MutableList<Detection>?,
    inferenceTime: Long,
    imageHeight: Int,
    imageWidth: Int
  )
}
```

이 예에서 사용된 모델의 경우, 각 예측에는 객체의 경계 상자 위치, 객체에 대한 레이블, 그리고 0과 1 사이의 예측 점수가 예측의 신뢰도를 나타내는 부동 소수점으로 포함됩니다(1이 가장 높은 신뢰도 등급). 일반적으로 점수가 50%(0.5) 미만인 예측은 결정적이지 않은 것으로 간주됩니다. 그러나 낮은 값의 예측 결과를 처리하는 방법은 사용자와 사용 목적의 요구 사항에 달려 있습니다.

모델 예측 결과를 처리하려면:

1. 리스너 패턴을 사용하여 앱 코드 또는 사용자 인터페이스 객체에 결과를 전달합니다. 예제 앱은 이 패턴을 사용하여 `ObjectDetectorHelper` 객체의 감지 결과를 `CameraFragment` 객체로 전달합니다.

    ```
    objectDetectorListener.onResults(
    // instance of CameraFragment
        results,
        inferenceTime,
        tensorImage.height,
        tensorImage.width)
    ```

2. 사용자에게 예측을 표시하는 것과 같이 결과에 대해 작업을 수행합니다. 이 예에서는 CameraPreview 객체에 오버레이를 그려 결과를 표시합니다.

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

모델이 예측 결과를 반환하면 애플리케이션은 결과를 사용자에게 제시하거나 추가 논리를 실행하여 해당 예측에 따른 작업을 수행할 수 있습니다. 예제 코드의 경우 애플리케이션은 식별된 객체 주위에 경계 상자를 그리고 화면에 클래스 이름을 표시합니다.

## 다음 단계

- [예제](../../examples)에서 TensorFlow Lite의 다양한 용도를 살펴봅니다.
- [모델](../../models) 섹션에서 TensorFlow Lite와 함께 머신 러닝 모델을 사용하는 방법에 대해 자세히 알아봅니다.
- [TensorFlow Lite 개발자 가이드](../../guide)에서 모바일 애플리케이션에서 머신 러닝을 구현하는 방법에 대해 자세히 알아봅니다.
