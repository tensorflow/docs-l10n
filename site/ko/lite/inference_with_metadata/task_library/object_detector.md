# 객체 감지기 통합하기

객체 감지기는 알려진 객체 세트 중 어떤 것이 존재할 수 있는지 식별하고 주어진 이미지 또는 비디오 스트림 내에서 이러한 객체의 위치에 대한 정보를 제공할 수 있습니다. 객체 감지기는 여러 객체 클래스의 유무와 위치를 감지하도록 훈련됩니다. 예를 들어, 다양한 과일 조각이 포함된 이미지와 함께 과일 이미지가 나타내는 과일의 클래스(예: 사과, 바나나 또는 딸기)를 지정하는 *레이블* 및 각 객체가 이미지에 나타나는 위치를 지정하는 데이터로 모델을 훈련할 수 있습니다. 객체 감지기에 대한 자세한 내용은 [객체 감지 소개](../../models/object_detection/overview.md)를 참조하세요.

작업 라이브러리 `ObjectDetector` API를 사용하여 사용자 정의 객체 감지기 또는 사전 훈련된 감지기를 모델 앱에 배포합니다.

## ObjectDetector API의 주요 특징

- 회전, 크기 조정 및 색 공간 변환을 포함한 입력 이미지 처리

- 레이블 맵 로케일

- 결과를 필터링하기 위한 스코어 임계값

- Top-k 감지 결과

- 레이블 허용 목록 및 거부 목록

## 지원되는 객체 감지기 모델

다음 모델은 `ObjectDetector` API와의 호환성이 보장됩니다.

- [TensorFlow Hub의 사전 훈련된 객체 감지 모델](https://tfhub.dev/tensorflow/collections/lite/task-library/object-detector/1)

- [AutoML Vision Edge 객체 감지](https://cloud.google.com/vision/automl/object-detection/docs)로 만든 모델

- [객체 감지기를 위한 TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker)로 만들어진 모델

- [모델 호환성 요구 사항](#model-compatibility-requirements)을 충족하는 사용자 정의 모델

## Java에서 추론 실행하기

Android 앱에서 `ObjectDetector`를 사용하는 방법의 예를 보려면 [객체 감지 참조 앱](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android/)을 참조하세요.

### 1단계: Gradle 종속성 및 기타 설정 가져오기

`.tflite` 모델 파일을 모델이 실행될 Android 모듈의 assets 디렉토리에 복사합니다. 파일을 압축하지 않도록 지정하고 TensorFlow Lite 라이브러리를 모듈의 `build.gradle` 파일에 추가합니다.

```java
android {
    // Other settings

    // Specify tflite file should not be compressed for the app apk
    aaptOptions {
        noCompress "tflite"
    }

}

dependencies {
    // Other dependencies

    // Import the Task Vision Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-vision:0.3.0'
    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.3.0'
}
```

참고: Android Gradle 플러그인 버전 4.1부터는 .tflite가 기본적으로 noCompress 목록에 추가되며 위의 aaptOptions는 더 이상 필요하지 않습니다.

### 2단계: 모델 사용하기

```java
// Initialization
ObjectDetectorOptions options =
    ObjectDetectorOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setMaxResults(1)
        .build();
ObjectDetector objectDetector =
    ObjectDetector.createFromFileAndOptions(
        context, modelFile, options);

// Run inference
List<Detection> results = objectDetector.detect(image);
```

`ObjectDetector`를 구성하는 추가 옵션은 [소스 코드 및 javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/detector/ObjectDetector.java)를 참조하세요.

## C++에서 추론 실행하기

```c++
// Initialization
ObjectDetectorOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_file);
std::unique_ptr<ObjectDetector> object_detector = ObjectDetector::CreateFromOptions(options).value();

// Run inference
const DetectionResult result = object_detector->Detect(*frame_buffer).value();
```

`ObjectDetector`를 구성하기 위한 추가 옵션은 [소스 코드](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/object_detector.h)를 참조하세요.

## 예제 결과

다음은 TensorFlow Hub에서 [ssd mobilenet v1](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1)의 감지 결과를 보여주는 예입니다.

<img src="images/dogs.jpg" alt="개" width="50%">

```
Results:
 Detection #0 (red):
  Box: (x: 355, y: 133, w: 190, h: 206)
  Top-1 class:
   index       : 17
   score       : 0.73828
   class name  : dog
 Detection #1 (green):
  Box: (x: 103, y: 15, w: 138, h: 369)
  Top-1 class:
   index       : 17
   score       : 0.73047
   class name  : dog
```

경계 상자를 입력 이미지에 렌더링합니다.

<img src="images/detection-output.png" alt="감지 출력" width="50%">

자체 모델 및 테스트 데이터로 간단한 [ObjectDetector용 CLI 데모 도구](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#object-detector)를 시도해 보세요.

## 모델 호환성 요구 사항

`ObjectDetector` API는 필수 [TFLite 모델 메타데이터](../../convert/metadata.md)가 있는 TFLite 모델을 예상합니다. [TensorFlow Lite Metadata Writer API](../../convert/metadata_writer_tutorial.ipynb#object_detectors)를 사용하여 객체 감지기에 대한 메타데이터를 생성하는 예를 참조하세요.

호환되는 객체 감지기 모델은 다음 요구 사항을 충족해야 합니다.

- 입력 이미지 텐서: (kTfLiteUInt8/kTfLiteFloat32)

    - 이미지 입력 크기는 `[batch x height x width x channels]`입니다.
    - 배치 추론은 지원되지 않습니다(`batch`는 1이어야 함).
    - RGB 입력만 지원됩니다(`channels`은 3이어야 함).
    - 유형이 kTfLiteFloat32인 경우, 입력 정규화를 위해 NormalizationOptions를 메타데이터에 첨부해야 합니다.

- 출력 텐서는 `DetectionPostProcess` op의 4개 출력이어야 합니다. 즉, 다음과 같습니다.

    - 위치 텐서(kTfLiteFloat32)

        - `[1 x num_results x 4]` 크기의 텐서, [top, left, right, bottom] 형식의 경계 상자를 나타내는 내부 배열입니다.
        - BoundingBoxProperties는 메타데이터에 첨부되어야 하며 `type=BOUNDARIES` 및 `coordinate_type = RATIO를 지정해야 합니다.

    - 클래스 텐서(kTfLiteFloat32)

        - `[1 x num_results]` 크기의 텐서, 각 값은 클래스의 정수 인덱스를 나타냅니다.
        - 선택적(그러나 권장함) 레이블 맵은 한 줄에 하나의 레이블을 포함하여 TENSOR_VALUE_LABELS 유형의 AssociatedFile-s로 첨부할 수 있습니다. [예제 레이블 파일](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/python/tests/testdata/object_detector/labelmap.txt)을 참조하세요. 이러한 첫 번째 AssociatedFile(있는 경우)은 결과의 `class_name` 필드를 채우는 데 사용됩니다. `display_name` 필드는 생성 시 사용된 `ObjectDetectorOptions`의 `display_names_locale` 필드와 로케일이 일치하는 AssociatedFile(있는 경우)로부터 채워집니다(기본적으로 "en", 즉 영어). 이들 중 어느 것도 사용할 수 없는 경우, 결과의 `index` 필드만 채워집니다.

    - 스코어 텐서(kTfLiteFloat32)

        - `[1 x num_results]` 크기의 텐서, 각 값은 감지된 객체의 스코어를 나타냅니다.

    - 감지 텐서의 수(kTfLiteFloat32)

        - `[1]` 크기의 텐서인 정수 num_results
