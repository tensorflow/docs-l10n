# 이미지 분할기 통합하기

이미지 분할기는 이미지의 각 픽셀이 특정 클래스와 연관되어 있는지 여부를 예측합니다. 이는 직사각형 영역에서 객체를 감지하는 <a href="../../models/object_detection/overview.md">객체 감지</a>, 그리고 전체 이미지를 분류하는 <a href="../../models/image_classification/overview.md">이미지 분류</a>와 대조적입니다. 이미지 분할기에 대한 자세한 내용은 [이미지 분할 소개](../../models/segmentation/overview.md)를 참조하세요.

Task Library `ImageSegmenter` API를 사용하여 사용자 정의 이미지 분할기 또는 사전 훈련된 분할기를 모델 앱에 배포합니다.

## ImageSegmenter API의 주요 특징

- 회전, 크기 조정 및 색 공간 변환을 포함한 입력 이미지 처리

- 레이블 맵 로케일

- 범주 마스크와 신뢰 마스크의 두 가지 출력 유형

- 표시 목적의 컬러 레이블

## Supported image segmenter models

The following models are guaranteed to be compatible with the `ImageSegmenter` API.

- The [pretrained image segmentation models on TensorFlow Hub](https://tfhub.dev/tensorflow/collections/lite/task-library/image-segmenter/1).

- [모델 호환성 요구 사항](#model-compatibility-requirements)을 충족하는 사용자 정의 모델

## Java에서 추론 실행하기

Android 앱에서 `ImageSegmenter`를 사용하는 방법의 예는 [Image Segmentation 참조 앱](https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android/)을 참조하세요.

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

Note: starting from version 4.1 of the Android Gradle plugin, .tflite will be added to the noCompress list by default and the aaptOptions above is not needed anymore.

### 2단계: 모델 사용하기

```java
// Initialization
ImageSegmenterOptions options =
    ImageSegmenterOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setOutputType(OutputType.CONFIDENCE_MASK)
        .build();
ImageSegmenter imageSegmenter =
    ImageSegmenter.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Segmentation> results = imageSegmenter.segment(image);
```

`ImageSegmenter` 구성에 대한 추가 옵션은 [소스 코드 및 javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/segmenter/ImageSegmenter.java)를 참조하세요.

## C++에서 추론 실행하기

```c++
// Initialization
ImageSegmenterOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_file);
std::unique_ptr<ImageSegmenter> image_segmenter = ImageSegmenter::CreateFromOptions(options).value();

// Run inference
const SegmentationResult result = image_segmenter->Segment(*frame_buffer).value();
```

`ImageSegmenter` 구성에 대한 추가 옵션은 [소스 코드](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/image_segmenter.h)를 참조하세요.

## Example results

다음은 TensorFlow Hub에서 사용할 수 있는 일반적인 분할 모델인 [deeplab_v3](https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/1)의 분할 결과를 보여주는 예입니다.


<img src="images/plane.jpg" alt="plane" width="50%">

```
Color Legend:
 (r: 000, g: 000, b: 000):
  index       : 0
  class name  : background
 (r: 128, g: 000, b: 000):
  index       : 1
  class name  : aeroplane

# (omitting multiple lines for conciseness) ...

 (r: 128, g: 192, b: 000):
  index       : 19
  class name  : train
 (r: 000, g: 064, b: 128):
  index       : 20
  class name  : tv
Tip: use a color picker on the output PNG file to inspect the output mask with
this legend.
```

분할 범주 마스크는 다음과 같아야 합니다.


<img src="images/segmentation-output.png" alt="segmentation-output" width="30%">

자체 모델 및 테스트 데이터로 간단한 [ImageSegmenter용 CLI 데모 도구](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#image-segmenter)를 시도해 보세요.

## Model compatibility requirements

`ImageSegmenter` API는 필수 [TFLite 모델 메타데이터](../../convert/metadata.md)가 있는 TFLite 모델을 예상합니다. [TensorFlow Lite Metadata Writer API](../../convert/metadata_writer_tutorial.ipynb#image_segmenters)를 사용하여 이미지 분류자에 대한 메타데이터를 생성하는 예를 참조하세요.

- 입력 이미지 텐서(kTfLiteUInt8/kTfLiteFloat32)

    - 이미지 입력 크기는 `[batch x height x width x channels]`입니다.
    - 배치 추론은 지원되지 않습니다(`batch`는 1이어야 함).
    - RGB 입력만 지원됩니다(`channels`은 3이어야 함).
    - 유형이 kTfLiteFloat32인 경우, 입력 정규화를 위해 NormalizationOptions를 메타데이터에 첨부해야 합니다.

- 출력 마스크 텐서: (kTfLiteUInt8/kTfLiteFloat32)

    - tensor of size `[batch x mask_height x mask_width x num_classes]`, where `batch` is required to be 1, `mask_width` and `mask_height` are the dimensions of the segmentation masks produced by the model, and `num_classes` is the number of classes supported by the model.
    - 선택적 (권장함) 레이블 맵은 한 줄에 하나의 레이블을 포함하여 TENSOR_VALUE_LABELS 유형의 AssociatedFile-s로 첨부할 수 있습니다. 첫 번째 AssociatedFile(있는 경우)은 결과의 `label` 필드(C++에서 `class_name`으로 명명됨)를 채우는 데 사용됩니다. `display_name` 필드는 생성 시 사용된 `ImageSegmenterOptions`의 `display_names_locale` 필드와 로케일이 일치하는 AssociatedFile(있는 경우)로부터 채워집니다(기본적으로 "en", 즉 영어). 이들 중 어느 것도 사용할 수 없는 경우, 결과의 `index` 필드만 채워집니다.
