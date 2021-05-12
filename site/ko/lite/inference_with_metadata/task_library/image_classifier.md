# 이미지 분류자 통합

이미지 분류는 이미지가 나타내는 내용을 식별하기 위해 머신 러닝을 활용하는 일반적인 예입니다. 예를 들어 주어진 그림에 어떤 종류의 동물이 나타나는지 알고 싶을 수 있습니다. 이미지가 나타내는 내용을 예측하는 작업을 *이미지 분류*라고 합니다. 이미지 분류자는 다양한 이미지 클래스를 인식하도록 훈련됩니다. 예를 들어 모델은 토끼, 햄스터, 개 등 세 가지 동물 유형을 나타내는 사진을 인식하도록 훈련될 수 있습니다. 이미지 분류 자에 대한 자세한 내용은 [이미지 분류 소개](../../models/image_classification/overview.md)를 참조하세요.

Task Library `ImageClassifier` API를 사용하여 사용자 정의 이미지 분류자 또는 사전 훈련된 분류자를 모델 앱에 배포합니다.

## ImageClassifier API의 주요 기능

- Input image processing, including rotation, resizing, and color space conversion.

- 입력 이미지의 관심 영역

- Label map locale.

- Score threshold to filter results.

- Top-k 분류 결과

- Label allowlist and denylist.

## 지원되는 이미지 분류자 모델

다음 모델은 `ImageClassifier` API와의 호환성이 보장됩니다.

- [이미지 분류용 TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification)에 의해 만들어진 모델

- [TensorFlow Lite Hosted Models의 사전 학습된 이미지 분류 모델](https://www.tensorflow.org/lite/guide/hosted_models#image_classification)

- [TensorFlow 허브의 사전 학습된 이미지 분류 모델](https://tfhub.dev/tensorflow/collections/lite/task-library/image-classifier/1)

- [AutoML Vision Edge 이미지 분류](https://cloud.google.com/vision/automl/docs/edge-quickstart)로 만들어진 모델

- Custom models that meet the [model compatibility requirements](#model-compatibility-requirements).

## Run inference in Java

Android 앱에서 `ImageClassifier`를 사용하는 방법의 예는 [이미지 분류 참조 앱](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/EXPLORE_THE_CODE.md)을 참조하세요.

### Step 1: Import Gradle dependency and other settings

Copy the `.tflite` model file to the assets directory of the Android module where the model will be run. Specify that the file should not be compressed, and add the TensorFlow Lite library to the module’s `build.gradle` file:

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

    // Import the Task Vision Library dependency
    implementation 'org.tensorflow:tensorflow-lite-task-vision:0.1.0'
}
```

### Step 2: Using the model

```java
// Initialization
ImageClassifierOptions options = ImageClassifierOptions.builder().setMaxResults(1).build();
ImageClassifier imageClassifier = ImageClassifier.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Classifications> results = imageClassifier.classify(image);
```

<code>ImageClassifier</code> 구성에 대한 추가 옵션은 <a>소스 코드 및 javadoc</a>를 참조하세요.

## Run inference in C++

Note: we are working on improving the usability of the C++ Task Library, such as providing prebuilt binaries and creating user-friendly workflows to build from source code. The C++ API may be subject to change.

```c++
// Initialization
ImageClassifierOptions options;
options.mutable_model_file_with_metadata()->set_file_name(model_file);
std::unique_ptr<ImageClassifier> image_classifier = ImageClassifier::CreateFromOptions(options).value();

// Run inference
const ClassificationResult result = image_classifier->Classify(*frame_buffer).value();
```

<code>ImageClassifier</code> 구성에 대한 추가 옵션은 [소스 코드](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/image_classifier.h)를 참조하세요.

## Example results

다음은 [새 분류자](https://tfhub.dev/google/lite-model/aiy/vision/classifier/birds_V1/3)의 분류 결과를 보여주는 예입니다.

<img src="images/sparrow.jpg" alt="sparrow" width="50%">

```
Results:
  Rank #0:
   index       : 671
   score       : 0.91406
   class name  : /m/01bwb9
   display name: Passer domesticus
  Rank #1:
   index       : 670
   score       : 0.00391
   class name  : /m/01bwbt
   display name: Passer montanus
  Rank #2:
   index       : 495
   score       : 0.00391
   class name  : /m/0bwm6m
   display name: Passer italiae
```

자체 모델 및 테스트 데이터로 [ImageClassifier를 위한 간단한 CLI 데모 도구](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#image-classifier)를 사용해 보세요.

## Model compatibility requirements

`ImageClassifier` API는 필수 [TFLite 모델 메타데이터](../../convert/metadata.md)가 있는 TFLite 모델을 예상합니다.

호환되는 이미지 분류자 모델은 다음 요구 사항을 충족해야 합니다.

- Input image tensor (kTfLiteUInt8/kTfLiteFloat32)

    - image input of size `[batch x height x width x channels]`.
    - batch inference is not supported (`batch` is required to be 1).
    - only RGB inputs are supported (`channels` is required to be 3).
    - if type is kTfLiteFloat32, NormalizationOptions are required to be attached to the metadata for input normalization.

- 출력 점수 텐서(kTfLiteUInt8/kTfLiteFloat32)

    - `N` 클래스와 2 또는 4 차원, 즉 `[1 x N]` 또는 `[1 x 1 x 1 x N]`
    - 선택적(그러나 권장함) 레이블 맵은 한 줄에 하나의 레이블을 포함하여 TENSOR_VALUE_LABELS 유형의 AssociatedFile-s로 매핑할 수 있습니다. 첫 번째 AssociatedFile(있는 경우)은 결과의 `label` 필드(C++에서 `class_name`으로 명명됨)를 채우는 데 사용됩니다. `display_name` 필드는 생성 시 사용된 `ImageClassifierOptions`의 `display_names_locale` 필드와 로케일이 일치하는 AssociatedFile(있는 경우)로부터 채워집니다(기본적으로 "en", 즉 영어). 이들 중 어느 것도 사용할 수 없는 경우, 결과의 `index` 필드만 채워집니다.
