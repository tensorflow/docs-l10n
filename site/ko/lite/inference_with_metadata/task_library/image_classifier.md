# 이미지 분류자 통합

이미지 분류는 이미지가 나타내는 내용을 식별하기 위해 머신 러닝을 활용하는 일반적인 예입니다. 예를 들어 주어진 그림에 어떤 종류의 동물이 나타나는지 알고 싶을 수 있습니다. 이미지가 나타내는 내용을 예측하는 작업을 *이미지 분류*라고 합니다. 이미지 분류자는 다양한 이미지 클래스를 인식하도록 훈련됩니다. 예를 들어 모델은 토끼, 햄스터, 개 등 세 가지 동물 유형을 나타내는 사진을 인식하도록 훈련될 수 있습니다. 이미지 분류 자에 대한 자세한 내용은 [이미지 분류 개요](../../examples/image_classification/overview)를 참조하세요.

작업 라이브러리 `ImageClassifier` API를 사용하여 사용자 정의 이미지 분류자 또는 사전 훈련된 분류자를 모델 앱에 배포합니다.

## ImageClassifier API의 주요 기능

- 회전, 크기 조정 및 색 공간 변환을 포함한 입력 이미지 처리

- 입력 이미지의 관심 영역

- 레이블 맵 로케일

- 결과를 필터링하기 위한 스코어 임계값

- Top-k 분류 결과

- 레이블 허용 목록 및 거부 목록

## 지원되는 이미지 분류자 모델

다음 모델은 `ImageClassifier` API와의 호환성이 보장됩니다.

- [이미지 분류용 TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/image_classification)에 의해 만들어진 모델

- [TensorFlow 허브의 사전 학습된 이미지 분류 모델](https://tfhub.dev/tensorflow/collections/lite/task-library/image-classifier/1)

- [AutoML Vision Edge 이미지 분류](https://cloud.google.com/vision/automl/docs/edge-quickstart)로 만들어진 모델

- [모델 호환성 요구 사항](#model-compatibility-requirements)을 충족하는 사용자 정의 모델

## Java에서 추론 실행하기

Android 앱에서 `ImageClassifier`를 사용하는 방법의 예는 [이미지 분류 참조 앱](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/EXPLORE_THE_CODE.md)을 참조하세요.

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
    implementation 'org.tensorflow:tensorflow-lite-task-vision'
    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

### 2단계: 모델 사용하기

```java
// Initialization
ImageClassifierOptions options =
    ImageClassifierOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setMaxResults(1)
        .build();
ImageClassifier imageClassifier =
    ImageClassifier.createFromFileAndOptions(
        context, modelFile, options);

// Run inference
List<Classifications> results = imageClassifier.classify(image);
```

<code>ImageClassifier</code> 구성에 대한 추가 옵션은 [소스 코드 및 javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/classifier/ImageClassifier.java)를 참조하세요.

## iOS에서 추론 실행하기

### 1단계: 종속성 설치하기

작업 라이브러리는 CocoaPods를 사용한 설치를 지원합니다. 시스템에 CocoaPods가 설치되어 있는지 확인하세요. 지침은 [CocoaPods 설치 가이드](https://guides.cocoapods.org/using/getting-started.html#getting-started)를 참조하세요.

Xcode 프로젝트에 포드를 추가하는 방법에 대한 자세한 내용은 [CocoaPods 가이드](https://guides.cocoapods.org/using/using-cocoapods.html)를 참조하세요.

Podfile에 `TensorFlowLiteTaskVision` 포드를 추가합니다.

```
target 'MyAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskVision'
end
```

추론에 사용할 `.tflite` 모델이 앱 번들에 있어야 합니다.

### 2단계: 모델 사용하기

#### Swift

```swift
// Imports
import TensorFlowLiteTaskVision

// Initialization
guard let modelPath = Bundle.main.path(forResource: "birds_V1",
                                            ofType: "tflite") else { return }

let options = ImageClassifierOptions(modelPath: modelPath)

// Configure any additional options:
// options.classificationOptions.maxResults = 3

let classifier = try ImageClassifier.classifier(options: options)

// Convert the input image to MLImage.
// There are other sources for MLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
guard let image = UIImage (named: "sparrow.jpg"), let mlImage = MLImage(image: image) else { return }

// Run inference
let classificationResults = try classifier.classify(mlImage: mlImage)
```

#### Objective C

```objc
// Imports
#import <TensorFlowLiteTaskVision/TFLTaskVision.h>

// Initialization
NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"birds_V1" ofType:@"tflite"];

TFLImageClassifierOptions *options =
    [[TFLImageClassifierOptions alloc] initWithModelPath:modelPath];

// Configure any additional options:
// options.classificationOptions.maxResults = 3;

TFLImageClassifier *classifier = [TFLImageClassifier imageClassifierWithOptions:options
                                                                          error:nil];

// Convert the input image to MLImage.
UIImage *image = [UIImage imageNamed:@"sparrow.jpg"];

// There are other sources for GMLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
GMLImage *gmlImage = [[GMLImage alloc] initWithImage:image];

// Run inference
TFLClassificationResult *classificationResult =
    [classifier classifyWithGMLImage:gmlImage error:nil];
```

<code>TFLImageClassifier</code> 구성에 대한 추가 옵션은 <a>소스 코드</a>를 참조하세요.

## Python에서 추론 실행하기

### 1단계: pip 패키지 설치하기

```
pip install tflite-support
```

### 2단계: 모델 사용하기

```python
# Imports
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor

# Initialization
base_options = core.BaseOptions(file_name=model_path)
classification_options = processor.ClassificationOptions(max_results=2)
options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)
classifier = vision.ImageClassifier.create_from_options(options)

# Alternatively, you can create an image classifier in the following manner:
# classifier = vision.ImageClassifier.create_from_file(model_path)

# Run inference
image = vision.TensorImage.create_from_file(image_path)
classification_result = classifier.classify(image)
```

<code>ImageClassifier</code> 구성에 대한 추가 옵션은 <a>소스 코드</a>를 참조하세요.

## C++에서 추론 실행하기

```c++
// Initialization
ImageClassifierOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<ImageClassifier> image_classifier = ImageClassifier::CreateFromOptions(options).value();

// Create input frame_buffer from your inputs, `image_data` and `image_dimension`.
// See more information here: tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h

std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      image_data, image_dimension);

// Run inference
const ClassificationResult result = image_classifier->Classify(*frame_buffer).value();
```

`ImageClassifier` 구성에 대한 추가 옵션은 [소스 코드](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/image_classifier.h)를 참조하세요.

## 예제 결과

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

## 모델 호환성 요구 사항

`ImageClassifier` API는 필수 [TFLite 모델 메타데이터](../../models/convert/metadata)가 있는 TFLite 모델을 예상합니다. [TensorFlow Lite Metadata Writer API](../../models/convert/metadata_writer_tutorial.ipynb#image_classifiers)를 사용하여 이미지 분류자에 대한 메타데이터를 생성하는 예를 참조하세요.

호환되는 이미지 분류자 모델은 다음 요구 사항을 충족해야 합니다.

- 입력 이미지 텐서(kTfLiteUInt8/kTfLiteFloat32)

    - 이미지 입력 크기는 `[batch x height x width x channels]`입니다.
    - 배치 추론은 지원되지 않습니다(`batch`는 1이어야 함).
    - RGB 입력만 지원됩니다(`channels`은 3이어야 함).
    - 유형이 kTfLiteFloat32인 경우, 입력 정규화를 위해 NormalizationOptions를 메타데이터에 첨부해야 합니다.

- 출력 점수 텐서(kTfLiteUInt8/kTfLiteFloat32)

    - `N` 클래스와 2 또는 4 차원, 즉 `[1 x N]` 또는 `[1 x 1 x 1 x N]`
    - 선택적(그러나 권장함) 레이블 맵은 한 줄에 하나의 레이블을 포함하여 TENSOR_AXIS_LABELS 유형의 AssociatedFile-s로 매핑할 수 있습니다. [예제 레이블 파일](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/python/tests/testdata/image_classifier/labels.txt)을 참조하세요. 이러한 첫 번째 AssociatedFile(있는 경우)은 결과의 `label` 필드(C++에서 `class_name`으로 명명됨)를 채우는 데 사용됩니다. `display_name` 필드는 생성 시 사용된 `ImageClassifierOptions`의 `display_names_locale` 필드와 로케일이 일치하는 AssociatedFile(있는 경우)로부터 채워집니다(기본적으로 "en", 즉 영어). 이들 중 어느 것도 사용할 수 없는 경우, 결과의 `index` 필드만 채워집니다.
