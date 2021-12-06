# 메타데이터를 이용한 TensorFlow Lite 추론

[메타데이터를 사용하여 모델](../convert/metadata.md)을 추론하는 작업은 코드 몇 줄만 작성하면 될 만큼 쉽습니다. TensorFlow Lite 메타데이터에는 모델이 하는 작업과 모델 사용 방법에 대한 풍부한 설명이 포함되어 있습니다. 이를 통해 코드 생성기가 [Android Studio ML Binding 특성](codegen.md#mlbinding)이나 [TensorFlow Lite Android 코드 생성기](codegen.md#codegen)를 사용하는 등 추론 코드를 자동으로 생성할 수 있습니다. 또한 사용자 정의 추론 파이프라인을 구성하는 데 사용할 수도 있습니다.

## 도구 및 라이브러리

TensorFlow Lite는 다음과 같이 다양한 계층의 배포 요구 사항을 해결하도록 다양한 도구와 라이브러리를 제공합니다.

### Generate model interface with Android code generators

메타데이터를 사용하여 TensorFlow Lite 모델에 필요한 Android 래퍼 코드를 자동으로 생성하는 방법에는 다음과 같이 두 가지가 있습니다.

1. [Android Studio ML Model Binding](codegen.md#mlbinding)은 그래픽 인터페이스를 통해 TensorFlow Lite 모델을 가져오기 위해 Android 스튜디오 내에서 사용할 수 있는 도구입니다. Android Studio는 프로젝트에 대한 설정을 자동으로 구성하고 모델 메타데이터를 기반으로 래퍼 클래스를 생성합니다.

2. [TensorFlow Lite 코드 생성기](codegen.md)는 메타데이터를 기반으로 모델 인터페이스를 자동으로 생성하는 실행 파일입니다. 현재 Java가 설치된 Android를 지원합니다. 래퍼 코드가 있어서 `ByteBuffer`와 직접 상호 작용할 필요가 없습니다. 대신, 개발자는 `Bitmap` 및 `Rect`와 같은 형식화된 객체를 사용하여 TensorFlow Lite 모델과 상호 작용할 수 있습니다. Android Studio 사용자는 [Android Studio ML Binding](codegen.md#generate-code-with-android-studio-ml-model-binding)을 통해 codegen 특성에 액세스할 수도 있습니다.

### TensorFlow Lite Task Library로 즉시 사용 가능한 API 활용하기

[TensorFlow Lite Task Library](task_library/overview.md)는 이미지 분류, 질문 및 답변 등과 같은 주요 머신러닝 작업에 즉시 사용 가능한 최적화된 모델 인터페이스를 제공합니다. 모델 인터페이스는 각 작업에 맞게 특별히 설계되어 최상의 성능과 유용성을 제공합니다. Task Library는 교차 플랫폼으로 동작하며 Java, C++ 및 Swift에서 지원됩니다.

### TensorFlow Lite Support Library로 사용자 정의 추론 파이프라인 빌드하기

[TensorFlow Lite Support Library](lite_support.md)는 모델 인터페이스를 사용자 정의하고 추론 파이프라인을 구축하는 데 도움을 주는 교차 플랫폼 라이브러리입니다. 여기에는 사전/사후 처리 및 데이터 변환을 수행하기 위한 다양한 util 메서드와 데이터 구조가 포함되어 있습니다. 또한 TF.Image 및 TF.Text와 같은 TensorFlow 모듈의 동작과 일치하도록 설계되어 훈련에서 추론까지 일관성을 보장합니다.

## 메타데이터를 포함한 사전 훈련된 모델 찾아보기

[TensorFlow Lite 호스팅 모델](https://www.tensorflow.org/lite/guide/hosted_models) 및 [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite)를 검색하여 비전 및 텍스트 작업에 모두 사용할 수 있는 메타데이터가 포함된 사전 훈련된 모델을 다운로드하세요. 또한 [메타데이터 시각화](../convert/metadata.md#visualize-the-metadata)의 다양한 옵션을 참조하세요.

## TensorFlow Lite Support GitHub repo

더 많은 예제와 소스 코드를 보려면 [TensorFlow Lite Support GitHub 저장소](https://github.com/tensorflow/tflite-support)를 방문하세요. [새로운 GitHub 문제](https://github.com/tensorflow/tflite-support/issues/new)를 생성하여 피드백을 보내주세요.
