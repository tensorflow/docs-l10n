# TensorFlow Lite Task 라이브러리

TensorFlow Lite Task 라이브러리에는 앱 개발자가 TFLite로 ML 경험을 만들 수 있는 강력하고 사용하기 쉬운 작업별 라이브러리 세트가 포함되어 있습니다. 이미지 분류, 질문 및 답변 등과 같은 주요 머신 러닝 작업에 최적화된 기본 제공 모델 인터페이스가 제공됩니다. 모델 인터페이스는 각 작업에 맞게 특별히 설계되어 최상의 성능과 유용성을 제공합니다. Task 라이브러리는 크로스 플랫폼에서 작동하며 Java, C++ 및 Swift에서 지원됩니다.

## Task 라이브러리에서 기대할 수 있는 사항

- **ML 전문가가 아니더라도 사용할 수 있는 명료하고 잘 구성된 API** <br> 단 5줄의 코드 내에서 추론을 수행할 수 있습니다. Task 라이브러리의 강력하고 사용하기 쉬운 API를 빌딩 블록으로 사용하여 모바일 기기에서 TFLite로 ML을 쉽게 개발할 수 있습니다.

- **복잡하지만 일반적인 데이터 처리** <br> 공통 비전 및 자연어 처리 논리를 지원하여 데이터와 모델에 필요한 데이터 형식 사이에서 변환할 수 있습니다. 학습 및 추론에 동일하고 공유 가능한 처리 로직을 제공합니다.

- **높은 성능 이득** <br> 데이터 처리에 수 밀리 초 이상 걸리지 않으므로 TensorFlow Lite를 사용한 빠른 추론 경험이 보장됩니다.

- **확장성 및 사용자 정의 기능**<br> Task 라이브러리 인프라가 제공하는 모든 이점을 활용하고 자신만의 Android/iOS 추론 API를 쉽게 구축할 수 있습니다.

## 지원되는 작업

다음은 지원되는 작업 유형의 목록입니다. 점차 더 많은 사용 사례가 계속 개발됨에 따라 이 목록은 더 늘어날 것으로 예상됩니다.

- **비전 API**

    - [ImageClassifier](image_classifier.md)
    - [ObjectDetector](object_detector.md)
    - [ImageSegmenter](image_segmenter.md)

- **자연어(NL) API**

    - [NLClassifier](nl_classifier.md)
    - [BertNLCLassifier](bert_nl_classifier.md)
    - [BertQuestionAnswerer](bert_question_answerer.md)

- **사용자 정의 API**

    - Task API 인프라를 확장하고 [사용자 정의 API](customized_task_api.md)를 구축합니다.
