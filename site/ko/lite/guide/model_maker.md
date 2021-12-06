# TensorFlow Lite Model Maker

## 개요

TensorFlow Lite Model Maker 라이브러리는 사용자 정의 데이터세트를 사용하여 TensorFlow Lite 모델 훈련 프로세스를 단순화합니다. 전이 학습을 사용하여 필요한 훈련 데이터의 양을 줄이고 훈련 시간을 단축할 수 있습니다.

## 지원되는 작업

Model Maker 라이브러리는 현재 다음 ML 작업을 지원합니다. 모델 훈련 방법에 대한 가이드를 보려면 아래 링크를 클릭하세요.

지원되는 작업 | 작업 유틸리티
--- | ---
이미지 분류: [튜토리얼](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification), [API](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/image_classifier) | 이미지를 미리 정의된 범주로 분류합니다.
객체 감지: [튜토리얼](https://www.tensorflow.org/lite/tutorials/model_maker_object_detection), [API](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/object_detector) | 실시간으로 객체를 감지합니다.
텍스트 분류: [튜토리얼](https://www.tensorflow.org/lite/tutorials/model_maker_text_classification), [API](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/text_classifier) | 텍스트를 미리 정의된 범주로 분류합니다.
BERT 질문 답변: [튜토리얼](https://www.tensorflow.org/lite/tutorials/model_maker_question_answer), [API](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/question_answer) | BERT에 관해 주어진 질문에 대한 특정 컨텍스트에서 답변을 찾습니다.
오디오 분류: [튜토리얼](https://www.tensorflow.org/lite/tutorials/model_maker_audio_classification), [API](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/audio_classifier) | 오디오를 미리 정의된 범주로 분류합니다.
권장 사항: [데모](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/demo/recommendation_demo.py), [API](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/recommendation) | 장치 내 시나리오에 대한 컨텍스트 정보를 바탕으로 항목을 권장합니다.

작업이 지원되지 않는 경우, 먼저 [TensorFlow](https://www.tensorflow.org/guide)를 사용하여 전이 학습으로 TensorFlow 모델을 재훈련하거나([images](https://www.tensorflow.org/tutorials/images/transfer_learning), [text](https://www.tensorflow.org/official_models/fine_tuning_bert), [audio](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio)와 같은 가이드를 따름) 처음부터 훈련한 다음 TensorFlow Lite 모델로 [변환](https://www.tensorflow.org/lite/convert)하세요.

## 엔드 투 엔드 예제

Model Maker를 사용하면 단 몇 줄의 코드로 사용자 정의 데이터세트를 사용하여 TensorFlow Lite 모델을 훈련할 수 있습니다. 예를 들어, 다음은 이미지 분류 모델을 훈련하는 단계입니다.

```python
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

# Load input data specific to an on-device ML app.
data = DataLoader.from_folder('flower_photos/')
train_data, test_data = data.split(0.9)

# Customize the TensorFlow model.
model = image_classifier.create(train_data)

# Evaluate the model.
loss, accuracy = model.evaluate(test_data)

# Export to Tensorflow Lite model and label file in `export_dir`.
model.export(export_dir='/tmp/')
```

자세한 내용은 [이미지 분류 가이드](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification)를 참조하세요.

## 설치

Model Maker를 설치하는 방법에는 두 가지가 있습니다.

- 미리 빌드된 pip 패키지를 설치합니다.

```shell
pip install tflite-model-maker
```

야간 버전을 설치하려면 다음 명령을 따릅니다.

```shell
pip install tflite-model-maker-nightly
```

- GitHub에서 소스 코드를 복사하고 설치합니다.

```shell
git clone https://github.com/tensorflow/examples
cd examples/tensorflow_examples/lite/model_maker/pip_package
pip install -e .
```

TensorFlow Lite Model Maker는 TensorFlow [pip 패키지](https://www.tensorflow.org/install/pip)에 의존합니다. GPU 드라이버는 TensorFlow의 [GPU 가이드](https://www.tensorflow.org/install/gpu) 또는 [설치 가이드](https://www.tensorflow.org/install)를 참조하세요.

## Python API 참조

[API 참조](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker)에서 Model Maker의 공개 API를 찾을 수 있습니다.
