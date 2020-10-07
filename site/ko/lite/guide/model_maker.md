# TensorFlow Lite Model Maker

## Overview

TensorFlow Lite Model Maker 라이브러리는 사용자 정의 데이터세트를 사용하여 TensorFlow Lite 모델 훈련 프로세스를 단순화합니다. 전이 학습을 사용하여 필요한 훈련 데이터의 양을 줄이고 훈련 시간을 단축할 수 있습니다.

## Supported Tasks

Model Maker 라이브러리는 현재 다음 ML 작업을 지원합니다. 모델 훈련 방법에 대한 가이드를 보려면 아래 링크를 클릭하세요.

Supported Tasks | Task Utility
--- | ---
Image Classification [guide](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification) | 이미지를 미리 정의된 범주로 분류합니다.
Text Classification [guide](https://www.tensorflow.org/lite/tutorials/model_maker_text_classification) | 텍스트를 미리 정의된 범주로 분류합니다.
Question Answer [guide](https://www.tensorflow.org/lite/tutorials/model_maker_question_answer) | 주어진 질문에 대한 특정 컨텍스트에서 답변을 찾습니다.

## 엔드 투 엔드 예제

Model Maker를 사용하면 단 몇 줄의 코드로 사용자 정의 데이터세트를 사용하여 TensorFlow Lite 모델을 훈련할 수 있습니다. 예를 들어, 다음은 이미지 분류 모델을 훈련하는 단계입니다.

```python
# Load input data specific to an on-device ML app.
data = ImageClassifierDataLoader.from_folder('flower_photos/')
train_data, test_data = data.split(0.9)

# Customize the TensorFlow model.
model = image_classifier.create(data)

# Evaluate the model.
loss, accuracy = model.evaluate(test_data)

# Export to Tensorflow Lite model and label file in `export_dir`.
model.export(export_dir='/tmp/')
```

자세한 내용은 [이미지 분류 가이드](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification)를 참조하세요.

## Installation

There are two ways to install Model Maker.

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
