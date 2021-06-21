# What-If Tool 대시보드를 사용한 모델 이해

![What-If Tool](./images/what_if_tool.png)

WIT(What-If Tool)는 블랙 박스 분류 및 회귀 ML 모델에 대한 이해를 확장할 수 있는 사용하기 쉬운 인터페이스를 제공합니다. 이 플러그인을 사용하면 많은 예제에서 추론을 수행하고 다양한 방법으로 결과를 즉시 시각화할 수 있습니다. 또한, 예제를 수동 또는 프로그래밍 방식으로 편집하고 모델을 다시 실행하여 변경 결과를 확인할 수 있습니다. 여기에는 데이터세트의 하위 세트에 대한 모델 성능과 공정성을 조사하기 위한 도구가 포함되어 있습니다.

이 도구의 목적은 코드가 전혀 필요하지 않은 시각적 인터페이스를 통해 훈련된 ML 모델을 탐색하고 조사할 수 있는 간단하고 직관적이며 강력한 방법을 제공하는 것입니다.

이 도구는 TensorBoard를 통해, 또는 Jupyter 또는 Colab 노트북에서 직접 액세스할 수 있습니다. 노트북 모드에서 WIT 사용과 관련된 자세한 내용, 데모, 연습 및 정보를 보려면 [What-If Tool 웹 사이트](https://pair-code.github.io/what-if-tool)를 참조하세요.

## 요구 사항

TensorBoard에서 WIT를 사용하려면 두 가지가 필요합니다.

- 탐색하려는 모델은 분류, 회귀 또는 예측 API를 사용하여 [TensorFlow Serving](https://github.com/tensorflow/serving)에서 제공되어야 합니다.
- 모델에서 추론할 데이터세트는 TensorBoard 웹 서버에서 액세스할 수 있는 TFRecord 파일에 있어야 합니다.

## 사용법

TensorBoard에서 What-If Tool 대시보드를 열면, 모델 서버의 호스트와 포트, 제공할 모델 이름, 모델 유형 및 로드할 TFRecords 파일의 경로를 입력해야 하는 설정 화면이 표시됩니다. 이 정보를 입력하고 "Accept"를 클릭하면 WIT가 데이터세트를 로드하고 모델에 대한 추론을 실행하여 결과를 표시합니다.

WIT의 다양한 기능과 이러한 기능이 모델에 대한 이해와 공정성을 조사하는 데 어떤 도움을 주는지 자세히 알아보려면 [What-If Tool 웹 사이트](https://pair-code.github.io/what-if-tool)의 연습을 참조하세요.

## 데모 모델 및 데이터세트

사전 훈련된 모델로 TensorBoard에서 WIT를 테스트하려면 https://storage.googleapis.com/what-if-tool-resources/uci-census-demo/uci-census-demo.zip에서 사전 훈련된 모델과 데이터세트를 다운로드하고 압축을 풉니다. 이 모델은 [UCI Census](https://archive.ics.uci.edu/ml/datasets/census+income) 데이터세트를 사용하여 어떤 개인의 연간 수입이 $50k 이상인지를 예측하는 이진 분류 모델입니다. 이 데이터세트와 예측 작업은 머신러닝 모델링과 공정성 연구에 종종 이용됩니다.

환경 변수 MODEL_PATH를 머신의 결과 모델 디렉토리 위치로 설정합니다.

[공식 설명서](https://www.tensorflow.org/serving/docker)에 따라 docker 및 TensorFlow Serving을 설치합니다.

`docker run -p 8500:8500 --mount type=bind,source=${MODEL_PATH},target=/models/uci_income -e MODEL_NAME=uci_income -t tensorflow/serving`으로 docker를 이용해 모델을 제공합니다. 해당 docker 설정에 따라 `sudo`으로 명령을 실행해야 할 수도 있습니다.

이제 TensorBoard를 시작하고 대시보드 드롭다운을 사용하여 What-If Tool로 이동합니다.

설정 화면에서 추론 주소를 "localhost:8500"으로, 모델 이름을 "uci_income"으로, 예제 경로를 다운로드한 `adult.tfrecord` 파일의 전체 경로로 설정한 다음 "Accept"를 누릅니다.

![데모 설정 화면](./images/what_if_tool_demo_setup.png)

이 데모에서는 What-If Tool로 다음과 같은 몇 가지를 시도합니다.

- 단일 데이터 포인트를 편집하고 결과적인 추론의 변경 확인
- 부분 종속성 플롯을 통해 데이터세트의 개별 특성과 모델의 추론 결과 간의 관계 검토
- 데이터세트를 하위 세트로 분할하고 조각 간의 성능 비교

이 도구의 기능에 대한 자세한 내용은 [What-If Tool 연습](https://pair-code.github.io/what-if-tool/walkthrough.html)을 확인하세요.

이 모델이 예측하려는 데이터세트의 지상 실측 특성이 "Target"이라는 점에 주목하세요. 따라서 "Performance &amp; Fairness" 탭을 사용할 때 "Target"은 지상 실측 특성 드롭다운에서 지정하려는 항목입니다.
