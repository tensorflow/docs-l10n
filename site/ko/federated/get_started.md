# TensorFlow 페더레이션

TensorFlow Federated(TFF) 플랫폼은 두 개의 레이어로 구성됩니다.

- [Federated Learning(FL)](federated_learning.md), 기존 Keras 또는 비 Keras 머신러닝 모델을 TFF 프레임워크에 연결하는 상위 수준 인터페이스입니다. 페데레이션 학습 알고리즘의 세부 사항을 연구하지 않고도 페데레이션 훈련 또는 평가와 같은 기본 작업을 수행할 수 있습니다.
- [Federated Core(FC)](federated_core.md), 강력한 형식의 함수형 프로그래밍 환경 내에서 TensorFlow와 분산 통신 연산자를 결합하여 사용자 정의 페데레이션 알고리즘을 간결하게 표현하는 하위 수준 인터페이스입니다.

실제 예제를 사용하여 주요 TFF 개념 및 API를 안내하는 다음 튜토리얼을 읽어보세요. TFF를 사용할 환경을 구성하려면 [설치 지침](install.md)을 따르세요.

- [이미지 분류를 위한 Federated Learning](tutorials/federated_learning_for_image_classification.ipynb)은 Federated Learning(FL) API의 주요 부분을 소개하고 TFF를 사용하여 페데레이션 MNIST와 유사한 데이터에 대한 페데레이션 학습을 시뮬레이션하는 방법을 보여줍니다.
- [텍스트 생성을 위한 Federated Learning](tutorials/federated_learning_for_text_generation.ipynb)은 TFF의 FL API를 사용하여 언어 모델링 작업의 직렬화된 사전 훈련된 모델을 구체화하는 방법을 추가로 보여줍니다.
- [사용자 정의 페더레이션 알고리즘, 1부: Federated Core 소개](tutorials/custom_federated_algorithms_1.ipynb) 및 [2부: Federated Averaging 구현](tutorials/custom_federated_algorithms_2.ipynb)은 Federated Core API(FC API)에서 제공하는 주요 개념과 인터페이스를 소개하고, 간단한 페더레이션 평균화 훈련 알고리즘을 구현하는 방법과 페더레이션 평가를 수행하는 방법을 보여줍니다.
