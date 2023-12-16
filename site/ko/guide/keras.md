# Keras: TensorFlow용 고수준 API

Keras는 TensorFlow 플랫폼의 고수준 API입니다. Keras는 최신 딥러닝에 중점을 두고 머신러닝(ML) 문제 해결을 위한 접근하기 쉽고 생산성이 높은 인터페이스를 제공합니다. Keras는 데이터 처리부터 하이퍼파라미터 튜닝, 배포에 이르기까지 머신러닝 워크플로의 모든 단계를 다룹니다. Keras는 빠른 실험을 가능하게 하는 것을 목표로 개발되었습니다.

Keras를 사용하면 TensorFlow의 확장성과 크로스 플랫폼 기능에 대한 전체 액세스 권한을 갖습니다. TPU 포드 또는 대규모 GPU 클러스터에서 Keras를 실행할 수 있으며, 브라우저 또는 모바일 장치에서 실행하도록 Keras 모델을 내보낼 수 있습니다. 웹 API를 통해 Keras 모델을 제공할 수도 있습니다.

Keras는 다음과 같은 목표를 달성함으로써 인지 부하를 줄이도록 설계되어 있습니다.

- 간단하고 일관된 인터페이스를 제공합니다.
- 일반적인 사용 사례에 필요한 작업 수를 최소화합니다.
- 명확하고 실행 가능한 오류 메시지를 제공합니다.
- 복잡성을 점진적으로 공개하는 원칙을 따릅니다. 쉽게 시작할 수 있으며, 학습을 진행해 나가며 고급 워크플로를 완성할 수 있습니다.
- 간결하고 읽기 쉬운 코드를 작성하는 데 도움이 됩니다.

## Keras를 사용해야 하는 대상

간단히 말해서, 모든 TensorFlow 사용자는 기본적으로 Keras API를 사용해야 합니다. 엔지니어, 연구원 또는 ML 실무자라면 누구나 Keras로 시작해야 합니다.

저수준의 [TensorFlow Core API](https://www.tensorflow.org/guide/core)가 필요한 몇 가지 사용 사례가 있습니다(예: TensorFlow 위에 도구를 구축하거나 자체 고성능 플랫폼을 개발하는 경우). 다만 자신의 사용 사례가 [코어 API 애플리케이션](https://www.tensorflow.org/guide/core#core_api_applications) 중 하나에 해당하지 않는 경우 Keras를 선호할 수 있습니다.

## Keras API 구성 요소

Keras의 핵심 데이터 구조는 [레이어](https://keras.io/api/layers/)와 [모델](https://keras.io/api/models/)입니다. 레이어는 간단한 입력/출력 변환이고 모델은 레이어의 방향성 비순환 그래프(DAG)입니다.

### 레이어

`tf.keras.layers.Layer` 클래스는 Keras의 기본 추상화입니다. `Layer`는 상태(가중치)와 일부 계산(`tf.keras.layers.Layer.call` 메소드에서 정의됨)을 캡슐화합니다.

레이어로 생성한 가중치는 훈련이 가능하거나 불가능할 수 있습니다. 레이어는 재귀적으로 구성할 수 있습니다: 레이어 인스턴스를 다른 레이어의 속성으로 할당하면 외부 레이어가 내부 레이어에서 생성한 가중치를 추적하기 시작합니다.

또한 레이어를 사용하여 정규화와 텍스트 벡터화와 같은 데이터 전처리 작업을 처리할 수도 있습니다. 전처리 레이어는 훈련 중 또는 훈련 후에 모델에 직접 포함할 수 있으므로 모델을 이식할 수 있습니다.

### 모델

모델은 레이어를 함께 그룹화하고 데이터를 훈련할 수 있는 객체입니다.

가장 간단한 모델 유형은 선형 레이어 스택인 [`Sequential` 모델](https://www.tensorflow.org/guide/keras/sequential_model)입니다. 더 복잡한 아키텍처의 경우, 임의의 레이어 그래프를 빌드할 수 있는 [Keras 함수형 API](https://www.tensorflow.org/guide/keras/functional_api)를 사용하거나 [하위 클래스화를 사용하여 처음부터 모델을 작성](https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing)할 수 있습니다.

`tf.keras.Model` 클래스는 다음과 같은 훈련 및 평가 방법을 기본으로 제공합니다.

- `tf.keras.Model.fit`: 고정된 수의 epoch에 대해 모델을 훈련합니다.
- `tf.keras.Model.predict`: 입력 샘플에 대한 출력 예측을 생성합니다.
- `tf.keras.Model.evaluate`: 모델에 대한 손실 및 메트릭 값을 반환하며, `tf.keras.Model.compile` 메서드를 통해 구성됩니다.

이러한 메서드를 사용하면 다음과 같은 기본 제공 훈련 기능에 액세스할 수 있습니다:

- [콜백](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks). 기본 제공 콜백을 조기 중지, 모델 체크포인트, [TensorBoard](https://www.tensorflow.org/tensorboard) 모니터링에 활용할 수 있습니다. [사용자 정의 콜백을 구현](https://www.tensorflow.org/guide/keras/writing_your_own_callbacks)할 수도 있습니다.
- [분산 훈련](https://www.tensorflow.org/guide/keras/distributed_training). 훈련을 여러 GPU, TPU 또는 장치로 쉽게 확장할 수 있습니다.
- 단계 융합. `tf.keras.Model.compile`의 `steps_per_execution` 인수를 사용하면 한 번의 `tf.function` 호출로 여러 배치를 처리할 수 있으므로 TPU의 기기 활용도가 크게 향상됩니다.

`fit` 사용 방법에 대한 자세한 개요는 [훈련 및 평가 가이드](https://www.tensorflow.org/guide/keras/training_with_built_in_methods)를 참조해 주세요. 기본 제공 훈련 및 평가 루프를 사용자 정의하는 방법은 [`fit()`에서 수행되는 작업 사용자 정의하기](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit)를 참조해 주세요.

### 기타 API 및 도구

Keras는 다음과 같이 다양한 딥러닝용 API와 도구를 제공합니다.

- [옵티마이저](https://keras.io/api/optimizers/)
- [메트릭](https://keras.io/api/metrics/)
- [손실](https://keras.io/api/losses/)
- [데이터 로드 유틸리티](https://keras.io/api/data_loading/)

사용할 수 있는 전체 API 목록은 [Keras API 참조](https://keras.io/api/)를 참조해 주세요. 다른 Keras 프로젝트와 이니셔티브에 대해 자세히 알아보려면 [케라스 생태계](https://keras.io/getting_started/ecosystem/)를 참조해 주세요.

## 다음 단계

TensorFlow에서 Keras 사용을 시작하려면 다음 주제를 확인해보세요.

- [순차형 모델](https://www.tensorflow.org/guide/keras/sequential_model)
- [함수형 API](https://www.tensorflow.org/guide/keras/functional)
- [기본 제공 메서드를 사용하여 훈련 및 평가하기](https://www.tensorflow.org/guide/keras/training_with_built_in_methods)
- [하위 클래스화를 통한 새로운 레이어 및 모델 만들기](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
- [직렬화 및 저장하기](https://www.tensorflow.org/guide/keras/save_and_serialize)
- [전처리 레이어를 사용하여 작업하기](https://www.tensorflow.org/guide/keras/preprocessing_layers)
- [fit()에서 수행되는 작업 사용자 정의하기](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit)
- [훈련 루프 처음부터 작성하기](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)
- [RNN을 사용하여 작업하기](https://www.tensorflow.org/guide/keras/rnn)
- [마스킹 및 패딩 이해하기](https://www.tensorflow.org/guide/keras/masking_and_padding)
- [직접 콜백 작성하기](https://www.tensorflow.org/guide/keras/custom_callback)
- [전이 학습 및 미세 조정하기](https://www.tensorflow.org/guide/keras/transfer_learning)
- [멀티 GPU 및 분산 훈련](https://www.tensorflow.org/guide/keras/distributed_training)

Keras에 대한 자세한 내용은 [keras.io](http://keras.io)에서 다음 주제를 확인해보세요.

- [Keras 정보](https://keras.io/about/)
- [엔지니어를 위한 Keras 소개](https://keras.io/getting_started/intro_to_keras_for_engineers/)
- [연구자를 위한 Keras 소개](https://keras.io/getting_started/intro_to_keras_for_researchers/)
- [Keras API 참조](https://keras.io/api/)
- [Keras 생태계](https://keras.io/getting_started/ecosystem/)
