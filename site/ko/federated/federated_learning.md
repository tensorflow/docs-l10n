# 페더레이션 학습

## 개요

이 문서에서는 TensorFlow에서 구현된 기존 머신러닝 모델을 사용한 페더레이션 훈련 또는 평가와 같은 페더레이션 학습 작업을 용이하게 하는 인터페이스를 소개합니다. 인터페이스를 설계할 때의 주요 목표는 내부 동작에 대한 지식이 없이도 페더레이션 학습을 실험하고 다양한 기존 모델 및 데이터에서 페더레이션 학습 알고리즘을 평가하는 것이었습니다. 여러분이 플랫폼에 많이 기여해 주시길 바랍니다. TFF는 확장성과 구성성을 염두에 두고 설계되었으며 여러분의 기여를 환영합니다. 많은 의견을 기대합니다!

이 레이어에서 제공하는 인터페이스는 다음 3가지 주요 부분으로 구성됩니다.

- **모델**: TFF에서 사용하기 위해 기존 모델을 래핑할 수 있는 클래스 및 도우미 함수입니다. 모델 래핑은 단일 래핑 함수(예: `tff.learning.from_keras_model`)를 호출하거나 전체 사용자 정의를 위해 `tff.learning.Model` 인터페이스의 서브 클래스를 정의하는 것처럼 간단할 수 있습니다.

- **페더레이션 계산 빌더**: 기존 모델을 사용하여 훈련 또는 평가를 위한 페더레이션 계산을 구성하는 도우미 함수입니다.

- **데이터세트**: 페더레이션 학습 시나리오의 시뮬레이션에 사용하기 위해 다운로드하여 Python에서 액세스할 수 있는 미리 준비된 데이터 모음입니다. 페더레이션 학습은 중앙 집중식 위치에서 간단히 다운로드할 수 없는 분산 데이터와 함께 사용하도록 설계되었지만, 로컬에서 다운로드하여 조작할 수 있는 데이터를 사용하여 초기 실험을 수행하는 것이 종종 편리한데, 특히 이 접근 방식을 처음 접하는 개발자의 경우 더욱 그렇습니다.

이들 인터페이스는 연구 데이터세트 및 `tff.simulation`으로 그룹화된 기타 시뮬레이션 관련 기능을 제외하고 주로 `tff.learning` 네임스페이스에 정의됩니다. 이 레이어는 런타임 환경을 제공하는, [FC (Federated Core)](federated_core.md)에서 제공하는 하위 레벨 인터페이스를 사용하여 구현됩니다.

계속하기 전에, [image classification](tutorials/federated_learning_for_image_classification.ipynb) 및 [text generation](tutorials/federated_learning_for_text_generation.ipynb)에 대한 튜토리얼을 먼저 검토하는 것이 좋은데, 구체적인 예를 사용하여 여기에 설명된 대부분의 개념을 소개하고 있기 때문입니다. TFF의 동작 방식에 대해 더 자세히 알고 싶다면, 페더레이션 계산의 논리를 표현하고 `tff.learning` 인터페이스의 기존 구현을 연구하는 데 사용하는 하위 레벨 인터페이스에 대한 소개로 [사용자 지정 알고리즘 튜토리얼](tutorials/custom_federated_algorithms_1.ipynb)을 살펴볼 수 있습니다.

## 모델

### 설계상 가정

#### 직렬화

TFF는 작성하는 머신러닝 모델 코드가 다양한 기능을 가진 수많은 이기종 클라이언트에서 실행될 수 있는 다양한 분산 학습 시나리오를 지원하는 것을 목표로 합니다. 일부 애플리케이션에서 해당 클라이언트가 강력한 데이터베이스 서버가 될 수도 있지만, 당사 플랫폼에서 지원하려는 많은 중요한 용도에는 제한된 리소스를 가진 모바일 및 임베디드 기기가 포함됩니다. 이들 기기가 Python 런타임을 호스팅할 수 있다고 가정할 수는 없습니다. 이 시점에서 가정할 수 있는 것은 이들 기기에서 로컬 TensorFlow 런타임을 호스팅할 수 있다는 것입니다. 따라서 TFF에서 설계상 기본 가정은 모델 코드를 TensorFlow 그래프로 직렬화할 수 있어야 한다는 것입니다.

즉시 모드를 사용하는 것과 같은 최신 모범 사례에 따라 TF 코드를 계속 개발할 수 있고 또 그래야 합니다. 그러나 최종 코드는 직렬화 가능해야 합니다(예: 즉시 모드 코드의 경우 `tf.function`로 래핑될 수 있음). 이를 통해 실행 시 필요한 Python 상태 또는 제어 흐름을 직렬화할 수 있습니다([Autograph](https://www.tensorflow.org/guide/autograph)의 도움으로).

현재 TensorFlow는 즉시 모드 TensorFlow의 직렬화 및 역직렬화를 완전히 지원하지 않습니다. 따라서 TFF에서 직렬화는 현재 TF 1.0 패턴을 따르며, 모든 코드는 TFF가 제어하는 `tf.Graph` 내에 구성되어야 합니다. 이것은 현재 TFF가 이미 구성된 모델을 소비할 수 없음을 의미합니다. 대신, 모델 정의 로직은 `tff.learning.Model`을 반환하는 인수 없는 함수로 패키지됩니다. 그런 다음, 이 함수를 TFF에서 호출하여 모델의 모든 컴포넌트가 직렬화되도록 합니다. 또한, 강력한 형식의 환경인 TFF에는 모델의 입력 유형 사양과 같은 약간의 추가 *메타 데이터*가 필요합니다.

#### 집계

대부분의 사용자는 Keras를 사용하여 모델을 구성하는 것이 좋습니다. 아래 [Keras용 변환기](#converters-for-keras) 섹션을 참조하세요. 이들 래퍼는 모델 업데이트 및 모델에 대해 정의된 모든 메트릭의 집계를 자동으로 처리합니다. 그러나 일반 `tff.learning.Model`에 대한 집계가 처리되는 방식을 이해하는 것이 여전히 유용할 수 있습니다.

페더레이션 학습에는 로컬 기기 내 집계 및 교차 기기(또는 페더레이션) 집계라는 최소한 두 개의 집계 레이어가 있습니다.

- **로컬 집계**: 이 레벨의 집계는 개별 클라이언트가 소유한 예제의 여러 배치에 대한 집계를 나타냅니다. 모델이 로컬에서 훈련됨에 따라 순차적으로 계속 진화하는 두 모델 매개변수(변수) 모두와 각 개별 클라이언트의 로컬 데이터 스트림을 반복할 때 모델이 로컬에서 다시 업데이트하는 통계(예: 평균 손실, 정확성 및 기타 메트릭)에 적용됩니다.

    이 레벨에서 집계를 수행하는 것은 모델 코드의 책임이며 표준 TensorFlow 구문을 사용하여 수행됩니다.

    처리의 일반적인 구조는 다음과 같습니다.

    - 모델은 먼저 `tf.Variable`를 구성하여 배치 수 또는 처리된 예제 수, 배치당 또는 예제당 손실의 합계 등과 같은 집계를 보유합니다.

    - TFF는 클라이언트 데이터의 후속 배치에 대해 순차적으로 `Model`에서 `forward_pass` 메서드를 여러 번 호출합니다. 부작용으로 다양한 집계를 보유하는 변수를 업데이트할 수 있습니다.

    - Finally, TFF invokes the `report_local_unfinalized_metrics` method on your Model to allow your model to compile all the summary statistics it collected into a compact set of metrics to be exported by the client. This is where your model code may, for example, divide the sum of losses by the number of examples processed to export the average loss, etc.

- **페더레이션 집계**: 이 레벨의 집계는 시스템의 여러 클라이언트(기기)에 대한 집계를 나타냅니다. 다시 말하자면, 클라이언트 전체에서 평균화되는 모델 매개변수(변수)와 로컬 집계의 결과로서 모델이 내보낸 메트릭에 모두 적용됩니다.

    이 레벨에서 집계를 수행하는 것은 TFF의 책임입니다. 그러나 모델 작성자는 이 프로세스를 제어할 수 있습니다(자세한 내용은 아래 참조).

    처리의 일반적인 구조는 다음과 같습니다.

    - 초기 모델 및 훈련에 필요한 모든 매개변수는 서버에서 훈련 또는 평가 라운드에 참여할 클라이언트의 하위 세트로 배포됩니다.

    - 각 클라이언트에서 독립적으로, 그리고 병렬로, 로컬 데이터 배치의 스트림에서 모델 코드가 반복적으로 호출되어 새로운 모델 매개변수 세트(훈련 시)와 위에 설명된 대로(로컬 집계) 새로운 로컬 메트릭 세트를 생성합니다.

    - TFF runs a distributed aggregation protocol to accumulate and aggregate the model parameters and locally exported metrics across the system. This logic is expressed in a declarative manner using TFF's own *federated computation* language (not in TensorFlow). See the [custom algorithms](tutorials/custom_federated_algorithms_1.ipynb) tutorial for more on the aggregation API.

### 추상 인터페이스

이 기본 *생성자* + *메타 데이터* 인터페이스는 다음과 같이 `tff.learning.Model` 인터페이스로 표시됩니다.

- The constructor, `forward_pass`, and `report_local_unfinalized_metrics` methods should construct model variables, forward pass, and statistics you wish to report, correspondingly. The TensorFlow constructed by those methods must be serializable, as discussed above.

- `input_spec` 속성과 훈련 가능, 훈련 불가능 및 로컬 변수의 하위 세트를 반환하는 3개의 속성은 메타 데이터를 나타냅니다. TFF는 이 정보를 사용하여 모델의 일부를 페더레이션 최적화 알고리즘에 연결하는 방법을 결정하고, 생성된 시스템의 정확성을 확인하는 데 도움이 되는 내부 유형 시그니처를 정의합니다(모델이 소비하도록 설계된 것과 일치하지 않는 데이터에 대해 모델을 인스턴스화할 수 없도록).

In addition, the abstract interface `tff.learning.Model` exposes a property `metric_finalizers` that takes in a metric's unfinalized values (returned by `report_local_unfinalized_metrics()`) and returns the finalized metric values. The `metric_finalizers` and `report_local_unfinalized_metrics()` method will be used together to build a cross-client metrics aggregator when defining the federated training processes or evaluation computations. For example, a simple `tff.learning.metrics.sum_then_finalize` aggregator will first sum the unfinalized metric values from clients, and then call the finalizer functions at the server.

당신은 당신의 자신의 사용자 정의 정의하는 방법의 예를 찾을 수 있습니다 `tf.learning.Model` 우리의 두 번째 부분에서 [이미지 분류](tutorials/federated_learning_for_image_classification.ipynb) 뿐만 아니라 우리가 테스트에 사용할 예제 모델, 튜토리얼 [`model_examples.py`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/model_examples.py) .

### Keras용 변환기

TFF에 필요한 거의 모든 정보는 `tf.keras` 인터페이스를 호출하여 얻을 수 있으므로 Keras 모델이 있다면, `tff.learning.from_keras_model`을 사용하여 `tff.learning.Model`을 구성할 수 있습니다.

TFF는 여전히 다음과 같은 생성자(인수가 없는 *모델 함수*)를 제공하기를 원합니다.

```python
def model_fn():
  keras_model = ...
  return tff.learning.from_keras_model(keras_model, sample_batch, loss=...)
```

모델 자체 외에도, TFF가 모델 입력의 유형 및 형상을 결정하는 데 사용하는 샘플 데이터 배치를 제공합니다. 이렇게 하면 TFF가 클라이언트 기기에 실제로 존재하는 데이터를 위한 모델을 올바르게 인스턴스화할 수 있습니다 (여러분이 직렬화하려는 TensorFlow를 구성할 때 이 데이터는 일반적으로 사용할 수 없다고 가정하기 때문에).

Keras 래퍼의 사용은 [이미지 분류](tutorials/federated_learning_for_image_classification.ipynb) 및 [텍스트 생성](tutorials/federated_learning_for_text_generation.ipynb) 튜토리얼에 설명되어 있습니다.

## 페더레이션 계산 빌더

`tff.learning` 패키지는 학습 관련 작업을 수행하는 `tff.Computation`을 위한 여러 빌더를 제공합니다. 앞으로 계산 세트가 확장될 것으로 기대합니다.

### 설계상 가정

#### 실행

페더레이션 계산을 실행하는 데는 두 가지 단계가 있습니다.

- **컴파일**: TFF는 먼저 페더레이션 학습 알고리즘을 전체 분산 계산의 직렬화된 추상 표현으로 *컴파일*합니다. 이때 TensorFlow 직렬화가 발생하지만, 보다 효율적인 실행을 지원하기 위해 다른 변환이 발생할 수 있습니다. 컴파일러에 의해 생성된 직렬화된 표현을 *페더레이션 계산*이라고 합니다.

- **실행**: TFF는 이들 계산을 *실행*할 수 있는 방법을 제공합니다. 현재로서는 로컬 시뮬레이션을 통해서만 실행이 지원됩니다(예: 시뮬레이션 분산 데이터를 사용하는 노트북에서).

[페더레이션 모델 평균화](https://arxiv.org/abs/1602.05629)를 사용하는 훈련 알고리즘 또는 페더레이션 평가와 같은, TFF의 Federated Learning API에 의해 생성된 페더레이션 계산에는 다음과 같은 여러 요소가 포함됩니다.

- 모델 코드의 직렬화된 형식과 모델의 훈련/평가 루프(예: 옵티마이저 구성, 모델 업데이트 적용, `tf.data.Dataset` 반복, 메트릭 계산, 서버에 집계 업데이트 적용 등)를 구동하기 위해 Federated Learning 프레임워크에 의해 구성된 추가 TensorFlow 코드

- *클라이언트*와 *서버* 간의 통신에 대한 선언적 사양(일반적으로 클라이언트 기기의 다양한 *집계* 형식 및 서버에서 모든 클라이언트로의 *브로드캐스팅*) 및 분산 통신에서 TensorFlow 코드의 클라이언트-로컬 또는 서버-로컬 실행을 인터리브하는 방법

직렬화된 형식으로 표현된 *페더레이션 계산*은 Python과는 별개의 플랫폼 독립적인 내부 언어로 표현되지만, Federated Learning API를 사용하기 위해 이 표현의 세부 사항에 대해 걱정할 필요가 없습니다. 계산은 Python 코드에서 유형 `tff.Computation`의 객체로 표시되며 대부분의 경우 불투명한 Python `callable`로 처리할 수 ​​있습니다.

튜토리얼에서는 이러한 페더레이션 계산을 마치 정규 Python 함수인 것처럼 로컬에서 실행하도록 호출합니다. 그러나 TFF는 실행 환경의 측면 대부분과 관계없이 페더레이션 계산을 표현하도록 설계되어, 예를 들어 `Android`를 실행하는 기기 그룹 또는 데이터 센터의 클러스터에 잠재적으로 배포할 수 있습니다. 다시 말하자면, 이것의 주요 결과는 [직렬화](#serialization)에 대한 강력한 가정입니다. 특히, 아래 설명된 `build_...` 메서드 중 하나를 호출하면 계산이 완전히 직렬화됩니다.

#### 상태 모델링하기

TFF는 함수형 프로그래밍 환경이지만, 페더레이션 학습에서 관심 프로세스는 상태 저장입니다. 예를 들어, 여러 라운드의 페더레이션 모델 평균화를 포함하는 훈련 루프는 *상태 저장 프로세스*로 분류할 수 있는 예입니다. 이 프로세스에서, 라운드에서 라운드로 진화하는 상태는 훈련되고 있는 모델 매개변수의 세트 및 옵티마이저와 관련된 추가 상태(예를 들어, 운동량 벡터)를 포함한다.

TFF는 함수형이므로 상태 저장 프로세스는 현재 상태를 입력으로 받아들인 다음 업데이트된 상태를 출력으로 제공하는 계산으로 TFF에서 모델링됩니다. 상태 저장 프로세스를 완전히 정의하려면, 초기 상태의 출처를 지정해야 합니다. (그렇지 않으면 프로세스를 부트스트랩할 수 없습니다.) 이것은 도우미 클래스 `tff.templates.IterativeProcess`의 정의에서 캡처되며, 두 개의 속성 `initialize` 및 `next`가 각각 초기화 및 반복에 해당합니다.

### 사용 가능한 빌더

현재 TFF는 페더레이션 훈련 및 평가를 위한 페더레이션 계산을 생성하는 두 가지 빌더 함수를 제공합니다.

- `tff.learning.build_federated_averaging_process`는 *모델 함수*와 *클라이언트 옵티마이저*를 사용하여 상태 저장 `tff.templates.IterativeProcess`를 반환합니다.

- 평가는 상태 저장이 아니므로 `tff.learning.build_federated_evaluation`은 *모델 함수*를 사용하여 모델의 페더레이션 평가를 위한 단일 페더레이션 계산을 반환합니다.

## 데이터세트

### 설계상 가정

#### 클라이언트 선택

일반적인 페더레이션 학습 시나리오에서는 잠재적으로 수억 대의 클라이언트 기기 *모집단*이 있으며, 이 중 작은 부분만 활성화되어 언제든지 활성화되어 훈련에 사용할 수 있습니다(예를 들어, 데이터 통신 연결 네트워크가 아닌 전원에 연결되어 있거나 아니면 유휴 상태인 클라이언트로 제한될 수 있습니다). 일반적으로 훈련 또는 평가에 참여할 수 있는 클라이언트 세트는 개발자가 제어할 수 없습니다. 또한, 수백만 대의 클라이언트를 조정하는 것은 비현실적이므로 일반적인 훈련 또는 평가 라운드에는 사용 가능한 클라이언트의 일부만 포함되며 [무작위로 샘플링](https://arxiv.org/pdf/1902.01046.pdf)될 수 있습니다.

이것의 주요 결과는 페더레이션 계산은 설계상 정확한 참가자 세트에 대해 알지 못하는 방식으로 표현된다는 것입니다. 모든 처리는 익명의 추상 *클라이언트* 그룹에 대한 집계 연산으로 표현되며, 해당 그룹은 훈련의 라운드마다 다를 수 있습니다. 따라서 구체적인 참가자에 대한 계산의 실제 바인딩, 즉 계산에 입력되는 구체적인 데이터에 대한 실제 바인딩은 계산 자체 외부에서 모델링됩니다.

페더레이션 학습 코드의 실제 배포를 시뮬레이션하기 위해 일반적으로 다음과 같은 훈련 루프를 작성합니다.

```python
trainer = tff.learning.build_federated_averaging_process(...)
state = trainer.initialize()
federated_training_data = ...

def sample(federate_data):
  return ...

while True:
  data_for_this_round = sample(federated_training_data)
  state, metrics = trainer.next(state, data_for_this_round)
```

이를 용이하게 하기 위해 시뮬레이션에서 TFF를 사용할 때 페더레이션 데이터는 Python `list`로 허용되며, 참여 클라이언트 기기당 하나의 요소가 해당 기기의 로컬 `tf.data.Dataset`를 나타냅니다.

### 추상 인터페이스

시뮬레이션된 페더레이션 데이터세트 처리를 표준화하기 위해 TFF는 클라이언트 세트를 열거하고 특정 클라이언트의 데이터를 포함하는 `tf.data.Dataset`를 구성할 수 있는 추상 인터페이스인 `tff.simulation.datasets.ClientData`를 제공합니다. 이러한 `tf.data.Dataset`은 강제 실행 모드에서 생성된 페더레이션 계산에 입력으로 직접 공급될 수 있습니다.

클라이언트 ID에 액세스하는 기능은 시뮬레이션에 사용하기 위해 데이터세트에서만 제공되는 기능으로, 클라이언트의 특정 하위 세트에서 데이터에 대해 훈련하는 기능이 필요할 수 있습니다(예: 다양한 클라이언트 유형의 하루 동안의 가용성을 시뮬레이션합니다). 컴파일된 계산 및 기본 런타임에는 클라이언트 ID의 개념이 포함되지 *않습니다*. 특정 클라이언트 하위 세트의 데이터가 입력으로 선택되면(예: `tff.templates.IterativeProcess.next` 호출), 클라이언트 ID가 더 이상 표시되지 않습니다.

### 사용 가능한 데이터세트

We have dedicated the namespace `tff.simulation.datasets` for datasets that implement the `tff.simulation.datasets.ClientData` interface for use in simulations, and seeded it with datasets to support the [image classification](tutorials/federated_learning_for_image_classification.ipynb) and [text generation](tutorials/federated_learning_for_text_generation.ipynb) tutorials. We'd like to encourage you to contribute your own datasets to the platform.
