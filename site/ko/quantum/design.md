# TensorFlow 양자 설계

TensorFlow Quantum(TFQ)은 NISQ 시대의 양자 머신러닝 문제를 위해 설계되었습니다. TFQ는 양자 회로 빌드와 같은 양자 컴퓨팅 기본 형식을 TensorFlow 에코시스템에 제공합니다. TensorFlow로 빌드된 모델 및 연산은 이러한 기본 요소를 사용하여 강력한 양자 고전 하이브리드 시스템을 만듭니다.

TFQ에서는 양자 데이터세트, 양자 모델 및 고전적인 제어 매개변수를 사용하여 TensorFlow 그래프를 구성할 수 있습니다. 이들은 모두 단일 계산 그래프에서 텐서로 표시됩니다. 고전적인 확률 이벤트로 이어지는 양자 측정의 결과는 TensorFlow 연산에서 얻습니다. 훈련은 표준 [Keras](https://www.tensorflow.org/guide/keras/overview) API로 수행됩니다. `tfq.datasets` 모듈을 통해 새롭고 흥미로운 양자 데이터세트를 실험할 수 있습니다.

## Cirq

<a href="https://github.com/quantumlib/Cirq" class="external">Cirq</a>는 Google의 양자 프로그래밍 프레임워크입니다. 양자 컴퓨터 또는 시뮬레이션된 양자 컴퓨터에서 양자 회로를 생성, 수정 및 호출하기 위해 큐비트(qubit), 게이트, 회로 및 측정과 같은 모든 기본 연산을 제공합니다. TensorFlow Quantum은 이러한 Cirq 프리미티브를 사용하여 배치 계산, 모델 빌드 및 그래디언트 계산을 위해 TensorFlow를 확장합니다. TensorFlow Quantum을 효과적으로 사용하려면 Cirq를 사용하는 것이 좋습니다.

## TensorFlow Quantum 프리미티브

TensorFlow Quantum은 TensorFlow를 양자 컴퓨팅 하드웨어와 통합하는 데 필요한 구성 요소를 구현합니다. 이를 위해 TFQ에서는 두 가지 데이터 유형 프리미티브를 도입합니다.

- *양자 회로*: TensorFlow 내의 <a href="https://github.com/quantumlib/Cirq" class="external">Cirq</a> 정의 양자 회로(`cirq.Circuit`)를 나타냅니다. 다른 실수값 데이터 포인트의 배치와 유사하게 다양한 크기의 회로 배치를 생성합니다.
- *파울리 합계(Pauli sum)*: Cirq(`cirq.PauliSum`)에 정의된 Pauli 연산자의 텐서 곱의 선형 조합을 나타냅니다. 회로와 마찬가지로, 다양한 크기의 연산자 배치를 만듭니다.

### 기본 연산

`tf.Tensor` 내의 양자 회로 프리미티브를 사용하여 TensorFlow Quantum은 이들 회로를 처리하고 의미 있는 출력을 생성하는 연산을 구현합니다.

TensorFlow 연산은 최적화된 C++로 작성되었습니다. 이러한 연산은 회로에서 샘플링하고, 기대값을 계산하고, 주어진 회로에서 생성된 상태를 출력합니다. 유연하고 성능이 좋은 작성 연산에는 몇 가지 문제가 있습니다.

1. 회로는 같은 크기가 아닙니다. 시뮬레이션된 회로의 경우 정적 연산(예: `tf.matmul` 또는 `tf.add`)을 만든 다음, 크기가 다른 회로를 다른 숫자로 대체할 수 없습니다. 정적 연산은 정적으로 크기가 지정된 TensorFlow 컴퓨팅 그래프에서 허용하지 않는 동적 크기를 허용해야 합니다.
2. 양자 데이터는 완전히 다른 회로 구조를 유도할 수 있습니다. 이는 TFQ ops에서 동적 크기를 지원하는 또 다른 이유입니다. 양자 데이터는 원래 회로에 대한 변경 사항으로 표시되는 기본 양자 상태의 구조적 변화를 나타낼 수 있습니다. 새로운 데이터 포인트가 런타임에 교체되기 때문에 TensorFlow 컴퓨팅 그래프는 빌드된 후에 수정할 수 없으므로 이러한 다양한 구조에 대한 지원이 필요합니다.
3. `cirq.Circuits`는 일련의 연산이라는 점에서 계산 그래프와 유사하며 일부는 기호/자리 표시자를 포함할 수 있습니다. 가능한 한 TensorFlow와 호환되도록 하는 것이 중요합니다.

성능상의 이유로 Eigen(많은 TensorFlow 연산에 사용되는 C++ 라이브러리)은 양자 회로 시뮬레이션에 적합하지 않습니다. 대신 <a href="https://ai.googleblog.com/2019/10/quantum-supremacy-using-programmable.html" class="external">양자 우월성 기존 실험</a>에 사용된 회로 시뮬레이터는 검증 도구로 사용되며 TFQ 연산의 기반으로 확장됩니다(모두 AVX2 및 SSE 명령어로 작성됨). 물리적 양자 컴퓨터를 사용하는 동일한 함수형 서명을 가진 ops가 만들어졌습니다. 시뮬레이션된 양자 컴퓨터와 물리적 양자 컴퓨터 간에 전환하는 것은 한 줄의 코드를 변경하는 것만큼 쉽습니다. 해당 연산은 <a href="https://github.com/tensorflow/quantum/blob/master/tensorflow_quantum/core/ops/circuit_execution_ops.py" class="external"><code>circuit_execution_ops.py</code></a>에 있습니다.

### 레이어

TensorFlow Quantum 레이어는 `tf.keras.layers.Layer` 인터페이스로 개발자에게 샘플링, 기대값 및 상태 계산을 나타냅니다. 고전적인 제어 매개변수 또는 판독 연산을 위한 회로 레이어를 만드는 것이 편리합니다. 또한 배치 회로, 배치 제어 매개변수 값을 지원하는 고도의 복잡성을 가진 레이어를 생성하고, 배치 판독 연산을 수행할 수 있습니다. 예제는 `tfq.layers.Sample`을 참조하세요.

### 미분기 요소

많은 TensorFlow 연산과 달리, 양자 회로의 관찰 가능 항목에는 상대적으로 계산하기 쉬운 그래디언트에 대한 공식이 없습니다. 이는 고전적인 컴퓨터가 양자 컴퓨터에서 실행되는 회로의 샘플만 읽을 수 있기 때문입니다.

위의 문제를 해결하기 위해 `tfq.differentiators` 모듈은 몇 가지 표준 차별화 기술을 제공합니다. 또한 사용자는 샘플에 기반한 기대값 계산의 '실제 세계' 설정과 정확한 분석 세계에서 그래디언트를 계산하는 자체 메서드를 정의할 수 있습니다. 유한 차분과 같은 메서드는 분석/정확한 환경에서 가장 빠른(벽시계 시간) 경우가 많습니다. 또한 더 느리지만(벽시계 시간) <a href="https://arxiv.org/abs/1811.11184" class="external">매개변수 이동</a>이나 <a href="https://arxiv.org/abs/1901.05374" class="external">확률적 메서드</a>와 같은 보다 실용적인 메서드가 종종 더 효과적입니다. `tfq.differentiators.Differentiator`는 인스턴스화되어 `generate_differentiable_op`로 기존 연산에 연결되거나, `tfq.layers.Expectation` 또는 `tfq.layers.SampledExpectation`의 생성자에 전달됩니다. 사용자 정의 차별화 요소를 구현하려면 `tfq.differentiators.Differentiator` 클래스에서 상속하세요. 샘플링 또는 상태 벡터 계산을 위한 그래디언트 연산을 정의하려면 `tf.custom_gradient`를 사용하세요.

### 데이터세트

양자 컴퓨팅 분야가 성장함에 따라, 더 많은 양자 데이터와 모델 조합이 발생하여 구조적 비교가 더 어려워집니다. `tfq.datasets` 모듈은 양자 머신 러닝 작업의 데이터 소스로 사용됩니다. 이 모듈은 모델 및 성능에 대한 구조화된 비교를 보장합니다.

대규모 커뮤니티의 기여를 통해 `tfq.datasets` 모듈로 더 투명하고 재현 가능한 연구가 가능할 수 있기를 바랍니다. 신중하게 선별된 문제인 양자 제어, 페르미온 시뮬레이션, 근접 상전이 분류, 양자 감지 등은 모두 `tfq.datasets`에 추가할 수 있는 좋은 후보입니다. 새 데이터세트를 제안하려면 <a href="https://github.com/tensorflow/quantum/issues">GitHub 문제</a>를 여세요.
