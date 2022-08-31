# 자주 묻는 질문

## TensorFlow Federated를 운영 환경(예: 휴대폰)에서 사용할 수 있습니까?

현재는 아닙니다. 실제 기기에 대한 배포를 염두에 두고 TFF를 설계했지만, 현재 단계에서는 그런 목적을 위한 도구를 제공하지 않습니다. 현재 릴리스는 포함된 시뮬레이션 런타임을 사용하여 새로운 페더레이션 알고리즘을 표현하거나 자신의 데이터세트로 페더레이션 학습을 시도하는 것과 같은 실험 용도로 사용됩니다.

TFF의 오픈 소스 에코시스템은 시간이 지남에 따라 물리적 배포 플랫폼을 대상으로 하는 런타임을 포함하는 방향으로 발전할 것으로 예상됩니다.

## 큰 데이터세트를 실험할 때 TFF를 어떻게 사용합니까?

TFF의 초기 릴리스에 포함된 기본 런타임은 (모든 시뮬레이션 클라이언트의) 모든 데이터가 동시에 단일 머신의 메모리에 저장하기 적합하고 전체 실험이 colab 노트북 내에서 로컬로 실행되는 소규모 실험(자습서에 설명되어 있음)에만 사용됩니다.

가까운 미래의 로드맵에는 매우 큰 데이터세트와 많은 수의 클라이언트의 실험을 위한 고성능 런타임이 포함됩니다.

## TFF의 임의성이 내 기대와 일치하도록 하려면 어떻게 해야 합니까?

TFF는 컴퓨팅을 핵심으로 페더레이션했으므로 TFF의 작성자는 TensorFlow `Session`이 어디에 어떻게 들어가는지, 그리고 `run`이 해당 세션 내에서 호출되는지를 제어할 수 있다고 가정해서는 안 됩니다. 임의성의 의미 체계는 시드가 설정된 경우 TensorFlow `Session`의 시작 및 종료에 따라 달라질 수 있습니다. TF 1.14의 `tf.random.experimental.Generator`를 사용하여 TensorFlow 2 스타일의 임의성을 사용하는 것이 좋습니다. `tf.Variable`을 사용하여 내부 상태를 관리합니다.

기대치를 관리하기 위해 TFF는 직렬화하는 TensorFlow가 그래프 레벨 시드가 아닌 op 레벨 시드를 갖도록 허용합니다. 이는 op 레벨 시드의 의미 체계가 TFF 설정에서 더 명확해야 하기 때문입니다. `tf_computation`으로 래핑된 함수를 호출할 때마다 결정론적 시퀀스가 생성되며 이 호출 내에서만 의사 난수 생성기에 의해 보장이 유지됩니다. 이것은 즉시 모드에서 `tf.function`을 호출하는 의미 체계와는 다릅니다. TFF는 `tf_computation`이 호출될 때마다 고유한 `tf.Session`을 효과적으로 입력 및 종료하지만, 즉시 모드에서 함수를 반복적으로 호출하는 것은 같은 세션 내 출력 텐서에서 `sess.run`을 반복적으로 호출하는 것과 유사합니다.

## 어떻게 기여할 수 있습니까?

[README](https://github.com/tensorflow/federated/blob/main/README.md), [기여](https://github.com/tensorflow/federated/blob/main/CONTRIBUTING.md) 가이드라인 및 [공동 작업](collaborations/README.md)을 참조하세요.

## FedJAX와 TensorFlow Federated의 관계는 무엇입니까?

TensorFlow Federated(TFF)는 다양한 알고리즘과 기능을 쉽게 구성하고 다양한 시뮬레이션 및 배포 시나리오에서 코드를 이식할 수 있도록 설계된 통합 학습 및 분석을 위한 본격적인 프레임워크입니다. TFF는 확장 가능한 런타임을 제공하고 표준 API를 통해 많은 개인 정보 보호, 압축 및 최적화 알고리즘을 지원합니다. TFF는 또한 [google-research repo](https://github.com/google-research/federated) 에 게시된 Google 논문의 예 모음과 함께 [다양한 유형의 FL 연구를](https://www.tensorflow.org/federated/tff_for_research) 지원합니다.

대조적으로, [FedJAX](https://github.com/google/fedjax) 는 연구 목적을 위한 연합 학습 알고리즘의 사용 용이성과 신속한 프로토타이핑에 중점을 둔 경량 Python 및 JAX 기반 시뮬레이션 라이브러리입니다. TensorFlow Federated 및 FedJAX는 코드 이식성에 대한 기대 없이 별도의 프로젝트로 개발됩니다.
