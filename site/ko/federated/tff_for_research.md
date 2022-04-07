# Federated Learning 연구에 TFF 사용하기

<!-- Note that some section headings are used as deep links into the document.
     If you update those section headings, please make sure you also update
     any links to the section. -->

## 개요

TFF는 현실적인 프록시 데이터세트에 대한 페더레이션 계산을 시뮬레이션하여 federated learning(FL) 연구를 수행하기 위한 확장 가능하고 강력한 프레임워크입니다. 이 페이지에서는 연구 시뮬레이션과 관련된 주요 개념 및 구성 요소와 TFF에서 다양한 종류의 연구를 수행하기 위한 자세한 지침을 설명합니다.

## TFF의 일반적인 연구 코드 구조

TFF에서 구현된 연구 FL 시뮬레이션은 일반적으로 3가지 주요 논리 유형으로 구성됩니다.

1. Individual pieces of TensorFlow code, typically `tf.function`s, that encapsulate logic that runs in a single location (e.g., on clients or on a server). This code is typically written and tested without any `tff.*` references, and can be re-used outside of TFF. For example, the [client training loop in Federated Averaging](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py#L184-L222) is implemented at this level.

2. TensorFlow Federated orchestration logic, which binds together the individual `tf.function`s from 1. by wrapping them as `tff.tf_computation`s and then orchestrating them using abstractions like `tff.federated_broadcast` and `tff.federated_mean` inside a `tff.federated_computation`. See, for example, this [orchestration for Federated Averaging](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py#L112-L140).

3. An outer driver script that simulates the control logic of a production FL system, selecting simulated clients from a dataset and then executing federated computations defined in 2. on those clients. For example, [a Federated EMNIST experiment driver](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py).

## 페더레이션 학습 데이터세트

TensorFlow 페더레이션은 페더레이션 학습으로 해결할 수 있는 실제 문제의 특성을 대표하는 [여러 데이터세트를 호스팅](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets)합니다.

참고: 이들 데이터세트는 [ClientData API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/ClientData)에 문서화된 대로, Python 기반 ML 프레임워크에서 Numpy 배열로 사용할 수도 있습니다.

데이터세트에는 다음이 포함됩니다.

- [**StackOverflow**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data) 훈련 세트에 135,818,730개의 예제(문장)를 가진 342,477명의 고유 사용자가 있는 언어 모델링 또는 지도 학습 작업을 위한 현실적인 텍스트 데이터세트입니다.

- [**Federated EMNIST**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data) EMNIST 문자 및 숫자 데이터세트의 페더레이션 전처리입니다. 각 클라이언트는 다른 작성자에 해당합니다. 전체 훈련 세트에는 62개 레이블의 671,585개 예제를 가진 3400명의 사용자가 포함됩니다.

- [**Shakespeare**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/shakespeare/load_data) William Shakespeare의 전체 작품을 기반으로 한 더 작은 문자 수준의 텍스트 데이터세트입니다. 데이터세트는 715명의 사용자 (셰익스피어 연극의 캐릭터)로 구성되며, 각 예제는 주어진 연극에서 캐릭터가 말한 연속적인 라인 세트에 해당합니다.

- [**CIFAR-100**.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data) A federated partitioning of the CIFAR-100 dataset across 500 training clients and 100 test clients. Each client has 100 unique examples. The partitioning is done in a way to create more realistic heterogeneity between clients. For more details, see the [API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data).

- [**Google Landmark v2 데이터세트 데이터**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/gldv2/load_data) 세트는 다양한 세계 랜드마크의 사진으로 구성되며, 데이터의 연합 분할을 달성하기 위해 사진사가 이미지를 그룹화합니다. 233개의 클라이언트와 23080개의 이미지가 있는 더 작은 데이터 세트와 1262개의 클라이언트와 164172개의 이미지가 있는 더 큰 데이터 세트의 두 가지 유형의 데이터 세트를 사용할 수 있습니다.

- [**CelebA**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/celeba/load_data) 유명인 얼굴의 예(이미지 및 얼굴 속성) 데이터세트입니다. 연합 데이터 세트에는 클라이언트를 형성하기 위해 함께 그룹화되는 각 유명인의 예가 있습니다. 9343개의 클라이언트가 있으며 각각 최소 5개의 예제가 있습니다. 데이터 세트는 클라이언트 또는 예제별로 훈련 및 테스트 그룹으로 분할할 수 있습니다.

- [**iNaturalist**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/inaturalist/load_data) 데이터 세트는 다양한 종의 사진으로 구성됩니다. 데이터 세트에는 1,203종에 대한 120,300개의 이미지가 포함되어 있습니다. 7가지 유형의 데이터 세트를 사용할 수 있습니다. 그 중 하나는 사진 작가에 의해 그룹화되며 9257명의 클라이언트로 구성됩니다. 나머지 데이터 세트는 사진을 찍은 지리적 위치별로 그룹화됩니다. 이 6가지 유형의 데이터 세트는 11 - 3,606개의 클라이언트로 구성됩니다.

## 고성능 시뮬레이션

*FL 시뮬레이션* 의 실제 시간은 알고리즘을 평가하기 위한 관련 측정항목이 아니지만(시뮬레이션 하드웨어는 실제 FL 배포 환경을 나타내지 않기 때문에) FL 시뮬레이션을 빠르게 실행할 수 있는 것은 연구 생산성에 매우 중요합니다. 따라서 TFF는 고성능 단일 및 다중 시스템 런타임을 제공하는 데 많은 투자를 했습니다. 문서가 개발 중이지만 지금은 [TFF를 사용한 고성능 시뮬레이션](https://www.tensorflow.org/federated/tutorials/simulations) 자습서, [가속기를 사용한 TFF 시뮬레이션](https://www.tensorflow.org/federated/tutorials/simulations_with_accelerators) [지침 및 GCP에서 TFF를 사용한 시뮬레이션 설정](https://www.tensorflow.org/federated/gcp_setup) 지침을 참조하십시오. 고성능 TFF 런타임은 기본적으로 활성화되어 있습니다.

## 다양한 연구 분야를 위한 TFF

### 페더레이션 최적화 알고리즘

페더레이션 최적화 알고리즘에 대한 연구는 원하는 사용자 정의 수준에 따라 TFF에서 다양한 방식으로 수행할 수 있습니다.

A minimal stand-alone implementation of the [Federated Averaging](https://arxiv.org/abs/1602.05629) algorithm is provided [here](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg). The code includes [TF functions](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py) for local computation, [TFF computations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py) for orchestration, and a [driver script](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py) on the EMNIST dataset as an example. These files can easily be adapted for customized applciations and algorithmic changes following detailed instructions in the [README](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/README.md).

A more general implementation of Federated Averaging can be found [here](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/algorithms/fed_avg.py). This implementation allows for more sophisticated optimization techniques, including the use of different optimizers on both the server and client. Other federated learning algorithms, including federated k-means clustering, can be found [here](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/algorithms/).

### 압축 모델링 및 업데이트

TFF는 [tensor_encoding](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/internal/tensor_encoding) API를 사용하여 손실 압축 알고리즘을 활성화하여 서버와 클라이언트 간의 통신 비용을 줄입니다. [Federated Averaging 알고리즘](https://arxiv.org/abs/1812.07210)을 사용한 서버-클라이언트 및 클라이언트-서버 압축을 사용한 훈련의 예는 [이 실험](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/compression/run_experiment.py)을 참조하세요.

사용자 정의 압축 알고리즘을 구현하고 훈련 루프에 적용하기 위해 다음을 수행할 수 있습니다.

1. Implement a new compression algorithm as a subclass of [`EncodingStageInterface`](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/internal/tensor_encoding/core/encoding_stage.py#L75) or its more general variant, [`AdaptiveEncodingStageInterface`](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/internal/tensor_encoding/core/encoding_stage.py#L274) following [this example](https://github.com/google-research/federated/blob/master/compression/sparsity.py).
2. Construct your new [`Encoder`](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/internal/tensor_encoding/core/core_encoder.py#L38) and specialize it for [model broadcast](https://github.com/google-research/federated/blob/master/compression/run_experiment.py#L118) or [model update averaging](https://github.com/google-research/federated/blob/master/compression/run_experiment.py#L144).
3. Use those objects to build the entire [training computation](https://github.com/google-research/federated/blob/master/compression/run_experiment.py#L247).

### 차등 프라이버시

TFF는 [TensorFlow Privacy](https://github.com/tensorflow/privacy) 라이브러리와 상호 운용이 가능하며, 차등 프라이버시를 사용하는 모델의 페더레이션 훈련을 위한 새로운 알고리즘을 연구할 수 있습니다. [기본 DP-FedAvg 알고리즘](https://arxiv.org/abs/1710.06963) 및 [확장](https://arxiv.org/abs/1812.06210)을 사용하는 DP 훈련의 예는 [이 실험 드라이버](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/differential_privacy/stackoverflow/run_federated.py)을 참조하세요.

사용자 지정 DP 알고리즘을 구현하고 연합 평균의 집계 업데이트에 적용하려면 [`tensorflow_privacy.DPQuery`](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/dp_query/dp_query.py#L54) 의 하위 클래스로 새 DP 평균 알고리즘을 구현하고 쿼리 인스턴스로 `tff.aggregators.DifferentiallyPrivateFactory` 를 만들 수 있습니다. [DP-FTRL 알고리즘](https://arxiv.org/abs/2103.00039) 을 구현하는 예는 여기에서 찾을 수 [있습니다.](https://github.com/google-research/federated/blob/master/dp_ftrl/dp_fedavg.py)

Federated GAN([아래에서](#generative_adversarial_networks) 설명)은 사용자 레벨의 차등 프라이버시를 구현하는 TFF 프로젝트의 또 다른 예입니다(예: [코드에서 여기](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/gans/tff_gans.py#L293)).

### 견고함과 공격

TFF는 페더레이션 학습 시스템에 대한 표적 공격과 *[Can You Really Back door Federated Learning?](https://arxiv.org/abs/1911.07963)*에서 고려한 차등 프라이버시 기반 방어를 시뮬레이션하는 데 사용할 수 있습니다. 이는 잠재적으로 악의적인 클라이언트로 반복적인 프로세스를 빌드하여 수행됩니다([`build_federated_averaging_process_attacked`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/research/targeted_attack/attacked_fedavg.py#L412) 참조). [target_attack](https://github.com/tensorflow/federated/tree/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/research/targeted_attack) 디렉토리에 자세한 내용이 포함되어 있습니다.

- New attacking algorithms can be implemented by writing a client update function which is a Tensorflow function, see [`ClientProjectBoost`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack/attacked_fedavg.py#L460) for an example.
- New defenses can be implemented by customizing ['tff.utils.StatefulAggregateFn'](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/core/utils/computation_utils.py#L103) which aggregates client outputs to get a global update.

시뮬레이션을 위한 예제 스크립트는 [`emnist_with_targeted_attack.py`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/research/targeted_attack/emnist_with_targeted_attack.py)를 참조하세요.

### 대립적인 생성 네트워크

GAN은 표준 Federated Averaging과 약간 다른 흥미로운 [페더레이션 오케스트레이션 패턴](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/gans/tff_gans.py#L266-L316)을 제공합니다. 여기에는 각각 고유한 최적화 단계로 훈련된 두 개의 개별 네트워크(생성기와 판별기)가 포함됩니다.

TFF can be used for research on federated training of GANs. For example, the DP-FedAvg-GAN algorithm presented in [recent work](https://arxiv.org/abs/1911.06679) is [implemented in TFF](https://github.com/tensorflow/federated/tree/main/federated_research/gans). This work demonstrates the effectiveness of combining federated learning, generative models, and [differential privacy](#differential_privacy).

### 개인화

페더레이션 학습 환경에서 개인화는 활발한 연구 분야입니다. 개인화의 목표는 다른 사용자에게 다른 추론 모델을 제공하는 것입니다. 이 문제에 대한 잠재적으로 다양한 접근 방식이 있습니다.

One approach is to let each client fine-tune a single global model (trained using federated learning) with their local data. This approach has connections to meta-learning, see, e.g., [this paper](https://arxiv.org/abs/1909.12488). An example of this approach is given in [`emnist_p13n_main.py`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/personalization/emnist_p13n_main.py). To explore and compare different personalization strategies, you can:

- Define a personalization strategy by implementing a `tf.function` that starts from an initial model, trains and evaluates a personalized model using each client's local datasets. An example is given by [`build_personalize_fn`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/personalization/p13n_utils.py).

- 전략 이름을 해당 개인화 전략에 매핑하는 `OrderedDict`를 정의하고 [`tff.learning.build_personalization_eval`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/build_personalization_eval)에서 `personalize_fn_dict` 인수로 사용합니다.

Another approach is to avoid training a fully global model by training part of a model entirely locally. An instantiation of this approach is described in [this blog post](https://ai.googleblog.com/2021/12/a-scalable-approach-for-partially-local.html). This approach is also connected to meta learning, see [this paper](https://arxiv.org/abs/2102.03448). To explore partially local federated learning, you can:

- Check out the [tutorial](https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization) for a complete code example applying Federated Reconstruction and [follow-up exercises](https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization#further_explorations).

- Create a partially local training process using [`tff.learning.reconstruction.build_training_process`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/reconstruction/build_training_process), modifying `dataset_split_fn` to customize process behavior.
