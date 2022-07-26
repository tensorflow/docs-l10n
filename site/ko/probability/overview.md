# TensorFlow Probability

TensorFlow Probability는 TensorFlow에서 확률적 추론 및 통계 분석을 수행하기 위한 라이브러리입니다. TensorFlow 에코시스템의 일부인 TensorFlow Probability는 확률적 방법과 심층 네트워크의 통합, 자동 미분을 사용한 그래디언트 기반 추론, 하드웨어 가속(GPU) 및 분산 계산을 통한 대규모 데이터세트 및 모델로의 확장성을 제공합니다.

TensorFlow Probability를 시작하려면 [설치 가이드](./install)를 참조하고 [Python 노트북 튜토리얼](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/){:.external}를 살펴보세요.

## 구성 요소

확률적 머신러닝 도구는 다음과 같이 구성됩니다.

### 레이어 0: TensorFlow

*수치 연산* - 특히, `LinearOperator` 클래스를 이용하면 특정 구조(대각선, 낮은 순위 등)를 활용할 수 있는 행렬 없는 구현이 가능하여 계산 효율이 개선됩니다. 이러한 연산은 TensorFlow Probability 팀에서 빌드 및 유지관리하며 핵심 TensorFlow의 [`tf.linalg`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/ops/linalg)에 포함되어 있습니다.

### 레이어 1: 통계 구성 요소

- *분포*([`tfp.distributions`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/distributions)): 배치 및 [브로드캐스팅](https://docs.scipy.org/doc/numpy-1.14.0/user/basics.broadcasting.html){:.external} 의미 체계를 사용하는 대규모 확률 분포 및 관련 통계 모음입니다.
- *Bijector*([`tfp.bijectors`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/bijectors)): 확률 변수의 가역적이고 구성 가능한 변환입니다. Bijector는 [로그 정규 분포](https://en.wikipedia.org/wiki/Log-normal_distribution){:.external}와 같은 전형적인 예부터 [마스킹된 자기 회귀 흐름](https://arxiv.org/abs/1705.07057){:.external}과 같은 정교한 딥 러닝 모델에 이르기까지 다양한 종류의 변환된 분포를 제공합니다.

### 레이어 2: 모델 구축

- 결합 분포(예: [`tfp.distributions.JointDistributionSequential`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/distributions/joint_distribution_sequential.py)): 하나 이상의 상호 의존적 분포에 대한 결합 분포입니다. TFP의 `JointDistribution`을 사용한 모델링에 대한 소개는 [이 colab](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/Modeling_with_JointDistribution.ipynb)을 확인하세요.
- *확률적 레이어*([`tfp.layers`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/layers)): 레이어가 나타내는 전체 함수에 걸쳐 불확실성을 갖진 신경망 레이어로, TensorFlow 레이어를 확장합니다.

### 레이어 3: 확률적 추론

- *Markov 체인 Monte Carlo*([`tfp.mcmc`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/mcmc)): 샘플링을 통해 적분을 근사 계산하는 알고리즘입니다. [Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo){:.external}, 랜덤 워크 Metropolis-Hastings 및 사용자 정의 전환 커널을 빌드하는 기능이 포함됩니다.
- *변량 추론*([`tfp.vi`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/vi)): 최적화를 통해 적분을 근사 계산하는 알고리즘입니다.
- *옵티마이저*([`tfp.optimizer`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/optimizer)): 확률적 최적화 방법으로, TensorFlow 옵티마이저를 확장합니다. [확률적 그래디언트 Langevin Dynamics](http://www.icml-2011.org/papers/398_icmlpaper.pdf){:.external}를 포함합니다.
- *Monte Carlo*([`tfp.monte_carlo`](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/monte_carlo)): Monte Carlo 기대치를 계산하기 위한 도구입니다.

TensorFlow Probability는 현재 개발 중이며 인터페이스가 달라질 수 있습니다.

## 예

탐색에 나열된 [Python 노트북 튜토리얼](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/){:.external} 외에도 사용 가능한 몇 가지 예제 스크립트가 있습니다.

- [Variational Autoencoders](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/examples/vae.py) — 잠재 코드 및 변량 추론을 통한 표현 학습입니다.
- [Vector-Quantized Autoencoder](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/examples/vq_vae.py) — 벡터 양자화를 통한 분산된 표현 학습입니다.
- [Bayesian Neural Networks](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/examples/bayesian_neural_network.py) — 가중치에 불확실성이 존재하는 신경망입니다.
- [Bayesian Logistic Regression](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/examples/logistic_regression.py) — 바이너리 분류를 위한 베이지안 추론입니다.

## 문제 보고하기

[TensorFlow Probability 문제 추적기](https://github.com/tensorflow/probability/issues)를 사용하여 버그 또는 기능 요청을 보고하세요.
