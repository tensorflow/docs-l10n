# TensorFlow 페더레이션 튜토리얼

이 [colab 기반](https://colab.research.google.com/) 튜토리얼은 실제 예제를 사용하여 주요 TFF 개념과 API를 안내합니다. 참조 문서는 [TFF 가이드](../get_started.md)에서 찾을 수 있습니다.

Note: TFF currently requires Python 3.9 or later, but [Google Colaboratory](https://research.google.com/colaboratory/)'s hosted runtimes currently use Python 3.7, and so in order to run these notebooks you will need to use a [custom local runtime](https://research.google.com/colaboratory/local-runtimes.html).

**페더레이션 학습으로 시작하기**

- [이미지 분류를 위한 페데레이션 학습](federated_learning_for_image_classification.ipynb)은 페데레이션 학습(FL) API의 주요 부분을 소개하고 TFF를 사용하여 페데레이션 MNIST와 유사한 데이터에 대한 페데레이션 학습을 시뮬레이션하는 방법을 보여줍니다.
- [텍스트 생성을 위한 페더레이션 학습은](federated_learning_for_text_generation.ipynb) TFF의 FL API를 사용하여 언어 모델링 작업을 위해 직렬화된 사전 훈련된 모델을 구체화하는 방법을 추가로 보여줍니다.
- [학습을 위한 권장 집계 조정](tuning_recommended_aggregators.ipynb)은 `tff.learning`의 기본 FL 계산을 견고성, 차등 개인 정보 보호, 압축 등을 제공하는 전문 집계 루틴과 결합할 수 있는 방법을 보여줍니다.
- [행렬 분해를 위한 페더레이션 재구성](federated_reconstruction_for_matrix_factorization.ipynb)은 일부 클라이언트 매개변수가 서버에서 집계되지 않는 로컬 페더레이션 학습을 부분적으로 소개합니다. 이 튜토리얼은 페더레이션 학습 API를 사용하여 부분적으로 로컬 행렬 분해 모델을 훈련하는 방법을 보여줍니다.

**페더레이션 분석으로 시작하기**

- [Private Heavy Hitters](private_heavy_hitters.ipynb)는 `tff.analytics.heavy_hitters`를 사용하여 페더레이션 분석 계산을 구축하여 개인 Heavy Hitters를 찾는 방법을 보여줍니다.

**사용자 정의 페더레이션 계산 작성**

- [나만의 페더레이션 학습 알고리즘 구축](building_your_own_federated_learning_algorithm.ipynb)은 TFF 핵심 API를 사용(예: 페더레이션 평균 사용)하여 페더레이션 학습 알고리즘을 구현하는 방법을 보여줍니다.
- [학습 알고리즘 구성](composing_learning_algorithms.ipynb)은 TFF 학습 API를 사용하여 새로운 페더레이션 학습 알고리즘, 특히 페더레이션 평균화의 변형을 쉽게 구현하는 방법을 보여줍니다.
- [TFF Optimizers를 사용한 사용자 정의 페더레이션 알고리즘](custom_federated_algorithm_with_tff_optimizers.ipynb)은 `tff.learning.optimizers`를 사용하여 페더레이션 평균화를 위한 사용자 지정 반복 프로세스를 구축하는 방법을 보여줍니다.
- [사용자 정의 페더레이션 알고리즘, 1부: 페더레이션  코어 소개](custom_federated_algorithms_1.ipynb) 및 [2부: 페더레이션 평균화 구현](custom_federated_algorithms_2.ipynb)은 Federated Core API(FC API)에서 제공하는 주요 개념과 인터페이스를 소개합니다.
- [사용자 지정 집계 구현](custom_aggregators.ipynb)은<code>tff.aggregators</code> 모듈의 이면에 있는 디자인 원칙과 클라이언트에서 서버로 값의 사용자 정의 집계를 구현하기 위한 모범 사례를 설명합니다.

**시뮬레이션 모범 사례**

- [High-performance simulations with Kubernetes](high_performance_simulation_with_kubernetes.ipynb) describes how to setup and configure a high-performance TFF runtime running on Kubernetes.

- [가속기(GPU)를 사용한 TFF 시뮬레이션](simulations_with_accelerators.ipynb)은 TFF의 고성능 런타임을 GPU와 함께 사용할 수 있는 방법을 보여줍니다.

- [ClientData로 작업](working_with_client_data.ipynb)은 TFF의 [ClientData](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/ClientData) 기반 시뮬레이션 데이터 세트를 TFF 계산에 통합하기 위한 모범 사례를 제공합니다.

**중급 및 고급 튜토리얼**

- [랜덤 노이즈 생성](random_noise_generation.ipynb)은 분산 계산에서 임의성을 사용하여 약간의 미묘함을 지적하고 모범 사례를 제안하며 패턴을 권장합니다.

- [tff.federated_select를 사용하여 특정 클라이언트에 다른 데이터 보내기](federated_select.ipynb)는 `tff.federated_select` 연산자를 소개하고 다른 클라이언트에 다른 데이터를 보내는 사용자 정의 연합 알고리즘의 간단한 예를 제공합니다.

- [federated_select 및 희소 집계를 통한 클라이언트 효율적인 대형 모델 페더레이션 학습](sparse_federated_learning.ipynb)은 각 클라이언트 장치가 `tff.federated_select` 및 희소 집계를 사용하여 모델의 작은 부분만 다운로드 및 업데이트하는 초대형 모델을 훈련하는 데 TFF를 사용할 수 있는 방법을 보여줍니다.

- [페더레이션 학습 연구를 위한 TFF: 모델 및 업데이트 압축](tff_for_federated_learning_research_compression.ipynb)은 [tensor_encoding API](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/internal/tensor_encoding)를 기반으로 하는 사용자 정의 집계를 TFF에서 사용할 수 있는 방법을 보여줍니다.

- [TFF의 차등 개인 정보 보호 페더레이션 학습](federated_learning_with_differential_privacy.ipynb)은 TFF를 사용하여 사용자 수준 차등 개인 정보 보호를 사용하여 모델을 훈련하는 방법을 보여줍니다.

- [Loading Remote Data with TFF](loading_remote_data.ipynb) describes how to embed custom logic in the TFF runtime to load data on remote machines.

- [Support for JAX in TFF](../tutorials/jax_support.ipynb) shows how [JAX](https://github.com/google/jax) computations can be used in TFF, demonstrating how TFF is designed to be able to interoperate with other frontend and backend ML frameworks.
