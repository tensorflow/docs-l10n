# TensorFlow Quantum

TensorFlow Quantum(TFQ)은 [양자 머신러닝](concepts.md)을 위한 Python 프레임워크입니다. 애플리케이션 프레임워크인 TFQ를 사용하면 양자 알고리즘 연구원과 ML 애플리케이션 연구원이 모두 TensorFlow 내에서 Google의 양자 컴퓨팅 프레임워크를 활용할 수 있습니다.

TensorFlow Quantum은 *양자 데이터* 및 *하이브리드 양자 고전 모델*을 빌드하는 데 중점을 둡니다. <a href="https://github.com/quantumlib/Cirq" class="external">Cirq</a>에서 설계된 양자 알고리즘과 로직을 TensorFlow와 인터리브하는 도구를 제공합니다. TensorFlow Quantum을 효과적으로 사용하려면 양자 컴퓨팅에 대한 기본적인 이해가 필요합니다.

TensorFlow Quantum을 시작하려면 [설치 가이드](install.md)를 참조하고 실행 가능한 [노트북 튜토리얼](./tutorials/hello_many_worlds.ipynb) 중 일부를 읽어보세요.

## 설계

TensorFlow Quantum은 TensorFlow를 양자 컴퓨팅 하드웨어와 통합하는 데 필요한 구성 요소를 구현합니다. 이를 위해 TensorFlow Quantum은 두 가지 기본 데이터 형식을 도입합니다.

- *양자 회로* — TensorFlow 내의 Cirq 정의 양자 회로를 나타냅니다. 다른 실수값 데이터 포인트의 배치와 유사하게 다양한 크기의 회로 배치를 생성합니다.
- *파울리 합계(Pauli sum)* — Cirq에 정의된 파울리 연산자의 텐서 곱의 선형 조합을 나타냅니다. 회로와 마찬가지로 다양한 크기의 연산자 배치를 만듭니다.

이러한 기본 형식으로 양자 회로를 나타내는 TensorFlow Quantum은 다음 연산을 제공합니다.

- 회로 배치의 출력 분포를 샘플링합니다.
- 회로 배치에서 파울리 합계 배치의 기대값을 계산합니다. TFQ는 역전파 호환 그래디언트 계산을 구현합니다.
- 회로 및 상태 배치를 시뮬레이션합니다. 양자 회로 전체에서 직접 모든 양자 상태 진폭을 검사하는 것은 실제 세계에서는 비효율적이지만, 상태 시뮬레이션은 연구원이 양자 회로가 상태를 거의 정확한 수준의 정밀도로 매핑하는 방법을 이해하는 데 도움이 될 수 있습니다.

[설계 가이드](design.md)에서 TensorFlow Quantum 구현에 대해 자세히 알아보세요.

## 문제 보고

<a href="https://github.com/tensorflow/quantum/issues" class="external">TensorFlow Quantum 문제 추적기</a>로 버그 또는 특성 요청을 보고하세요.
