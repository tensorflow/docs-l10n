# XLA 아키텍처

<div style="width:50%; margin:auto; margin-bottom:10px; margin-top:20px;"> <img style="width:50%" src="./images/xlalogo.png"> </div>

## XLA를 빌드한 이유는 무엇입니까?

XLA가 TensorFlow와 동작하도록 하기 위한 몇 가지 목표가 있었습니다.

- *실행 속도 향상.* 서브그래프를 컴파일하여 수명이 짧은 연산의 실행 시간을 줄여 TensorFlow 런타임 오버헤드를 제거하고, 파이프라인 연산을 융합하여 메모리 오버헤드를 줄이며, 알려진 텐서 형상을 특화하여 더 적극적인 상수 전파를 허용합니다.

- *메모리 사용량 개선.* 원론적으로 많은 중간 저장소 버퍼를 제거하여 메모리 사용량을 분석하고 스케쥴을 짭니다.

- *사용자 정의 연산 의존도 감소.* 수동으로 융합된 사용자 정의 연산의 성능과 견줄 수 있게 자동으로 융합된 저수준 연산의 성능을 개선하여 많은 사용자 정의 연산의 필요성을 제거합니다.

- *모바일 풋프린트 감축.* 서브그래프를 미리 컴파일하고 다른 애플리케이션에 직접 연결될 수 있는 객체/헤더 파일 쌍을 내보내어 TensorFlow 런타임을 제거합니다. 그 결과 모바일 추론을 위한 풋프린트를 몇 배나 줄일 수 있습니다.

- *이식성 향상.* 새로운 하드웨어를 위한 새로운 백 엔드를 비교적 쉽게 작성할 수 있습니다. 이 부분에서, TensorFlow 프로그램의 상당 부분이 수정되지 않은 상태로 해당 하드웨어에서 실행됩니다. 이는 새로운 하드웨어를 위해 단일화된 개별적인 연산을 특화하는 접근 방식과 대조되는데,해당 연산을 사용하려면 TensorFlow 프로그램을 다시 작성해야 합니다.

## XLA는 어떻게 동작합니까?

XLA 입력 언어는 "HLO IR" 혹은 그저 HLO (고수준 연산)라고 불립니다. HLO의 의미 체계는 [연산 의미 체계](./operation_semantics.md) 페이지에 기술되어 있습니다. HLO를 [컴파일러 IR](https://en.wikipedia.org/wiki/Intermediate_representation)로 생각하는 것이 가장 편리합니다.

XLA는 HLO에 정의된 그래프('계산')를 사용하여 다양한 아키텍처를 위한 기계 명령어로 컴파일합니다. XLA는 [일부 새로운 HW 아키텍처를 겨냥](./developing_new_backend.md)하여 대체 백 엔드에 쉽게 끼워 넣을 수 있다는 점에서 모듈식입니다. x64용 및 ARM64용 CPU 백 엔드와 NVIDIA GPU 백 엔드는 TensorFlow 소스 트리에 있습니다.

다음 다이어그램은 XLA의 컴파일 프로세스를 보여줍니다:

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">   <img src="./images/how-does-xla-work.png"> </div>

XLA에는 [CSE](https://en.wikipedia.org/wiki/Common_subexpression_elimination), 대상 독립적 연산 융합 및 계산용 런타임 메모리 할당을 위한 버퍼 분석과 같이 대상 독립적인 여러 최적화 및 분석 과정이 제공됩니다.

대상 독립적 단계 후에 XLA는 HLO 계산을 백 엔드로 보냅니다. 백 엔드는 HLO 수준의 추가 최적화를 수행할 수 있으며, 이번에는 대상 특정 정보와 요구 사항을 염두에 두고 있습니다. 예를 들어, XLA GPU 백 엔드는 특히 GPU 프로그래밍 모델에 유익한 연산 융합을 수행하고 계산을 스트림으로 분할하는 방법을 결정할 수 있습니다. 이 단계에서 백 엔드는 최적화된 라이브러리 호출에 특정 연산 또는 그 조합의 패턴을 일치시킬 수도 있습니다.

다음 단계는 대상별 코드 생성입니다. XLA에 포함된 CPU 및 GPU 백 엔드는 저수준 IR, 최적화 및 코드 생성을 위해 [LLVM](http://llvm.org)을 사용합니다. 이러한 백 엔드는 XLA HLO 계산을 효율적으로 표현하는데 필요한 LLVM IR을 내보낸 다음 LLVM을 호출하여 이 LLVM IR에서 네이티브 코드를 내보냅니다.

GPU 백 엔드는 현재 LLVM NVPTX 백 엔드를 통해 NVIDIA GPU를 지원합니다. CPU 백 엔드는 여러 CPU ISA를 지원합니다.
