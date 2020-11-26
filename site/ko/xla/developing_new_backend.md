# XLA용 새 백엔드 개발하기

이 예비 가이드는 TensorFlow를 해당 하드웨어로 쉽게 다시 지정하여 효율적으로 사용하려는 얼리 어답터를 위한 것입니다. 이 가이드는 과정을 단계별로 안내하지 않으며 [LLVM](http://llvm.org), [Bazel](https://bazel.build/) 및 TensorFlow에 대한 지식이 있다고 가정합니다.

XLA는 TensorFlow 그래프를 실행할 목적으로 백엔드를 만들 때 새로운 아키텍처 또는 가속기가 구현할 수 있는 추상 인터페이스를 제공합니다. XLA를 대상으로 다시 지정하는 것은 새로운 하드웨어에 적합하게 기존의 모든 TensorFlow Op를 구현하는 것보다 훨씬 간단하고 확장성이 좋습니다.

대부분의 구현은 다음 시나리오 중 하나에 해당합니다.

1. 기존 [LLVM](http://llvm.org) 백엔드가 있거나 없는 상태에서, XLA에서 아직 공식적으로 지원되지 않는 기존 CPU 아키텍처
2. 기존 LLVM 백엔드가 있는 비 CPU 유사 하드웨어
3. 기존 LLVM 백엔드가 없는 비 CPU 유사 하드웨어

> 참고: LLVM 백엔드는 공식적으로 출시된 LLVM 백엔드 중 하나 또는 자체 개발한 사용자 정의 LLVM 백엔드를 의미할 수 있습니다.

## 시나리오 1: XLA에서 아직 공식적으로 지원되지 않는 기존 CPU 아키텍처

이 시나리오에서는 기존 [XLA CPU 백엔드](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/cpu/)를 살펴보는 것으로 시작합니다. XLA를 사용하면 LLVM을 통해 TensorFlow를 다른 CPU로 쉽게 다시 지정할 수 있습니다. CPU용 XLA 백엔드 사이의 주된 차이점은 LLVM에서 생성된 코드이기 때문입니다. Google은 x64 및 ARM64 아키텍처용 XLA를 테스트합니다.

하드웨어 공급업체가 해당 하드웨어용 LLVM 백엔드를 가지고 있는 경우, 백엔드를 XLA로 빌드된 LLVM과 연결하는 것은 간단합니다. JIT 모드에서 XLA CPU 백엔드는 호스트 CPU에 대한 코드를 내보냅니다. Ahead Of Time 컴파일의 경우, [`xla::AotCompilationOptions`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/compiler.h)가 LLVM 트리플을 제공하므로 대상 아키텍처를 구성할 수 있습니다.

기존 LLVM 백엔드가 없지만 다른 종류의 코드 생성기가 있는 경우, 대부분의 기존 CPU 백엔드를 재사용할 수 있습니다.

## 시나리오 2: 기존 LLVM 백엔드가 있는 비 CPU 유사 하드웨어

기존 [`xla::CPUCompiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/cpu/cpu_compiler.cc) 및 [`xla::GPUCompiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/gpu/nvptx_compiler.cc) 클래스는 이미 LLVM IR을 내보내므로 이들 클래스에서 새로운 [`xla::Compiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/compiler.h) 구현을 모델링할 수 있습니다. 하드웨어의 특성에 따라 LLVM IR 생성의 많은 부분을 변경해야 할 수 있지만 많은 코드를 기존 백엔드와 공유할 수 있습니다.

XLA의 [GPU 백엔드](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/gpu/)를 따르는 것이 좋습니다. GPU 백엔드는 비 CPU 유사 ISA를 대상으로 하므로, 코드 생성의 일부 측면은 GPU 도메인에 고유합니다. Hexagon(업스트림 LLVM 백엔드가 있음)과 같은 DSP 등 다른 종류의 하드웨어는 LLVM IR 내보내기 로직의 일부를 재사용할 수 있지만 다른 부분은 고유합니다.

## 시나리오 3: 기존 LLVM 백엔드가 없는 비 CPU 유사 하드웨어

LLVM을 사용할 수 없는 경우, 가장 좋은 방법은 원하는 하드웨어에 맞게 XLA용 새 백엔드를 구현하는 것입니다. 이 옵션에 가장 많은 노력이 필요합니다. 구현해야 하는 클래스는 다음과 같습니다.

- [`StreamExecutor`](https://www.tensorflow.org/code/tensorflow/stream_executor/stream_executor.h): 많은 기기의 경우, `StreamExecutor`의 모든 메서드가 필요하지는 않습니다. 자세한 내용은 기존 `StreamExecutor` 구현을 참조하세요.
- [`xla::Compiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/compiler.h): 이 클래스는 HLO 계산의 컴파일을 `xla::Executable`로 캡슐화합니다.
- [`xla::Executable`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/executable.h): 이 클래스는 플랫폼에서 컴파일된 계산을 시작하는 데 사용됩니다.
- [`xla::TransferManager`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/transfer_manager.h): 이 클래스를 사용하면 백엔드가 지정된 기기 메모리 핸들에서 XLA 리터럴 데이터를 구성하기 위한 플랫폼별 메커니즘을 제공할 수 있습니다. 즉, 호스트에서 기기로, 또는 그 반대로 데이터 전송을 캡슐화하는 데 도움이 됩니다.
