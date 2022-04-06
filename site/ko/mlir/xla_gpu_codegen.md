# XLA용 MLIR CodeGen

<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'timshen' reviewed: '2020-06-16' }
*-->

XLA는 `HloInstruction`에서 동작하며 이 표현에 대해 많은 최적화를 수행하여 대상 기기 간에 많은 것을 공유합니다. 어떤 시점에서 선형 일정이 계산되고, 메모리 버퍼가 각 값에 정적으로 할당됩니다. 기기별 codegen는 이 시퀀스를 순회하고 "emitter"를 호출하여 기기에 적합한 표현을 생성하는 방식으로 동작합니다(예를 들어, CPU에서 XLA 계산당 단일 LLVM 함수 또는 GPU 연산을 캡슐화하는 "thunk"의 시퀀스 및 GPU를 대상으로 할 때 생성된 PTX).

준비 단계로서 현재 XLA가 버퍼 할당 단계를 완료한 직후 프로세스를 가로채고 `lhlo` 언어로 MLIR 모듈을 대신 내보내는 단계에 있습니다. 기기에 따라 MLIR 구성 요소(주로 Linalg, affine 및 GPU 언어)를 사용하여 codegen을 수행합니다.

다음은 `lhlo`를 codegen 입력으로 사용하여 XLA/GPU를 점진적으로 마이그레이션하는 기록의 계획입니다.

## 과제

 | 호스트 | 장치
--- | --- | ---
입력 형식 | HloInstruction*(작업 1) | HloInstruction*(작업 1)
출력 형식 | xla::Thunk(작업 2) | LLVM IR(작업 3)

- **작업 1**은 호스트 및 기기 입력 형식을 HloInstruction*에서 LHLO로 변경합니다.
- **작업 2**는 호스트의 출력 형식을 thunk에서 "호스트용 랜딩 패드"로 변경합니다(아래 참조).
- **작업 3**은 LLVM IR의 기기 출력을 MLIR의 일부 형식으로 마이그레이션합니다. 이 프로젝트의 선택 사항이며, 자세한 내용은 "기기 LLVM IR 마이그레이션하기" 섹션을 참조하세요.

이 프로젝트는 가능한 한 LHLO emitter가 활성화된 엔드 투 엔드 실행 가능 모델을 우선적으로 사용합니다. 이는 우선 순위에 따른 목표의 목록을 의미합니다.

- LHLO emitter로 XLA/GPU를 실행 가능하게 만들고 기존 Thunk 및 emitter는 수정하지 않습니다.
- 경우에 따라 LHLO에서 HloInstruction*에 대한 참조를 제거합니다.
    - 레거시 emitter를 MLIR 기반 emitter(예: Linalg)로 전환하거나
    - 기존 emitter를 기계적으로 변환하여 MLIR 표현을 사용합니다(GPU Dialect를 사용하여 Standard로 마이그레이션합니다).

## Thunk 마이그레이션하기(작업 2)

xla::gpu::Thunk의 데이터 구조는 다음과 같습니다.

- 호스트(xla::gpu::Thunk::ExecuteOnStream())에서 호출할 수 있습니다.
- 하위 클래스에 다양한 데이터를 전달합니다.
- BufferAllocation::Slice 및 StreamExecutor와 상호 작용합니다.
- 커널을 시작합니다.
- 모든 런타임 라이브러리를 호출합니다.

비용에는 다음이 포함됩니다.

- 연산 관련 구성 데이터(예: 컨볼루션 구성)를 나타냅니다.
- op 형상 및 피연산자 형상을 마이그레이션합니다.
- thunk 트리 (while, condition 등)를 나타냅니다.

마이그레이션 작업은 LHLO/emitter 마이그레이션과 독립적입니다. 제한된 리소스에서 LHLO/emitter 마이그레이션 뒤에 우선 순위가 지정됩니다.

LHLO에서 호스트측 부분을 낮추는 방법에 대한 몇 가지 선택 사항이 있습니다.

- TFRT
    - (장점) 사용을 위한 훌륭한 CUDA 및 HIP 래퍼
    - (장점) TFRT 연산은 C++ 코드로 해석되므로 라이브러리 호출(cuDNN, cuBLAS, cuFFT 등)을 쉽게 구현할 수 있습니다.
    - (단점) 호스트측은 개발 중이며 테스트되지 않았습니다.
- JIT 컴파일된 CPU 코드
    - (장점) 매우 낮은 기능, 몇 개의 루프와 조건을 생성하면 완료됩니다.
    - (단점) GPUDialect는 아직 체인/스트림/비동기성/기기 할당을 모델링하지 않습니다.
    - (단점) CUDA/HIP 런타임 지원이 최소화됩니다(툴킷 경로, 버전, 동적 로딩 등).
- 기존 (해석) XLA 런타임

결정: TFRT를 채택하지만, TFRT에서 CPU 코드의 JIT 컴파일도 지원합니다.

## 기기 LLVM IR 마이그레이션하기(작업 3)

요소 emitter는 요소별로 채워서 대상 op를 생성합니다. 각 출력 요소는 피연산자의 요소 집합에 따라 다릅니다. 모든 요소는 버퍼를 동적 인덱스와 결합하여 설명됩니다. 거의 모든 "수학" ops를 설명하는 것으로 충분하지만, 성능상의 이유로 "수학" ops의 큰 하위 집합만 (Cpu|Gpu) ElementalIrEmitter에서 직접 구현됩니다.

ElementalIrEmitter는 다음과 같은 점에서 고유합니다.

- 코드의 상당 부분은 XLA/GPU와 CPU 간에 공유됩니다.
- 모든 요소별 ops를 포함하여 모델에서 볼 수 있는 ops의 상당 부분을 나타냅니다.
- 대부분의 융합은 ElementalIrEmitter에만 의존합니다.
- op 요소와 피연산자 요소 간의 데이터 종속성 DAG를 설명하므로 구조적으로 간단합니다.
- 대부분 이식 가능하고 높은 수준입니다(예: GPU kReduce 및 GPU kCopy와 달리).
- 최소한 요소별 ops에 대해 동적 형상 지원이 쉽습니다.

이제 요소 내보내기 여부와 관계없이 모든 ops에 대해 각 XLA op의 최종 상태에 대한 몇 가지 특징이 있습니다.

1. 기기 코드는 LLVM IR로 유지됩니다.
2. 이전 emitter를 LHLO -&gt; MLIR LLVM Dialect처럼 리팩터링합니다.
    - (비용) 궁극적으로 Standard로 마이그레이션하려는 경우, 삭제 작업이 됩니다.
    - (혜택) 쉽고 기계적입니다. 단기간에 할 수 있습니다.
    - (혜택) (1)에 비해 더 많은 혜택이 없습니다.
3. 이전 emitter를 LHLO -&gt; MLIR GPU + Standard + 루프처럼 리팩터링합니다.
    - (비용) 기존 emitter를 Standard로 올리면 몇 가지 문제가 발생합니다. 포인터와 GEP는 MemRef 및 SubView로 변환해야 합니다. amdgpu 완전성을 보장하는 것은 또 다른 하나입니다.
    - (비용) XLA/GPU는 LLVM 메타데이터에 크게 의존합니다.
        - 블록/스레드 인덱스의 `range`
        - 로드/저장을 위한 `align`, `dereferenceable`, `invariant.load`, `alias.scope`,`noalias`
        - 순차 루프를 위한 `llvm.loop.unroll.disable`, `llvm.loop.unroll.full`, `llvm.loop.vectorize.enable`
    - (혜택) 장기적일 수 있습니다. 이식성이 높습니다.
4. 이전 emitter를 LHLO -&gt; Linalg로 리팩터링하고 새 Linalg emitter 작성합니다.
    - (비용) 경우에 따라 다릅니다. 이전 옵션과 비교하여 XLA의 성능과 일치하는 새로운 구현은 벤치마크 &lt;-&gt; 최적화 워크플로를 거쳐야 하며, 이는 일부 ops에서 상당한 비용이 될 수 있습니다.
    - (혜택) 통합 스택, 지역 사회 지원, 이식성, 더 많은 최적화 가능성

결론:

- (2)는 권장하지 않습니다. (1) 또는 (3)이 (2)보다 낫습니다. (2)는 기계적 리팩터링이 많이 필요하기 때문에 (1)보다 비용이 많이 듭니다. (1)을 사용하면 XLA에서 MLIR emitter를 선택할 수 있다는 목표를 달성할 수 있습니다. LHLO -&gt; LLVM IR -&gt; 레거시 기기 emitter를 실행합니다.
- ElementalIrEmitter ops는 (4)로 진행되지만, 점진적으로는 아닙니다. 모든 요소 내보내기 ops가 같은 그래프에 연결되어 있으므로 연산별로 수행할 수 있는 방법이 없습니다. 이 작업은 또한 여러 진행 중인 힘(xla/service/mlir_gpu, kernel generator, Linalg)의 통합 지점 역할을 할 수 있습니다.
- 다른 모든 ops는 (1)로 진행합니다. 확장 목표로 (3) 또는 (4)로 마이그레이션할 수 있습니다.

## 우선 순위

위에서 언급한 3가지 작업은 모두 병렬화할 수 있지만, 제한된 리소스에서 직렬화해야 합니다. 우선 순위는 각 작업 완료에 대한 가시적인 결과에 중점을 둡니다.

우선 순위는 작업 1(레거시 emitter의 경우 LHLO) &gt; 작업 2(Thunks) &gt; 작업 3(MLIR emitter)입니다.

작업 1이 끝날 때까지 XLA 사용자는 LHLO(예: 커널 생성기)를 생성하고 실행할 수 있습니다. 컴파일 형식은 직렬화 가능한 MLIR이 아닙니다.

작업 2가 끝날 때까지 LHLO는 직렬화 가능한 적절한 MLIR로 수준을 낮춥니다. 이렇게 하면 오프라인 컴파일이 가능합니다.

작업 3이 끝날 때까지 모든 XLA emitter는 구현에서 MLIR 기반입니다.

## 세부 설계

### 1단계: (작업 1) LHLO 완료 및 레거시 emitter에서 LHLO 사용

이 단계에서는 기존의 모든 XLA/GPU emitter가 MLIR ops와 상호 작용합니다. 이 단계는 순수 리팩터링과 NFC입니다.

이 단계는 대부분 기계적이지만, 중첩되지 않은 HloComputation과 LHLO 간에 다음과 같은 불일치가 있음을 알아두는 것이 좋습니다.

- 각 HloInstruction은 피연산자(데이터 흐름 DAG)에 직접 액세스할 수 있습니다. 반대로 각 LHLO op는 피연산자 버퍼(ops과 버퍼 간의 이분)에만 액세스할 수 있습니다. LHLO ops는 피연산자 ops에 액세스하기 위해 use-def 체인을 거쳐야 합니다.
- 중첩되지 않은 레거시 emitter는 경험적으로 거의 피연산자에 액세스하지 않습니다. 유일한 예외는 kReduce입니다.
- 중첩되지 않은 레거시 emitter는 슬라이스를 얻기 위해서만 BufferAssignment에 액세스하고 dataflow_analysis() 또는 alias_analysis()와 같은 보조 데이터 구조에 액세스하기 위한 것이 아닙니다. llvm_ir는 슬라이스 정보를 기반으로 자체 alias_analysis()를 빌드합니다.

결론은 LHLO가 큰 번거로움 없이 적합해야 한다는 것입니다.

### 2단계: (선택 사항) 프로파일링 지원

**이 단계는 일부 XLA Thunk 로직을 삭제하기 시작하는 경우에만 필요합니다(다음 단계 참조).**

실제로 MLIR 기반 emitter를 켜기 전에 MLIR 기반 emitter에 대한 프로파일링이 필요합니다.

현재 XLA는 StreamExecutor의 타이머를 호출하여 자체 프로파일링을 수행합니다. 내부의 타이머는 커널 시작 전후에 두 개의 이벤트를 삽입하고 이 두 이벤트 간의 동기화 시간을 측정합니다.

MLIR에서 프로파일링을 지원하는 방법은 대략 3가지입니다.

- 프로파일러 엔드 투 엔드 실행
- 삽입된 프로파일러를 사용하여 LHLO에서 각 op에 대한 프로필 op를 추가합니다.

"엔드 투 엔드" 접근 방식은 MLIR에 투명하지만, XLA에서 처음에 이를 사용하지 못하는 같은 문제가 있습니다. 프로파일러(nvprof/...)가 수집한 라이브러리 호출은 HLO ops와 쉽게 관련될 수 없습니다. 예를 들어, cuDNN은 각 HLO에 대해 여러 커널을 시작하며 어떤 커널이 어떤 HLO에 해당하는지 알기 어렵습니다.

"삽입된 프로파일러" 접근 방식에는 다음이 필요합니다.

- 프로파일러를 매개변수로 사용하는 LHLO
- 각 op 전후에 profile.start/profile.end 삽입
- profile.{start,end}에서 C++ 구현으로 수준을 낮추는 전달

MLIR에서 생성된 op에 대해서는 다음과 같은 이유로 정확한 프로파일링을 쉽게 수행할 수 없습니다.

- MLIR에는 타이머가 없으며 TFRT/StreamExecutor에 의존하지 않습니다.
- MLIR은 복잡한 매개변수를 사용하는 C 함수를 쉽게 호출하지 않습니다.

### 3단계: (작업 2) Thunk 마이그레이션

참고로, 대략 3가지 종류의 thunk가 있습니다.

- 커널을 시작하는 KernelThunk
- 제어 흐름 thunk: 호스트 제어 흐름 논리(조건부, while, for, 시퀀스) 및 시작 본문 커널이 있습니다.
- 라이브러리 thunk: cuDNN, cuBLAS, cuFFT, NCCL 등

계획은 다음과 같습니다.

- Thunk를 (역)직렬화 가능하게 만듭니다.
- 해당 의미 체계를 지원할 수 있는 상태로 TFRT를 개선하는 데 도움이됩니다.
- 상태가 개선됨에 따라 개별 thunk를 점진적으로 마이그레이션합니다.

작업 항목은 부분적으로만 주문됩니다. 실제 실행 순서/엔지니어링 병렬 처리는 진행되는 대로 평가됩니다.

### 4단계: (작업 3) 마이그레이션된 ElementalIrEmitter

프로파일링이 준비되면, MLIR에서 모든 ElementalIrEmitter 기반 emitter를 완료하고 조정할 수 있습니다. 그런 다음 모든 MLIR 기반 emitter가 단일 스트림을 사용한다고 가정하여 기본적으로 설정합니다.

XLA/CPU의 ElementalIrEmitter도 코드의 많은 부분을 공유하므로 마이그레이션하는 것이 좋습니다.

모든 벤치마킹 및 성능 헌팅이 완료되면(TODO: 성능 패리티 정의), 새로운 MLIR 기반 요소 emitter를 켜고 레거시 ElementalIrEmitter를 삭제합니다.

이 단계는 또한 이후 마이그레이션을 위해 쉬운 융합 전환(중첩된 ops)을 제공합니다.

### 5단계: 멀티 스트림 지원 또는 중단

MLIR에서 [일부 emitter](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/gpu/stream_assignment.cc#L140)를 지원하거나 기능을 중단할 때까지 삭제할 수 없습니다. MLIR에서는 상대적으로 많은 양의 작업이고, XLA에서는 약간의 이득입니다. 멀티 스트림 XLA/GPU 사용자의 현재 사용자를 조사하고 합당한 경우 이 기능을 삭제해야 합니다.

### 6단계: (작업 3) 마이그레이션된 기기 Ops

이 단계에서는 모든 중첩되지 않은 ops를 마이그레이션한 다음 중첩되지 않은 모든 emitter를 삭제할 수 있습니다.

kCopy 및 kReduce에 대한 재작성/리팩터링이 필요합니다. kReduce는 이미 많은 작업을 수행하고 있으므로 수행해야 할 실제 작업량은 아직 확인되지 않았습니다.
