# 페더레이션 프로그램

이 문서는 페더레이션 프로그램 개념에 대한 개괄적인 개요에 관심이 있는 분들을 대상으로 작성된 문서입니다. 이 문서를 읽으려면 TensorFlow 페더레이션과 그 유형 시스템에 대한 지식이 있어야 합니다.

페더레이션 프로그램에 대한 자세한 내용은 다음을 참조하세요.

- [API 설명서](https://www.tensorflow.org/federated/api_docs/python/tff/program)
- [예제](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program)
- [개발자 가이드](https://github.com/tensorflow/federated/blob/main/docs/program/guide.md)

[목차]

## 페더레이션 프로그램이란 무엇인가요?

**페더레이션 프로그램**은 페더레이션(연합) 환경에서 계산 및 기타 처리 로직을 실행하는 프로그램입니다.

더 구체적으로 말하자면 **페더레이션 프로그램**은 다음과 같습니다.

- [계산](#computations)을 실행합니다.
- [프로그램 로직](#program-logic)을 사용합니다.
- [플랫폼별 구성 요소](#platform-specific-components)를 활용합니다.
- 그리고 [플랫폼에 구애받지 않는 구성 요소](#platform-agnostic-components)도 활용합니다.
- [프로그램](#program)으로 설정한 [매개변수](#parameters)를 제공받습니다.
- 그리고 [고객](#customer)이 설정한 [매개변수](#parameters)도 제공받습니다.
- 이때 조건은 [고객](#customer)이 [프로그램](#program)을 실행하는 것입니다.
- [플랫폼 저장소](#platform storage)에서 다음과 같은 방식으로 데이터를 [구체화](#materialize)할 수도 있습니다.
    - Python 로직을 사용하여
    - [내결함성](#fault tolerance)을 구현하여
- 그리고 [고객 저장소](#customer storage)로 데이터를 [릴리스](#release)할 수도 있습니다.

이러한 [개념](#concepts)과 추상화를 정의함으로써 페더레이션 프로그램의 [컴포넌트](#components) 간의 관계를 설명할 수 있으며, 이러한 컴포넌트를 서로 다른 [역할](#roles)로 소유 및 작성할 수 있습니다. 이러한 분리를 통해 개발자는 다른 페더레이션 프로그램과 공유되는 구성 요소를 사용하여 페더레이션 프로그램을 구성할 수 있으며, 이는 일반적으로 여러 다른 플랫폼에서 동일한 프로그램 로직을 실행하는 것을 의미합니다.

TFF의 페더레이션 프로그램 라이브러리([tff.program](https://www.tensorflow.org/federated/api_docs/python/tff/program))는 페더레이션 프로그램을 생성하는 데 필요한 추상화를 정의하고 [플랫폼에 구애받지 않는 구성 요소](#platform-agnostic-components)를 제공합니다.

## 구성 요소

TFF 페더레이션 프로그램 라이브러리의 **구성 요소**는 서로 다른 [역할](#roles)로 소유하고 작성할 수 있도록 설계되어 있습니다.

참고: 여기서는 구성 요소에 대한 개괄적인 개요만 설명하며 특정 API에 대한 문서는 [tff.program](https://www.tensorflow.org/federated/api_docs/python/tff/program)을 참조해 주세요.

### 프로그램

Python 바이너리인 이 **프로그램**은 다음과 같은 특성을 갖습니다.

1. [매개변수](#parameters)를 정의합니다(예: 플래그).
2. [플랫폼별 구성 요소](#platform-specific-components) 및 [플랫폼에 구애받지 않는 구성 요소](#platform-agnostic-components)를 구성합니다.
3. 페더레이션 컨텍스트에서 [프로그램 로직](#program_logic)을 사용하여 [계산](#computations)을 실행합니다.

예제:

```python
# Parameters set by the customer.
flags.DEFINE_string('output_dir', None, 'The output path.')

def main() -> None:

  # Parameters set by the program.
  total_rounds = 10
  num_clients = 3

  # Construct the platform-specific components.
  context = tff.program.NativeFederatedContext(...)
  data_source = tff.program.DatasetDataSource(...)

  # Construct the platform-agnostic components.
  summary_dir = os.path.join(FLAGS.output_dir, 'summary')
  metrics_manager = tff.program.GroupingReleaseManager([
      tff.program.LoggingReleaseManager(),
      tff.program.TensorBoardReleaseManager(summary_dir),
  ])
  program_state_dir = os.path.join(..., 'program_state')
  program_state_manager = tff.program.FileProgramStateManager(program_state_dir)

  # Define the computations.
  initialize = ...
  train = ...

  # Execute the computations using program logic.
  tff.framework.set_default_context(context)
  asyncio.run(
      train_federated_model(
          initialize=initialize,
          train=train,
          data_source=data_source,
          total_rounds=total_rounds,
          num_clients=num_clients,
          metrics_manager=metrics_manager,
          program_state_manager=program_state_manager,
      )
  )
```

### 매개변수

**매개변수**란 [프로그램](#program)에 대한 입력이며, 이러한 입력은 [고객](#customer)(customer)이 설정하거나, 플래그로 노출된 경우 프로그램에서 설정할 수 있습니다. 위의 예제에서 `output_dir`은 [고객](#customer)이 설정한 매개변수이고, `total_rounds` 및 `num_clients`은 프로그램에서 설정한 매개변수입니다.

### 플랫폼별 구성 요소

**플랫폼별 구성 요소**는 TFF의 페더레이션 프로그램 라이브러리에 정의된 추상 인터페이스를 구현하는 [플랫폼](#platform)에서 제공하는 구성 요소입니다.

### 플랫폼에 구애받지 않는 구성 요소

**플랫폼에 구애받지 않는 구성 요소**는 TFF의 페더레이션 프로그램 라이브러리에 정의된 추상 인터페이스를 구현하는 [라이브러리](#library)(예: TFF)에서 제공하는 구성 요소입니다.

### 계산

**계산**은 추상 인터페이스 [`tff.Computation`](https://www.tensorflow.org/federated/api_docs/python/tff/Computation)의 구현입니다.

예를 들어, TFF 플랫폼에서는 [`tff.tf_computation`](https://www.tensorflow.org/federated/api_docs/python/tff/tf_computation) 또는 [`tff.federated_computation`](https://www.tensorflow.org/federated/api_docs/python/tff/federated_computation) 데코레이터를 사용하여 [`tff.framework.ConcreteComputation`](https://www.tensorflow.org/federated/api_docs/python/tff/framework/ConcreteComputation)를 생성할 수 있습니다:

자세한 내용은 [계산의 수명](https://github.com/tensorflow/federated/blob/main/docs/design/life_of_a_computation.md)을 참조해 주세요.

### 프로그램 로직

**프로그램 로직**은 입력으로 다음을 받는 Python 함수입니다.

- [고객](#customer) 및 [프로그램](#program)으로 설정된 [매개변수](#parameters)
- [플랫폼별 구성 요소](#platform-specific-components)
- [플랫폼에 구애받지 않는 구성 요소](#platform-agnostic-components)
- [계산](#computations)

그리고 일반적으로 다음과 같은 몇 가지 작업을 수행합니다.

- [계산](#computations) 실행하기
- Python 로직 실행하기
- [플랫폼 저장소](#platform storage)에서 다음과 같은 방식으로 데이터를 [구체화](#materialize)하기
    - Python 로직의 사용
    - [내결함성](#fault tolerance)의 구현

그리고 다음과 같은 일부 출력을 생성할 수도 있습니다.

- [고객 저장소](#customer storage)로 데이터를 [메트릭](#metrics)으로 [릴리스](#release)

예제:

```python
async def program_logic(
    initialize: tff.Computation,
    train: tff.Computation,
    data_source: tff.program.FederatedDataSource,
    total_rounds: int,
    num_clients: int,
    metrics_manager: tff.program.ReleaseManager[
        tff.program.ReleasableStructure, int
    ],
) -> None:
  state = initialize()
  start_round = 1

  data_iterator = data_source.iterator()
  for round_number in range(1, total_rounds + 1):
    train_data = data_iterator.select(num_clients)
    state, metrics = train(state, train_data)

    _, metrics_type = train.type_signature.result
    metrics_manager.release(metrics, metrics_type, round_number)
```

## 역할

페더레이션 프로그램에 대해 논의할 때 정의에 유용하게 사용할 수 있는 세 가지 **역할**로 [고객](#customer), [플랫폼](#platform), [라이브러리](#library)가 있습니다. 이러한 각 역할은 페더레이션 프로그램을 만드는 데 사용되는 [구성 요소](#components) 중 일부를 소유하고 작성합니다. 다만 단일 엔터티 또는 그룹이 여러 역할을 수행할 수도 있습니다.

### 고객

**고객**은 일반적으로

- <a>고객 저장소</a>를 소유합니다.
- [프로그램](#program)을 실행합니다.

그리고

- [프로그램](#program)을 작성할 수도 있습니다.
- [플랫폼](#platform)의 기능을 수행할 수도 있습니다.

### 플랫폼

**플랫폼**은 일반적으로

- [플랫폼 저장소](#platform-storage)를 소유합니다.
- [플랫폼별 구성 요소](#platform-specific-components)를 작성합니다.

그리고

- [프로그램](#program)을 작성할 수도 있습니다.
- [라이브러리](#library)의 기능을 수행할 수도 있습니다.

### 라이브러리

**라이브러리**는 일반적으로

- [플랫폼에 구애받지 않는 구성 요소](#platform-agnostic-components)를 작성합니다.
- [계산](#computations)을 작성합니다.
- [프로그램 로직](#program-logic)을 작성합니다.

## 개념

페더레이션 프로그램에 대해 논의할 때 정의에 유용하게 사용할 수 있는 **개념**이 몇 가지 있습니다.

### 고객 저장소

**고객 저장소**는 [고객](#customer)에게 읽기 및 쓰기 액세스 권한이 있고 [플랫폼](#platform)에 쓰기 액세스 권한이 있는 저장소입니다.

### 플랫폼 저장소

**플랫폼 저장소**는 [platform](#platform)에만 읽기 및 쓰기 액세스 권한이 있는 저장소입니다.

### 릴리스

값을 **릴리스**하면 [고객 저장소](#customer-storage)에서 해당 값을 사용할 수 있습니다(예: 대시보드에 값 게시하기, 값 로깅하기 또는 디스크에 값 쓰기).

### 구체화

값 참조를 **구체화**하면 참조된 값을 [프로그램](#program)에서 사용할 수 있습니다. 값을 [릴리스](#release)하거나 [프로그램 로직](#program-logic)이 [내결함성](#fault-tolerance)을 갖도록 만들기 위해 값 참조를 구체화해야 하는 경우가 종종 있습니다.

### 내결함성

**내결함성**은 계산을 실행할 때 [프로그램 로직](#program-logic)을 장애로부터 복구할 수 있는 기능입니다. 예를 들어 100라운드 중 첫 90라운드를 성공적으로 훈련한 후 실패하면, 프로그램 로직이 91라운드부터 훈련을 재개할 수 있나요? 아니면 1라운드부터 훈련을 다시 시작해야 하나요?
