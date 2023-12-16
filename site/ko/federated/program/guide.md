# 페더레이션 프로그램 개발자 가이드

이 문서는 [페더레이션 프로그램 로직](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program-logic) 또는 [페더레이션 프로그램](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program) 작성에 관심이 있는 분들을 대상으로 작성되었습니다. 이 문서를 읽으려면 TensorFlow 페더레이션, 특히 유형 시스템과 [페더레이션 프로그램](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md)에 대한 지식이 있어야 합니다.

[목차]

## 프로그램 로직

이 섹션에서는 [프로그램 로직](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program-logic)을 작성하는 방법에 대한 가이드라인을 정의합니다.

자세한 정보는 [program_logic.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program_logic.py) 예제를 참조해 주세요.

### 문서 유형 서명

유형 서명이 있는 프로그램 로직에 공급된 각 매개변수에 대한 TFF 유형 서명을 **문서화합니다**.

```python
async def program_logic(
    train: tff.Computation,
    data_source: tff.program.FederatedDataSource,
    ...
) -> None:
  """Trains a federated model for some number of rounds.

  The following types signatures are required:

  1.  `train`:       `(<S@SERVER, D@CLIENTS> -> <S@SERVER, M@SERVER>)`
  2.  `data_source`: `D@CLIENTS`

  Where:

  *   `S`: The server state.
  *   `M`: The train metrics.
  *   `D`: The train client data.
  """
```

### 유형 서명 확인

유형 서명이 있는 프로그램 로직에 공급된 각 매개변수에 대한 TFF 유형 서명을 **확인합니다**.

```python
def _check_program_logic_type_signatures(
    train: tff.Computation,
    data_source: tff.program.FederatedDataSource,
    ...
) -> None:
  ...

async def program_logic(
    train: tff.Computation,
    data_source: tff.program.FederatedDataSource,
    ...
) -> None:
  _check_program_logic_type_signatures(
      train=train,
      data_source=data_source,
  )
  ...
```

### 유형 주석

프로그램 로직에 제공된 각 [`tff.program.ReleaseManager`](https://www.tensorflow.org/federated/api_docs/python/tff/program/ReleaseManager) 매개변수에 대해 잘 정의된 Python 유형을 **제공합니다.**

```python
async def program_logic(
    metrics_manager: Optional[
        tff.program.ReleaseManager[tff.program.ReleasableStructure, int]
    ] = None,
    ...
) -> None:
  ...
```

안 좋은 예제

```python
async def program_logic(
    metrics_manager,
    ...
) -> None:
  ...
```

```python
async def program_logic(
    metrics_manager: Optional[tff.program.ReleaseManager] = None,
    ...
) -> None:
  ...
```

### 프로그램 상태

프로그램 로직의 프로그램 상태를 설명하는 잘 정의된 구조를 **제공합니다**.

```python
class _ProgramState(NamedTuple):
  state: object
  round_num: int

async def program_loic(...) -> None:
  initial_state = ...

  # Load the program state
  if program_state_manager is not None:
    structure = _ProgramState(initial_state, round_num=0)
    program_state, version = await program_state_manager.load_latest(structure)
  else:
    program_state = None
    version = 0

  # Assign state and round_num
  if program_state is not None:
    state = program_state.state
    start_round = program_state.round_num + 1
  else:
    state = initial_state
    start_round = 1

  for round_num in range(start_round, ...):
    state, _ = train(state, ...)

    # Save the program state
    program_state = _ProgramState(state, round_num)
    version = version + 1
    program_state_manager.save(program_state, version)
```

### 릴리스 값 문서화

프로그램 로직에서 릴리스한 값을 **문서화합니다**.

```python
async def program_logic(
    metrics_manager: Optional[tff.program.ReleaseManager] = None,
    ...
) -> None:
  """Trains a federated model for some number of rounds.

  Each round, `loss` is released to the `metrics_manager`.
  """
```

### 특정 값 릴리스

필요 이상으로 프로그램 로직에서 더 많은 값을 해제하면 **안 됩니다**.

```python
async def program_logic(...) -> None:

  for round_number in range(...):
    state, metrics = train(state, ...)

    _, metrics_type = train.type_signature.result
    loss = metrics['loss']
    loss_type = metrics_type['loss']
    metrics_manager.release(loss, loss_type, round_number)
```

안 좋은 예제

```python
async def program_loic(...) -> None:

  for round_number in range(...):
    state, metrics = train(state, ...)

    _, metrics_type = train.type_signature.result
    metrics_manager.release(metrics, metrics_type, round_number)
```

참고: 필요한 경우에는 모든 값을 릴리스해도 좋습니다.

### 비동기식 함수

프로그램 로직을 [비동기 함수](https://docs.python.org/3/reference/compound_stmts.html#coroutine-function-definition)로 **정의합니다**. TFF 페더레이션 프로그램 라이브러리의 [구성 요소](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#components)는 [asyncio](https://docs.python.org/3/library/asyncio.html)를 사용하여 Python을 동시에 실행하고 프로그램 로직을 비동기식 함수로 정의하기에 이러한 구성 요소와 더 쉽게 상호 작용할 수 있습니다.

```python
async def program_logic(...) -> None:
  ...
```

안 좋은 예제

```python
def program_logic(...) -> None:
  ...
```

### 테스트

프로그램 로직에 대한 단위 테스트를 **제공합니다**(예: [program_logic_test.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program_logic_test.py)).

## 프로그램

이 섹션에서는 [프로그램](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program)을 작성하는 방법에 대한 가이드라인을 정의합니다.

자세한 정보는 [program.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program.py) 예제를 참조해 주세요.

### 프로그램 문서화

모듈의 docstring에 있는 고객(customer)에게 프로그램에 대한 세부 정보를 **문서화합니다**(예: [program.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program.py)).

- 프로그램을 수동으로 실행하는 방법.
- 프로그램에서 사용되는 플랫폼, 계산 및 데이터 소스.
- 고객이 프로그램에서 고객 저장소로 릴리스되는 정보에 액세스하는 방법.

### 너무 많은 매개변수

상호 배타적인 매개변수 모음이 있도록 프로그램을 매개변수화하지 **않아야 합니다**. 예를 들어 `foo`가 `X`로 설정된 경우 매개변수 `bar`, `baz`도 설정해야 하며, 그렇지 않으면 이러한 매개변수를 `None`으로 설정해야 합니다. 이는 `foo`의 다른 값에 대해 두 개의 다른 프로그램을 만들 수 있음을 나타냅니다.

### 매개변수 그룹화

다수의 FLAGS(go/absl.flags)를 정의하는 대신 관련이 있지만 복잡하거나 장황한 매개변수를 정의하려면 proto를 **사용합니다**.

> 참고: 디스크에서 읽어와서 Python 객체를 구성하는 데 proto를 사용할 수 있습니다.
>
> ```python
> with tf.io.gfile.GFile(config_path) as f:
>   proto = text_format.Parse(f.read(), vizier_pb2.StudyConfig())
> return pyvizier.StudyConfig.from_proto(proto)
> ```

### Python 로직

프로그램에서 로직(예: 통제 절차, 계산 호출, 테스트가 필요한 모든 것)을 작성하지 **말아야 합니다**. 대신 테스트할 수 있는 비공개 라이브러리나 프로그램이 호출하는 프로그램 로직으로 로직을 이동합니다.

### 비동기식 함수

프로그램에서 [비동기식 함수](https://docs.python.org/3/reference/compound_stmts.html#coroutine-function-definition)를 작성하지 **말아야 합니다**. 대신 테스트할 수 있는 개인 라이브러리 또는 프로그램이 호출하는 프로그램 로직으로 함수를 이동합니다.

### 테스트

프로그램에 대한 단위 테스트를 작성하지 **말아야 합니다**. 다만 프로그램 테스트가 유용한 경우에는 통합 테스트 측면에서 해당 테스트를 작성합니다.

참고: Python 로직과 비동기식 함수를 라이브러리로 이동하여 테스트하는 경우 프로그램을 테스트하는 것이 유용하지 않을 수 있습니다.
