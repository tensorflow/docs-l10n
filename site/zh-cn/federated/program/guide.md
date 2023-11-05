# 联合程序开发者指南

本文档适用于对创作[联合程序逻辑](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program-logic)或[联合程序](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program)感兴趣的任何人。它假定您了解 TensorFlow Federated，尤其是其类型系统和[联合程序](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md)。

[目录]

## 程序逻辑

本部分定义了如何创作[程序逻辑](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program-logic)的指南。

如需了解详情，请参阅示例 [program_logic.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program_logic.py)。

### 文档类型签名

**务必**记录提供给具有类型签名的程序逻辑的每个参数的 TFF 类型签名。

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

### 检查类型签名

**务必**检查提供给具有类型签名的程序逻辑的每个参数的 TFF 类型签名（在运行时）。

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

### 类型注解

**务必**为提供给程序逻辑的每个 [`tff.program.ReleaseManager`](https://www.tensorflow.org/federated/api_docs/python/tff/program/ReleaseManager) 参数提供定义良好的 Python 类型。

```python
async def program_logic(
    metrics_manager: Optional[
        tff.program.ReleaseManager[tff.program.ReleasableStructure, int]
    ] = None,
    ...
) -> None:
  ...
```

不是

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

### 程序状态

**务必**提供定义良好的结构来描述程序逻辑的程序状态。

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

### 记录发布的值

**务必**记录从程序逻辑中发布的值。

```python
async def program_logic(
    metrics_manager: Optional[tff.program.ReleaseManager] = None,
    ...
) -> None:
  """Trains a federated model for some number of rounds.

  Each round, `loss` is released to the `metrics_manager`.
  """
```

### 发布特定值

**请勿**从程序逻辑中发布超出需求数量的值。

```python
async def program_logic(...) -> None:

  for round_number in range(...):
    state, metrics = train(state, ...)

    _, metrics_type = train.type_signature.result
    loss = metrics['loss']
    loss_type = metrics_type['loss']
    metrics_manager.release(loss, loss_type, round_number)
```

不是

```python
async def program_loic(...) -> None:

  for round_number in range(...):
    state, metrics = train(state, ...)

    _, metrics_type = train.type_signature.result
    metrics_manager.release(metrics, metrics_type, round_number)
```

注意：如果需要，可以发布所有值。

### 异步函数

**务必**将程序逻辑定义为[异步函数](https://docs.python.org/3/reference/compound_stmts.html#coroutine-function-definition)。TFF 联合程序库的[组件](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#components)使用 [asyncio](https://docs.python.org/3/library/asyncio.html) 来并发执行 Python，并将程序逻辑定义为异步函数，从而简化与这些组件交互的过程。

```python
async def program_logic(...) -> None:
  ...
```

不是

```python
def program_logic(...) -> None:
  ...
```

### 测试

**务必**为程序逻辑提供单元测试（例如 [program_logic_test.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program_logic_test.py)）。

## 程序

本部分定义了应如何创作[程序逻辑](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program)的指南。

如需了解详情，请参阅示例 [program_logic.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program.py)。

### 记录程序

**务必**在模块的文档字符串（例如 [program.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program.py)）中为客户记录程序的详细信息：

- 如何手动运行程序。
- 程序中使用哪些平台、计算和数据源。
- 客户应如何访问从程序发布到客户存储空间的信息。

### 参数过多

**请勿**参数化程序，以免出现互斥的参数集合。例如，如果 `foo` 设置为 `X`，则还需要设置参数 `bar`、`baz`，否则这些参数必须为 `None`。这表明您可以为不同的 `foo` 值创建两个不同的程序。

### 组参数

**务必**使用 proto 来定义相关但复杂或冗长的参数，而不是定义许多 FLAGS (go/absl.flags)。

> 注：Proto 可以从磁盘读取并用于构造 Python 对象，例如：
>
> ```python
> with tf.io.gfile.GFile(config_path) as f:
>   proto = text_format.Parse(f.read(), vizier_pb2.StudyConfig())
> return pyvizier.StudyConfig.from_proto(proto)
> ```

### Python 逻辑

**请勿**在程序中编写逻辑（例如控制流、调用计算、需要测试的任何内容）。相反，请将逻辑移至可测试的私有库中或移至程序调用的程序逻辑中。

### 异步函数

**请勿**在程序中编写[异步函数](https://docs.python.org/3/reference/compound_stmts.html#coroutine-function-definition)。相反，请将函数移至可测试的私有库中或移至程序调用的程序逻辑中。

### 测试

**请勿**为程序编写单元测试，如果测试程序有用，请根据集成测试编写这些测试。

注：如果将 Python 逻辑和异步函数移至库中并进行测试，则测试程序不太可能有用。
