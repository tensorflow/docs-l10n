# 連合プログラム開発者ガイド

このドキュメントは、[連合プログラムロジック](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program-logic)や[連合プログラム](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program)の記述に関心のある方を対象としています。TensorFlow Federated の知識、特にその型システムの知識と、[連合プログラム](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md)に関する知識を前提としています。

[TOC]

## プログラムロジック

このセクションは、どのように[プログラムロジック](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program-logic)を記述すべきかに関するガイドラインを定義します。

詳細については、[program_logic.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program_logic.py) の例をご覧ください。

### 型シグネチャを文書化する

プログラムロジックに提供され、型シグネチャを持つパラメータごとに TFF 型シグネチャを**文書化してください**。

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

### 型シグネチャを確認する

プログラムロジックに提供され、型シグネチャを持つパラメータごとに TFF 型シグネチャを**確認してください**。

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

### 型注釈

プログラムロジックに提供される [`tff.program.ReleaseManager`](https://www.tensorflow.org/federated/api_docs/python/tff/program/ReleaseManager) パラメータごとに、十分に定義された Python 型を**指定してください。**

```python
async def program_logic(
    metrics_manager: Optional[
        tff.program.ReleaseManager[tff.program.ReleasableStructure, int]
    ] = None,
    ...
) -> None:
  ...
```

以下のようにしてはいけません。

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

### プログラムの状態

プログラムロジックのプログラムの状態を説明する、十分に定義された構造を**指定してください**。

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

### リリースされた値を文書化する

プログラムロジックからリリースされた値を**文書化してください**。

```python
async def program_logic(
    metrics_manager: Optional[tff.program.ReleaseManager] = None,
    ...
) -> None:
  """Trains a federated model for some number of rounds.

  Each round, `loss` is released to the `metrics_manager`.
  """
```

### 特定の値をリリースする

プログラムロジックから必要以上の数の値を**リリースしないでください**。

```python
async def program_logic(...) -> None:

  for round_number in range(...):
    state, metrics = train(state, ...)

    _, metrics_type = train.type_signature.result
    loss = metrics['loss']
    loss_type = metrics_type['loss']
    metrics_manager.release(loss, loss_type, round_number)
```

以下のようにしてはいけません。

```python
async def program_loic(...) -> None:

  for round_number in range(...):
    state, metrics = train(state, ...)

    _, metrics_type = train.type_signature.result
    metrics_manager.release(metrics, metrics_type, round_number)
```

注意: 必要されているのであれば、すべての値をリリースすることができます。

### 非同期関数

プログラムロジックを[非同期関数](https://docs.python.org/3/reference/compound_stmts.html#coroutine-function-definition)として**定義してください**。TFF の連合プログラムライブラリの[コンポーネント](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#components)は、[asyncio](https://docs.python.org/3/library/asyncio.html) を使用して Python を同時に実行するため、プログラムロジックを非同期関数として定義すると、それらのコンポーネントの操作がより簡単になります。

```python
async def program_logic(...) -> None:
  ...
```

以下のようにしてはいけません

```python
def program_logic(...) -> None:
  ...
```

### テスト

プログラムロジックの単体テストを**指定してください**（[program_logic_test.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program_logic_test.py) など）。

## プログラム

このセクションは、どのように[プログラム](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program)を記述すべきかに関するガイドラインを定義します。

詳細については、[program.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program.py) の例をご覧ください。

### プログラムを文書化する

モジュールの docstring に顧客向けのプログラムの詳細を**文書化してください**（[program.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program.py) など）。

- 手動でプログラムを実行する方法。
- プログラムに使用されているプラットフォーム、計算、およびデータソース。
- プログラムから顧客ストレージにリリースされた情報に顧客がアクセスする方法。

### パラメータが多すぎる

相互に排他的なパラメータのコレクションできるようにプログラムを**パラメータ化してはいけません**。たとえば、`foo` が `X` に設定されている場合、パラメータ `bar` と `baz` も設定する必要があります。設定しない場合、これらのパラメータは `None` にする必要があります。こうすうことで、異なる `foo` の値に 2 つの異なるプログラムを作成したことになります。

### パラメータをグループ化する

多数のフラグ（go/absl.flags）を定義する代わりに、関連していても複合または冗長なパラメータを定義するには、proto を**使用してください**。

> 注意: Proto は以下のように、ディスクから読み取り可能で、Python オブジェクトの構築に使用できます。
>
> ```python
> with tf.io.gfile.GFile(config_path) as f:
>   proto = text_format.Parse(f.read(), vizier_pb2.StudyConfig())
> return pyvizier.StudyConfig.from_proto(proto)
> ```

### Python ロジック

プログラムにロジック（制御フロー、計算の呼び出し、テストが必要となるものなど）を**記述してはいけません**。ロジックはテスト可能なプライベートライブラリやプログラムが呼び出すプログラムロジックに移動してください。

### 非同期関数

プログラムに[非同期関数](https://docs.python.org/3/reference/compound_stmts.html#coroutine-function-definition)を**記述してはいけません**。関数はテスト可能なプライベートライブラリやプログラムが呼び出すプログラムロジックに移動してください。

### テスト

プログラムに対して単体テストを**書いてはいけません**。プログラムのテストが有用である場合は、統合テストに関してテストを書いてください。

注意: Python ロジックと非同期関数がライブラリに移動されてテストされる場合、プログラムのテストが有用である可能性はあまりありません。
