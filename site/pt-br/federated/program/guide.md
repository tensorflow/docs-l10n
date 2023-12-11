# Guia de Desenvolvimento de Programa Federado

Este documento é destinado a todos que tiverem interesse em criar [lógica de programa federado](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program-logic) ou um [programa federado](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program). Pressupõem-se conhecimentos do TensorFlow Federated, principalmente o sistema de tipos, e [programas federados](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md).

[TOC]

## Lógica do programa

Esta seção define as diretrizes de como a [lógica do programa](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program-logic) deve ser criada.

Confira mais informações no exemplo [program_logic.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program_logic.py).

### Documente assinaturas de tipo

**Documente** a assinatura de tipo do TFF para cada parâmetro fornecido à lógica do programa que tenha um assinatura de tipo.

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

### Faça a checagem de assinaturas de tipo

**Faça a checagem** da assinatura de tipo do TFF (em tempo de execução) para cada parâmetro fornecido à lógica do programa que tenha um assinatura de tipo.

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

### Anotações de tipo

**Forneça** um tipo Python bem definido para cada parâmetro de [`tff.program.ReleaseManager`](https://www.tensorflow.org/federated/api_docs/python/tff/program/ReleaseManager) fornecido à lógica do programa.

```python
async def program_logic(
    metrics_manager: Optional[
        tff.program.ReleaseManager[tff.program.ReleasableStructure, int]
    ] = None,
    ...
) -> None:
  ...
```

Não faça isto:

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

### Estado do programa

**Forneça** uma estrutura bem definida que descreva o estado do programa referente à lógica do programa.

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

### Documente os valores liberados

**Documente** os valores liberados pela lógica do programa.

```python
async def program_logic(
    metrics_manager: Optional[tff.program.ReleaseManager] = None,
    ...
) -> None:
  """Trains a federated model for some number of rounds.

  Each round, `loss` is released to the `metrics_manager`.
  """
```

### Libere valores específicos

**Não** libere mais valores da lógica do programa do que o necessário.

```python
async def program_logic(...) -> None:

  for round_number in range(...):
    state, metrics = train(state, ...)

    _, metrics_type = train.type_signature.result
    loss = metrics['loss']
    loss_type = metrics_type['loss']
    metrics_manager.release(loss, loss_type, round_number)
```

Não faça isto:

```python
async def program_loic(...) -> None:

  for round_number in range(...):
    state, metrics = train(state, ...)

    _, metrics_type = train.type_signature.result
    metrics_manager.release(metrics, metrics_type, round_number)
```

Observação: não tem problema liberar todos os valores, desde que isso seja necessário.

### Funções assíncronas

**Defina** a lógica do programa como uma [função assíncrona](https://docs.python.org/3/reference/compound_stmts.html#coroutine-function-definition). Os [componentes](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#components) da biblioteca de programa federado do TFF usam [asyncio](https://docs.python.org/3/library/asyncio.html) para executar o Python de maneira simultânea, e definir a lógica do programa como uma função assíncrona facilita a interação com esses componentes.

```python
async def program_logic(...) -> None:
  ...
```

Não faça isto:

```python
def program_logic(...) -> None:
  ...
```

### Testes

**Forneça** testes de unidade à lógica do programa (por exemplo, [program_logic_test.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program_logic_test.py)).

## Programa

Esta seção define as diretrizes de como um [programa](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program) deve ser criado.

Confira mais informações no exemplo [program.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program.py).

### Documente o programa

**Documente** os detalhes do programa para o cliente na docstring do módulo (por exemplo, [program.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program.py)):

- Como executar o programa manualmente
- Que plataformas, computações e fontes de dados são usados no programa.
- Como um cliente deve acessar as informações liberadas pelo programa para o armazenamento do cliente.

### Parâmetros demais

**Não** parametrize o programa de tal forma que haja conjuntos de parâmetros mutuamente excludentes. Por exemplo, se `foo` for definido como `X`, então você também precisa definir os parâmetros `bar`, `baz`; caso contrário, esses parâmetros devem ser `None` (Nenhum). Isso indica que você pode ter dois programas diferentes para valores diferentes de `foo`.

### Agrupe parâmetros

**Use** proto para definir parâmetros relacionados, mas complexos ou detalhados, em vez de definir muitos SINALIZADORES (go/absl.flags).

> Observação: proto pode ser lido no disco e usado para construir objetos Python. Por exemplo:
>
> ```python
> with tf.io.gfile.GFile(config_path) as f:
>   proto = text_format.Parse(f.read(), vizier_pb2.StudyConfig())
> return pyvizier.StudyConfig.from_proto(proto)
> ```

### Lógica do Python

**Não** escreva lógica (por exemplo, fluxo de controle, invocação de computações, qualquer coisa que precise ser testada) no programa. Em vez disso, mova a lógica para uma biblioteca privada que possa ser testada ou para a lógica do programa que o programa invoca.

### Funções assíncronas

**Não** escreva [funções assíncronas](https://docs.python.org/3/reference/compound_stmts.html#coroutine-function-definition) no programa. Em vez disso, mova a função para uma biblioteca privada que possa ser testada ou para a lógica do programa que o programa invoca.

### Testes

**Não** escreva testes de unidade para o programa. Se testar o programa for útil, escreva testes de integração.

Observação: é improvável que testar o programa seja útil se a lógica do Python e as funções assíncronas forem movidas para bibliotecas e testadas.
