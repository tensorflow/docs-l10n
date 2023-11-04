# Guía para el desarrollador de programas federados

Esta documentación es para cualquier persona que esté interesada en escribir [lógica de programación federada](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program-logic) o un [programa federado](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program). Se da por supuesto un conocimiento previo de TensorFlow federado, de su sistema de tipo y de los [programas federados](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md).

[TOC]

## Lógica de programación

En esta sección se definen las pautas sobre cómo se debería escribir la [lógica de programación](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program-logic).

Para más información, consulte el ejemplo [program_logic.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program_logic.py).

### Documentación de firmas de tipo

**Documente** la firma de tipo de TFF para cada parámetro provisto a la lógica de programación que tenga una firma de tipo.

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

### Control de las firmas de tipo

**Controle** la firma de tipo de TFF (en tiempo de ejecución) para cada parámetro provisto a la lógica de programación que tenga una firma de tipo.

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

### Anotaciones de tipo

**Proporcione** un tipo de Python bien definido para cada parámetro [`tff.program.ReleaseManager`](https://www.tensorflow.org/federated/api_docs/python/tff/program/ReleaseManager) provisto a la lógica de programación.

```python
async def program_logic(
    metrics_manager: Optional[
        tff.program.ReleaseManager[tff.program.ReleasableStructure, int]
    ] = None,
    ...
) -> None:
  ...
```

No

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

### Estado del programa

**Proporcione** una estructura bien definida que describa el estado (estado del programa) de la lógica de programación.

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

### Documentación de los valores emitidos

**Documente** los valores emitidos por la lógica de programación.

```python
async def program_logic(
    metrics_manager: Optional[tff.program.ReleaseManager] = None,
    ...
) -> None:
  """Trains a federated model for some number of rounds.

  Each round, `loss` is released to the `metrics_manager`.
  """
```

### Emisión de los valores específicos

**No** emita más valores de la lógica de programación, de los que se requieran.

```python
async def program_logic(...) -> None:

  for round_number in range(...):
    state, metrics = train(state, ...)

    _, metrics_type = train.type_signature.result
    loss = metrics['loss']
    loss_type = metrics_type['loss']
    metrics_manager.release(loss, loss_type, round_number)
```

No

```python
async def program_loic(...) -> None:

  for round_number in range(...):
    state, metrics = train(state, ...)

    _, metrics_type = train.type_signature.result
    metrics_manager.release(metrics, metrics_type, round_number)
```

Nota: Es correcto emitir todos los valores, si es lo que se requiere.

### Funciones asincrónicas

**Defina** la lógica de programación como una [función asincrónica](https://docs.python.org/3/reference/compound_stmts.html#coroutine-function-definition). Los [componentes](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#components) de la biblioteca de programas federados de TFF usan [asyncio](https://docs.python.org/3/library/asyncio.html) para ejecutar Python en simultáneo y la definición de la lógica de programación como una función asincrónica facilita la interacción con esos componentes.

```python
async def program_logic(...) -> None:
  ...
```

No

```python
def program_logic(...) -> None:
  ...
```

### Pruebas

**Proporcione** pruebas de unidades para la lógica de programación (p. ej., [program_logic_test.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program_logic_test.py)).

## Programa

En esta sección se definen las pautas sobre cómo se debería escribir un [programa](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program).

Para más información, consulte el ejemplo [program.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program.py).

### Documentación del programa

**Documente** los detalles del programa para el cliente en el <em>docstring</em> del módulo (p. ej., [program.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program.py)):

- Cómo ejecutar el programa manualmente.
- Qué plataforma, cálculos y fuentes de datos se usan en el programa.
- Qué debería hacer un cliente para acceder a la información emitida desde el programa al almacenamiento del cliente.

### Demasiados parámetros

**No** parametrice el programa de modo tal que haya muchas colecciones mutuamente excluyentes de parámetros. Por ejemplo, si `foo` se establece como `X`, entonces, también hay que establecer parámetros para `bar`, `baz`, de lo contrario estos parámetros deben ser `None`. Esto indica que se podrían haber hecho dos programas diferentes para distintos valores de `foo`.

### Parámetros grupales

**Use** <em>proto</em> para definir parámetros relacionados pero complejos o verbosos en vez de definir muchas FLAGS (go/absl.flags).

> Nota: <em>Proto</em> se puede leer del disco y se puede usar para construir objetos de Python, por ejemplo:
>
> ```python
> with tf.io.gfile.GFile(config_path) as f:
>   proto = text_format.Parse(f.read(), vizier_pb2.StudyConfig())
> return pyvizier.StudyConfig.from_proto(proto)
> ```

### Lógica de Python

**No** escriba cálculos de lógica (p. ej., control de flujo, cálculos para invocar o cualquier cosa que deba ser probada) en el programa. En cambio, mueva la lógica a una biblioteca privada de la que se puedan hacer pruebas o muévala a la lógica del programa que el mismo programa invoque.

### Funciones asincrónicas

**No** escriba [funciones asincrónicas](https://docs.python.org/3/reference/compound_stmts.html#coroutine-function-definition) en el programa. En cambio, mueva la función a una biblioteca privada de la que se puedan hacer pruebas o muévala a la lógica del programa que el mismo programa invoque.

### Pruebas

**No** escriba pruebas de unidad para el programa, si hacerle pruebas al programa le resulta útil, escriba esas pruebas en términos de pruebas de integración.

Nota: Debería ser poco probable que probar el programa fuera útil, si las funciones de asincronía y lógica de Python se mueven a bibliotecas y se les realizan pruebas.
