# Programa federado

Esta documentación es para cualquier persona que esté interesada en acceder a una descripción de alto nivel de los conceptos de programa federado. Se da por supuesto un conocimiento previo de TensorFlow federado, en particular de su sistema de tipo.

Para más información sobre el programa federado, consulte:

- [Documentación sobre las API](https://www.tensorflow.org/federated/api_docs/python/tff/program)
- [Ejemplos](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program)
- [Guía del desarrollador](https://github.com/tensorflow/federated/blob/main/docs/program/guide.md)

[TOC]

## Qué es un programa federado

Un **programa federado** es un programa que ejecuta cálculos y otros procesamientos lógicos en un entorno federado.

Para ser más precisos, un **programa federado** hace específicamente lo siguiente:

- ejecuta [cálculos](#computations)
- usando la [lógica de programación](#program-logic)
- con [componentes específicos de la plataforma](#platform-specific-components)
- y [componentes agnósticos de la plataforma](#platform-agnostic-components)
- dados unos [parámetros](#parameters) establecidos por el [programa](#program)
- y [parámetros](#parameters) establecidos por el [cliente](#customer)
- cuando el [cliente](#customer) ejecuta el [programa](#program)
- y podría [materializar](#materialize) datos en el [almacenamiento de la plataforma](#platform storage) para:
    - usarlos con la lógica de Python
    - implementar la [tolerancia a fallos](#fault tolerance)
- y podría [lanzar](#release) datos para el [almacenamiento del cliente](#customer storage)

La definición de estos [conceptos](#concepts) y abstracciones posibilita la descripción de las relaciones entre los [componentes](#components) de un programa federado y permite que distintos [roles](#roles) puedan poseerlos y escribirlos. Este desacoplamiento permite que los desarrolladores compongan programas federados con componentes que se comparten con otros programas también federados. Normalmente, implica la ejecución de la misma lógica de programación en muchas plataformas diferentes.

En la biblioteca de programas federados de TFF ([tff.program](https://www.tensorflow.org/federated/api_docs/python/tff/program)) se definen las abstracciones requeridas para crear un programa federado y se brindan los [componentes agnósticos de la plataforma](#platform-agnostic-components).

## Componentes

Los **componentes** de la biblioteca de programas federados de TFF están diseñados de modo tal que diferentes [roles](#roles) puedan poseerlos o escribirlos.

Nota: Esta es una descripción general de alto nivel de los componentes. Para acceder a documentación más específica sobre las API, consulte [tff.program](https://www.tensorflow.org/federated/api_docs/python/tff/program).

### Programa

El **programa** es un binario de Python que hace lo siguiente:

1. define [parámetros](#parameters) (p. ej., marcas (<em>flags</em>))
2. construye [componentes específicos de la plataforma](#platform-specific-components) y [componentes agnósticos de la plataforma](#platform-agnostic-components)
3. ejecuta [cálculos](#computations) con una [lógica de programación](#program_logic) en un contexto federado

Por ejemplo:

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

### Parámetros

Los **parámetros** son las entradas del [programa](#program). A estas entradas las puede establecer el [cliente](#customer), si las expone como marcas, o pueden ser establecidas por el programa. En el ejemplo anterior `output_dir` es un parámetro establecido por el [cliente](#customer), y `total_rounds` y `num_clients` son parámetros establecidos por el programa.

### Componentes específicos de la plataforma

Los **componentes específicos de la plataforma** son aquellos componentes provistos por una [plataforma](#platform) mediante la implementación de interfaces abstractas definidas por la biblioteca de programación federada de TFF.

### Componentes agnósticos de la plataforma

Los **componentes agnósticos de la plataforma** son aquellos componentes provistos por una [biblioteca](#library) (p. ej., la de TFF) mediante la implementación de interfaces abstractas definidas por la biblioteca de programación federada de TFF.

### Cálculos

Los **cálculos** son las implementaciones de la interfaz abstracta [`tff.Computation`](https://www.tensorflow.org/federated/api_docs/python/tff/Computation).

Por ejemplo, en la plataforma TFF se pueden usar decoradores [`tff.tf_computation`](https://www.tensorflow.org/federated/api_docs/python/tff/tf_computation) o [`tff.federated_computation`](https://www.tensorflow.org/federated/api_docs/python/tff/federated_computation) para crear una [`tff.framework.ConcreteComputation`](https://www.tensorflow.org/federated/api_docs/python/tff/framework/ConcreteComputation):

Para más información consulte [la vida de un cálculo](https://github.com/tensorflow/federated/blob/main/docs/design/life_of_a_computation.md).

### Lógica de programación

La **lógica de programación** es una función de Python que toma como entrada lo siguiente:

- [parámetros](#parameters) establecidos por el [cliente](#customer) y el [programa](#program)
- [componentes específicos de la plataforma](#platform-specific-components)
- [componentes agnósticos de la plataforma](#platform-agnostic-components)
- [cálculos](#computations)

y realiza algunas operaciones, que normalmente incluyen lo siguiente:

- la ejecución de [cálculos](#computations)
- la ejecución de la lógica de Python
- la [materialización](#materialize) de los datos en el [almacenamiento de la plataforma](#platform storage) para:
    - usarlos con la lógica de Python
    - implementar la [tolerancia a fallos](#fault tolerance)

y puede producir algunas salidas, que normalmente incluyen:

- el [lanzamiento](#release) de datos para el [almacenamiento del cliente](#customer storage) como [métricas](#metrics)

Por ejemplo:

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

## Roles

Hay tres **roles** que son útiles para definir programas federados: el [cliente](#customer), la [plataforma](#platform) y la [biblioteca](#library). Cada uno de estos tres roles es dueño y autor de algunos de los [componentes](#components) usados para crear un programa federado. Sin embargo, es posible que una entidad o grupo solos cumplan varios roles.

### Cliente

El **cliente** por lo común:

- es dueño del [almacenamiento del cliente](#customer-storage)
- lanza el [programa](#program)

y podría:

- ser autor del [programa](#program)
- satisfacer cualquiera de las capacidades de la [plataforma](#platform)

### Plataforma

La **plataforma** por lo común:

- es dueña del [almacenamiento de la plataforma](#platform-storage)
- es autora de los [componentes específicos de la plataforma](#platform-specific-components)

y podría:

- ser autora del [programa](#program)
- satisfacer cualquiera de las capacidades de la [biblioteca](#library)

### Biblioteca

La **biblioteca** por lo común:

- es autora de los [componentes agnósticos de la plataforma](#platform-agnostic-components)
- es autora de los [cálculos](#computations)
- es autora de la [lógica de programación](#program-logic)

## Conceptos

Hay algunos **conceptos** que resulta útil entender para hablar sobre programas federados.

### Almacenamiento del cliente

El **almacenamiento del cliente** es todo almacenamiento al que el [cliente](#customer) tiene permiso de lectura y escritura, y al que la [plataforma](#platform) tiene permiso de escritura.

### Almacenamiento de la plataforma

El **almacenamiento de la plataforma** es todo almacenamiento al que solamente la [plataforma](#platform) tiene permiso de lectura y escritura.

### Lanzamiento

El **lanzamiento** de un valor hace que ese valor esté disponible para el [almacenamiento del cliente](#customer-storage) (p. ej., la publicación del valor en un panel, el registro del valor o la escritura del valor en el disco).

### Materialización

La **materialización** de una referencia de valor hace que el valor referido esté disponible para el [programa](#program). Con frecuencia, es necesario materializar una referencia de valor para [lanzar](#release) el valor o para hacer que la [lógica de programación](#program-logic) sea [tolerante a fallos](#fault-tolerance).

### Tolerancia a fallos

La **tolerancia a fallos** es la capacidad de la [lógica de programación](#program-logic) para recuperarse de un fallo mientras ejecuta un cálculo. Por ejemplo, si se entrena correctamente las primeras 90 rondas de 100 y después se produce un fallo, ¿la lógica de programación es capaz de reanudar a partir de la ronda 91 o hay que reiniciar el entrenamiento desde la ronda 1?
