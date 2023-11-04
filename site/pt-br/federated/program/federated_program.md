# Programa federado

Este documento é destinado a todos que tiverem interesse em uma visão geral dos conceitos de programa federado. Pressupõem-se conhecimentos do TensorFlow Federated, principalmente o sistema de tipos.

Confira mais informações sobre o programa federado em:

- [Documentação da API](https://www.tensorflow.org/federated/api_docs/python/tff/program)
- [Exemplos](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program)
- [Guia de Desenvolvimento](https://github.com/tensorflow/federated/blob/main/docs/program/guide.md)

[TOC]

## O que é um programa federado?

Um **programa federado** é um programa que executa computações e outras lógicas de processamento em um ambiente federado.

Mais especificamente, um **programa federado**:

- executa [computações](#computations)
- usando [lógica de programa](#program-logic)
- com [componentes específicos de plataformas](#platform-specific-components)
- e [componentes independentes de plataforma](#platform-agnostic-components)
- dados [parâmetros](#parameters) definidos pelo [programa](#program)
- e [parâmetros](#parameters) definidos pelo [cliente](#customer)
- quando o [cliente](#customer) executa o [programa](#program)
- e pode [materializar](#materialize) dados no [armazenamento da plataforma](#platform storage) para:
    - usar em lógicas do Python
    - implementar [tolerância a falhas](#fault tolerance)
- e pode [liberar](#release) dados para o [armazenamento do cliente](#armazenamento do cliente)

Ao definir esses [conceitos](#concepts) e abstrações, é possível descrever as relações entre os [componentes](#components) de um programa federado, e esses componentes podem ser de propriedade de diferentes [funções](#roles) e criados por elas. Com esse desacoplamento, os desenvolvedores podem criar um programa federado usando componentes compartilhados com outros programas federados, o que geralmente implica executar a mesma lógica do programa em várias plataformas diferentes.

A biblioteca de programa federado ([tff.program](https://www.tensorflow.org/federated/api_docs/python/tff/program)) do TFF define as abstrações necessárias para criar um programa federado, além de fornecer [componentes independentes de plataforma](#platform-agnostic-components).

## Componentes

Os **componentes** da biblioteca de programa federado do TFF foram criados para que possam ser de propriedade de diferentes [funções](#roles) e criados por elas.

Observação: esta é uma visão geral dos componentes. Confira a documentação de APIs específicas em [tff.program](https://www.tensorflow.org/federated/api_docs/python/tff/program).

### Programa

O **programa** é um binário do Python que:

1. define [parâmetros](#parameters) (por exemplo, sinalizadores)
2. constrói [componentes específicos de plataformas](#platform-specific-components) e [componentes independentes de plataforma](#platform-agnostic-components)
3. executa [computações](#computations) usando [lógica de programa](#program_logic) em um contexto federado

Por exemplo:

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

### Parâmetros

Os **parâmetros** são as entradas do [programa](#program), que podem ser definidos pelo [cliente](#customer), se estiverem expostos como sinalizadores, ou podem ser definidos pelo programa. No exemplo acima, `output_dir` é um parâmetro definido pelo [cliente](#customer), enquanto `total_rounds` e `num_clients` dão parâmetros definidos pelo programa.

### Componentes específicos de plataformas

Os **componentes específicos de plataformas** são os componentes fornecidos por uma [plataforma](#platform) que implementa as interfaces abstratas definidas pela biblioteca de programa federado do TFF.

### Componentes independentes de plataforma

Os **componentes independentes de plataforma** são os componentes fornecidos por uma [biblioteca](#library) (por exemplo, o TFF) que implementa as interfaces abstratas definidas pela biblioteca de programa federado do TFF.

### Computações

As **computações** são implementações da interface abstrata [`tff.Computation`](https://www.tensorflow.org/federated/api_docs/python/tff/Computation).

Por exemplo, na plataforma do TFF, é possível usar os decoradores [`tff.tf_computation`](https://www.tensorflow.org/federated/api_docs/python/tff/tf_computation) ou [`tff.federated_computation`](https://www.tensorflow.org/federated/api_docs/python/tff/federated_computation) para criar uma [`tff.framework.ConcreteComputation`](https://www.tensorflow.org/federated/api_docs/python/tff/framework/ConcreteComputation):

Confira mais informações em [Ciclo de vida de uma computação](https://github.com/tensorflow/federated/blob/main/docs/design/life_of_a_computation.md).

### Lógica do programa

A **lógica do programa** é uma função do Python que recebe como entrada:

- [parâmetros](#parameters) definidos pelo [cliente](#customer) e pelo [programa](#program)
- [componentes específicos de plataformas](#platform-specific-components)
- [componentes independentes de plataforma](#platform-agnostic-components)
- [computações](#computations)

e realiza operações, que costumam incluir:

- execução de [computações](#computations)
- execução de lógica do Python
- [materializando](#materialize) dados no [armazenamento da plataforma](#platform storage) para:
    - usar em lógicas do Python
    - implementar [tolerância a falhas](#fault tolerance)

e pode gerar algumas saídas, que costumam incluir:

- [liberação](#release) de dados para o [armazenamento do cliente](#customer storage) como [métricas](#metrics)

Por exemplo:

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

## Funções

Existem três **funções** úteis em programas federados: [cliente](#customer), [plataforma](#platform) e [biblioteca](#library). Cada uma dessas funções é proprietária e cria alguns [componentes](#components) usados para criar um programa federado. Porém, é possível que uma mesma entidade ou grupo tenha várias funções.

### Cliente

Tipicamente, o **cliente**:

- é proprietário do [armazenamento do cliente](#customer-storage)
- inicia o [programa](#program)

mas também pode:

- criar o [programa](#program)
- realizar qualquer funcionalidade da [plataforma](#platform)

### Plataforma

Tipicamente, a **plataforma**:

- é proprietário do [armazenamento da plataforma](#platform-storage)
- cria [componentes específicos de plataformas](#platform-specific-components)

mas também pode:

- criar o [programa](#program)
- realizar qualquer funcionalidade da [biblioteca](#library)

### Biblioteca

Tipicamente, uma **biblioteca**:

- cria [componentes independentes de plataforma](#platform-agnostic-components)
- cria [computações](#computations)
- cria [lógica de programa](#program-logic)

## Conceitos

Veja alguns **conceitos** úteis sobre programas federados.

### Armazenamento do cliente

**Armazenamento do cliente** é um armazenamento ao qual o [cliente](#customer) tem acesso de leitura e gravação e ao qual a [plataforma](#platform) tem acesso de gravação.

### Armazenamento da plataforma

**Armazenamento da plataforma** é um armazenamento ao qual somente a [plataforma](#platform) tem acesso de leitura e gravação.

### Liberação

**Liberar** um valor o disponibiliza para o [armazenamento do cliente](#customer-storage) (por exemplo, publicar o valor em um painel, registrar o valor em log ou gravar o valor em disco).

### Materializar

A **materialização** de uma referência do valor disponibiliza o valor referenciado para o [programa](#program). Em geral, é necessário materializar uma referência do valor para [liberar](#release) o valor ou fornecer à [lógica do programa](#program-logic) uma [tolerância a falhas](#fault-tolerance).

### Tolerância a falhas

**Tolerância a falhas** é a capacidade de a [lógica do programa](#program-logic) recuperar-se de uma falha ao executar uma computação. Por exemplo, se você treinar as primeiras 90 de 100 rodadas e depois houver uma falha, a lógica do programa consegue retomar o treinamento a partir da rodada 91 ou o treinamento precisa recomeçar da rodada 1?
