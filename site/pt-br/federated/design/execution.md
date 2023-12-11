# Execução

[TOC]

O pacote [executors](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors) (executores) contém classes core dos [executores](#executor) e funcionalidades relacionadas ao [runtime](#runtime).

## Runtime

O runtime é um conceito lógico que descreve um sistema que executa uma computação.

### Runtime do TFF

Tipicamente, um runtime do TFF executa uma [AST](compilation.md#ast) e delega a execução de computações matemáticas a um [runtime externo](#external-runtime), como o [TensorFlow](#tensorflow).

### Runtime externo

Um runtime externo é qualquer sistema ao qual o runtime do TFF delega a execução.

#### TensorFlow

O [TensorFlow](https://www.tensorflow.org/) é uma plataforma de código aberto para aprendizado de máquina. Atualmente, o runtime do TFF delega as computações matemáticas para o TensorFlow usando um [executor](#Executor) que pode ser composto em uma hierarquia, chamada de [pilha de execução](#execution-stack).

## `Executor`

Um [executor_base.Executor](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_base.py) é uma interface abstrata que define a API para execução de uma [AST](compilation.md#ast). O pacote [executors](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors) contém um conjunto de implementações concretas dessa interface.

## `ExecutorFactory`

Uma [executor_factory.ExecutorFactory](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_factory.py) (fábrica de executores) é uma interface abstrata que define a API para construir um [executor](#executor). Essas fábricas constroem o executor de maneira lenta (lazy) e gerencia o ciclo de vida do executor. O motivo para a construção lazy é inferir o número de clientes no momento da execução.

## Pilha de execução

Uma pilha de execução é uma hierarquia de [executores](#executor). O pacote [executor_stacks](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executor_stacks) (pilhas de execução) contém lógica para construir e compor pilhas de execução específicas.
