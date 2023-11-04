# Ejecución

[TOC]

El paquete de [ejecutores](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors) contiene clases esenciales (<em>core</em>) de [ejecutores](#executor) y la funcionalidad relativa del [tiempo de ejecución](#runtime).

## Tiempo de ejecución

Un tiempo de ejecución es un concepto lógico con el que se describe un sistema que ejecuta un cálculo.

### Tiempo de ejecución de TFF

Un tiempo de ejecución de TFF normalmente gestiona la ejecución de un [AST](compilation.md#ast) y delega la ejecución de cálculos a un [tiempo de ejecución externo](#external-runtime), como [TensorFlow](#tensorflow).

### Tiempo de ejecución externo

Un tiempo de ejecución externo es cualquier sistema al que el tiempo de ejecución de TFF le delega dicha ejecución.

#### TensorFlow

[TensorFlow](https://www.tensorflow.org/) es una plataforma de código abierto para el aprendizaje automático. Hoy en día el tiempo de ejecución de TFF delega cálculos matemáticos a TensorFlow con un [ejecutor](#Executor) que se puede componer con una jerarquía, a la que se consulte como [pila de ejecución](#execution-stack).

## `Executor`

Un [executor_base.Executor](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_base.py) es una interfaz abstracta que define la API para ejecutar un [AST](compilation.md#ast). El paquete de [ejecutores](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors) contiene una colección de implementaciones concretas de esta interfaz.

## `ExecutorFactory`

Un [executor_factory.ExecutorFactory](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_factory.py) es una interfaz abstracta que define la API para construir un [ejecutor](#executor). Estas factorías lo construyen lentamente y gestionan su ciclo de vida. El motivo de la construcción lenta de ejecutores es la de inferir la cantidad de clientes en el tiempo de ejecución.

## Pila de ejecución

Una pila de ejecución es una jerarquía de [ejecutores](#executor). El paquete [executor_stacks](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executor_stacks) contiene la lógica para construir y componer pilas de ejecuciones específicas.
