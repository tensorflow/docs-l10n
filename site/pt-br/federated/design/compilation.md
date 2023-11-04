# Compilação

[TOC]

O pacote [compiler](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler) (compilador) contém as estruturas de dados que definem a representação do Python para a [AST](#ast), funções de [transformação](#transformation) do core e funcionalidade relacionada ao [compilador](#compiler).

## AST

No TFF, uma árvore de sintaxe abstrata (AST) descreve a estrutura de uma computação federada.

### Bloco de construção

Um [building_block.ComputationBuildingBlock](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) é a representação de uma [AST](#ast) do Python.

#### `CompiledComputation`

Uma [building_block.CompiledComputation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) (computação compilada) é um [building_block.ComputationBuildingBlock](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) (bloco de construção de computação) que representa uma computação que será delegada a um [runtime externo](execution.md#external-runtime). No momento, o TFF tem suporte somente a [computações do TensorFlow](#tensorFlow-computation), mas poderá ser estendido para ter suporte a [computações](#computation) feitas por outros runtimes externos.

### `Computation`

Uma [pb.Computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/computation.proto) (computação) é o Proto ou a representação serializada da [AST](#ast).

#### Computação do TensorFlow

Uma [pb.Computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/computation.proto) representa uma [computação](#computation) que será delegada para o runtime do [TensorFlow](execution.md#tensorflow).

## Transformação

Uma transformação constrói uma nova [AST](#ast) para uma determinada AST após aplicar uma coleção de mutações. As transformações podem operar em [blocos de construção](#building-block) para transformar a representação da AST do Python ou podem operar em [transformações do TensorFlow](#tensorFlow-computation) para transformar um `tf.Graph`.

Uma transformação **atômica** aplica uma única mutação (possivelmente mais de uma vez) à entrada fornecida.

Uma transformação **composta** aplica várias transformações à entrada fornecida para oferecer algum recurso ou asserção.

Observação: as transformações podem ser compostas em série ou em paralelo, ou seja, você pode construir uma transformação composta que faça diversas transformações em um passo por uma AST. Porém, é difícil definir a ordem de aplicação das transformações e como essas transformações são paralelizadas; como resultado, as transformações compostas são feitas à mão, e a maioria é, de certa forma, frágil.

O módulo [tree_transformations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/tree_transformations.py) (transformações da árvore) contém transformações atômicas de [building block](#building-block) (blocos de construção).

O módulo [transformations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/transformations.py) (transformações) contém transformações compostas de [building block](#building-block) (blocos de construção).

O módulo [tensorflow_computation_transformations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/tensorflow_computation_transformations.py) (transformações de computação do TensorFlow) contém transformações atômicas de [computações do TensorFlow](#tensorflow-computation).

O módulo [compiled_computation_transformations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/compiled_computation_transformations.py) (transformações de computação compilada) contém transformações atômicas e compostas de [computações compiladas](#compiled-computation).

O módulo [transformation_utils](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/transformation_utils.py) (utilitários de transformações) contém funções, lógica transversal e estruturas de dados usadas por outros módulos de transformação.

## Compilador

Um compilador é uma coleção de [transformações](#transformation) que constroem uma forma que pode ser executada.
