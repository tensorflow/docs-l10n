# Compilación

[TOC]

El paquete [compilador](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler) contiene estructuras de datos que definen la representación de Python del [AST](#ast), las funciones esenciales de [transformación](#transformation) y la funcionalidad relacionada del [compilador](#compiler).

## AST

Con un árbol de sintaxis abstracta (AST) en TFF se describe la estructura de un cálculo federado.

### Bloque de construcción

Un [building_block.ComputationBuildingBlock](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) es la representación en Python del [AST](#ast).

#### `CompiledComputation`

Un [building_block.CompiledComputation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) es un [building_block.ComputationBuildingBlock](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) que representa un cálculo que se delegará a un [tiempo de ejecución externo](execution.md#external-runtime). Actualmente, TFF solamente es compatible con [cálculos de TensorFlow](#tensorFlow-computation), pero se podría expandir para ser compatible con [cálculos](#computation) respaldados por otros tiempos de ejecución externos.

### `Computation`

Un [pb.Computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/computation.proto) es la representación serializada o el prototipo del [AST](#ast).

#### Cálculo de TensorFlow

Un [pb.Computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/computation.proto) que representa [cálculos](#computation) que se delegarán al tiempo de ejecución de [TensorFlow](execution.md#tensorflow).

## Transformación

Una transformación construye un [AST](#ast) nuevo para un AST dado después de aplicar una colección de mutaciones. Las transformaciones pueden operar en [bloques de construcción](#building-block), a fin de transformar la representación Python del AST o en los [cálculos de TensorFlow](#tensorFlow-computation) para transformar un `tf.Graph`.

Una transformación **atómica** es una transformación que aplica una sola mutación (posiblemente, más de una vez) a una entrada dada.

Una transformación **compuesta** es la aplicación de múltiples transformaciones a la entrada dada, a fin de proporcionar alguna característica o afirmación.

Nota: Las transformaciones se pueden componer en serie o en paralelo, es decir, se puede construir una transformación compuesta que realice varias transformaciones en un pase a través de un AST. Sin embargo, el orden en que se apliquen esas transformaciones y el modo en que se paralelicen no es fácil de razonar. En consecuencia, las transformaciones compuestas se trabajan a mano y la mayoría son, en cierto modo, frágiles.

El módulo [tree_transformations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/tree_transformations.py) contiene transformaciones atómicas del [bloque de construcción](#building-block).

El módulo [transformations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/transformations.py) contiene transformaciones compuestas del [bloque de construcción](#building-block).

El módulo [tensorflow_computation_transformations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/tensorflow_computation_transformations.py) contiene transformaciones atómicas del [cálculo de TensorFlow](#tensorflow-computation).

El módulo [compiled_computation_transformations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/compiled_computation_transformations.py) contiene transformaciones atómicas y compuestas del [cálculo compilado](#compiled-computation).

El módulo [transformation_utils](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/transformation_utils.py) contiene funciones, lógica de recorrido (<em>traversal</em>) y estructuras de datos usadas por otros módulos de transformación.

## Compilador

Un compilador es una colección de [transformaciones](#transformation) que construyen una forma que se puede ejecutar.
