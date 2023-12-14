# Dialectos de MLIR

## Descripción general

Para separar destinos de hardware y software diferentes, MLIR tiene “dialectos”, entre los que se incluyen:

- TensorFlow IR, que representa a todas las cosas que puede haber en los grafos de TensorFlow.
- XLA HLO IR, que está diseñado para aprovechar la capacidad de compilación de XLA (con salida a las TPU, entre otras cosas).
- Un dialecto experimental afín, que se centra en las [representaciones poliédricas](https://en.wikipedia.org/wiki/Polytope_model) y en las optimizaciones.
- LLVM IR, que tiene un mapeo individual entre la propia representación de LLVM. Esto le permite a MLIR emitir código para GPU y CPU a través de LLVM.
- TensorFlow Lite, que traducirá al código de ejecución en plataformas móviles.

Cada dialecto está compuesto por un conjunto de operaciones definidas, con invariantes incluidas, como: “Es un operador binario, y las entradas y salidas tienen los mismos tipos”.

## Adiciones a MLIR

MLIR no tiene una lista integrada ni fija de operaciones globalmente conocidas (no tiene “funciones intrínsecas”). Los dialectos pueden definir por completo los tipos personalizados. Es el motivo por el que MLIR puede modelar cosas como el sistema de tipos LLVM IR (que tiene agregados de primera clase), las abstracciones de dominios importantes para los aceleradores de ML optimizados como los tipos cuantificados e, incluso, los sistemas de tipos Swift/Clang (que se crean en torno a nodos de declaración Swift/Clang), en el futuro.

Si desea conectar un compilador nuevo de bajo nivel, le convendrá crear un dialecto nuevo y las bajadas entre el dialecto del grafo de TensorFlow y el suyo. De este modo, les facilitará el camino a los creadores de compiladores y hardware. Incluso es posible tener dialectos como objetivo a distintos niveles en el mismo modelo; los optimizadores de alto nivel respetarán las partes desconocidas del IR y esperarán que un nivel más bajo se ocupe.
