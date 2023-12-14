# MLIR

## Descripción general

MLIR, o el lenguaje intermedio/intermediario multinivel, es un formato de representación y una biblioteca de utilidades de compiladores que se encuentra entre la representación del modelo y los ejecutores o compiladores de bajo nivel que generan código específico para el hardware.

MLIR, en esencia, es una infraestructura flexible para compiladores de optimización modernos. Significa que está compuesto por una especificación para lenguaje intermedio (IR) y por un kit de herramientas de código para realizar transformaciones en ese lenguaje. (En la jerga del compilador, a medida que uno se mueve de lenguaje de alto nivel a representaciones de bajo nivel, estas transformaciones se denominan “bajadas (<em>lowerings</em>)”)

[LLVM](https://llvm.org/) tiene gran influencia en MLIR y, abiertamente, se aprovecha de muchas de sus buenas ideas. Tiene un sistema de tipo flexible y permite representar, analizar y transformar grafos mediante la combinación de varios niveles de abstracción en la misma unidad de compilación. Estas abstracciones incluyen las operaciones en TensorFlow, las regiones de ciclos poliédricos anidados e, incluso, las instrucciones de LLVM y sus operaciones y tipos de hardware fijos.

Esperamos que MLIR resulte de interés para muchos grupos, incluidos los siguientes:

- Los investigadores e implementadores de compiladores que quieran optimizar el desempeño y el consumo de memoria para los modelos de aprendizaje automático.
- Los fabricantes de hardware que buscan una forma de conectar su hardware con TensorFlow; nos referimos al hardware como las TPU, el hardware neuronal portátil en teléfonos y otros ASIC personalizados.
- Personas que escriben enlaces entre lenguajes que además, quieran aprovecharlo para la optimización de los compiladores y la aceleración del hardware.

El ecosistema de TensorFlow contiene varios compiladores y optimizadores que funcionan a muchos niveles diferentes en software y hardware. Esperamos que la adopción gradual de MLIR simplifique todos los aspectos de esta tecnología.

<img src="./images/mlir-infra.svg" alt="Diagrama general de MLIR">
