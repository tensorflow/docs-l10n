# Cómo desarrollar un nuevo backend para XLA

Esta guía preliminar está dirigida a los primeros usuarios que desean redirigir fácilmente TensorFlow a su hardware de manera eficiente. La guía no es paso a paso y asume conocimientos de [LLVM](http://llvm.org), [Bazel](https://bazel.build/) y TensorFlow.

XLA proporciona una interfaz abstracta que se puede implementar en una nueva arquitectura o acelerador para crear un backend que ejecute grafos de TensorFlow. La reorientación de XLA debería ser significativamente más simple y escalable que implementar cada operación existente de TensorFlow para el nuevo hardware.

La mayoría de las implementaciones se ajustarán a uno de los siguientes escenarios:

1. Arquitectura de CPU existente que aún no es compatible oficialmente con XLA, con o sin un backend de [LLVM](http://llvm.org) existente.
2. Hardware que no es similar a una CPU con un backend de LLVM existente.
3. Hardware que no es similar a una CPU y sin un backend de LLVM existente.

> Nota: Un backend de LLVM puede significar uno de los backends de LLVM publicados oficialmente o un backend de LLVM personalizado que se haya desarrollado internamente.

## Escenario 1: arquitectura de CPU existente que aún no es compatible oficialmente con XLA

En este escenario, comience por observar el [backend de la CPU de XLA](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/cpu/) existente. XLA facilita la reorientación de TensorFlow a diferentes CPU mediante el uso de LLVM, ya que la principal diferencia entre los backends XLA para CPU es el código generado por LLVM. Google prueba XLA para arquitecturas x64 y ARM64.

Si el proveedor de hardware tiene un backend de LLVM para su hardware, es sencillo vincular el backend con la LLVM creada con XLA. En el modo JIT, el backend de la CPU de XLA emite código para la CPU host. Para una compilación anticipada, [`xla::AotCompilationOptions`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/compiler.h) puede proporcionar un triple LLVM para configurar la arquitectura objetivo.

Si no existe un backend de LLVM, pero existe otro tipo de generador de código, debería ser posible reutilizar la mayor parte del backend de la CPU existente.

## Escenario 2: hardware que no es similar a una CPU con un backend de LLVM existente

Es posible modelar una nueva implementación [`xla::Compiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/compiler.h) en las clases [`xla::CPUCompiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/cpu/cpu_compiler.cc) y [`xla::GPUCompiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/gpu/nvptx_compiler.cc) existentes, ya que estas ya emiten IR de LLVM. En función de la naturaleza del hardware, es posible que haya que cambiar muchos de los aspectos de la generación de IR de LLVM, pero se puede compartir una gran cantidad de código con los backends existentes.

Un buen ejemplo a seguir es el [backend de GPU](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/gpu/) de XLA. El backend de GPU apunta a una ISA que no es similar a una CPU y, por lo tanto, algunos aspectos de su generación de código son exclusivos del dominio de GPU. Otros tipos de hardware, por ejemplo, DSP como Hexagon (que tiene un backend de LLVM ascendente), pueden reutilizar partes de la lógica de emisión de IR de LLVM, pero otras partes serán únicas.

## Escenario 3: hardware que no es similar a una CPU sin un backend de LLVM existente

Si no es posible usar LLVM, entonces la mejor opción es implementar un nuevo backend para XLA para el hardware deseado. Esta opción es la que requiere más esfuerzo. Las clases que se tienen que implementar son las siguientes:

- [`StreamExecutor`](https://www.tensorflow.org/code/tensorflow/compiler/xla/stream_executor/stream_executor.h): para muchos dispositivos no se necesitan todos los métodos de `StreamExecutor`. Consulte las implementaciones existentes de `StreamExecutor` para obtener más información al respecto.
- [`xla::Compiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/compiler.h): esta clase encapsula la compilación de un cálculo de HLO en un `xla::Executable`.
- [`xla::Executable`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/executable.h): esta clase se usa para iniciar un cálculo compilado en la plataforma.
- [`xla::TransferManager`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/transfer_manager.h): esta clase permite que los servidores proporcionen mecanismos específicos de la plataforma para construir datos literales de XLA a partir de identificadores de memoria del dispositivo determinados. En otras palabras, ayuda a encapsular la transferencia de datos desde el host al dispositivo y viceversa.
