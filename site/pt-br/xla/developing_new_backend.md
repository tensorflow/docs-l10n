# Desenvolvendo um novo backend para o XLA

Este guia preliminar é para os primeiros usuários que queiram redirecionar facilmente o TensorFlow para seu hardware de maneira eficiente. O guia não é passo a passo e pressupõe conhecimentos de [LLVM](http://llvm.org), [Bazel](https://bazel.build/) e TensorFlow.

O XLA fornece uma interface abstrata que uma nova arquitetura ou acelerador pode implementar para criar um back-end para executar grafos do TensorFlow. O redirecionamento do XLA deve ser significativamente mais simples e escalonável do que implementar todas as operações TensorFlow existentes para o novo hardware.

A maioria das implementações se enquadrará num dos seguintes cenários:

1. Arquitetura de CPU existente ainda não suportada oficialmente pelo XLA, com ou sem um back-end [LLVM](http://llvm.org) existente.
2. Hardware não similar a CPU com um back-end LLVM existente.
3. Hardware não similar a CPU sem um back-end LLVM existente.

> Observação: um back-end LLVM pode significar um dos back-ends LLVM lançados oficialmente ou um back-end LLVM personalizado desenvolvido internamente.

## Cenário 1: Arquitetura de CPU existente ainda não suportada oficialmente pelo XLA

Neste cenário, comece examinando o [back-end da CPU XLA](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/cpu/) existente. O XLA facilita o redirecionamento do TensorFlow para CPUs diferentes usando LLVM, já que a principal diferença entre back-ends XLA para CPUs é o código gerado pelo LLVM. Google testa o XLA para arquiteturas x64 e ARM64.

Se o fornecedor de hardware tiver um backend LLVM para seu hardware, é simples vincular o backend ao LLVM criado com XLA. No modo JIT, o backend da CPU XLA produz código para a CPU host. Para compilação antecipada, [`xla::AotCompilationOptions`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/compiler.h) pode fornecer um LLVM triplo para configurar a arquitetura de destino.

Se não houver back-end LLVM existente, mas existir outro tipo de gerador de código, deverá ser possível reutilizar a maior parte do back-end da CPU existente.

## Cenário 2: Hardware não similar a CPU com um back-end LLVM existente

É possível modelar uma nova implementação de [`xla::Compiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/compiler.h) nas classes [`xla::CPUCompiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/cpu/cpu_compiler.cc) e [`xla::GPUCompiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/gpu/nvptx_compiler.cc) existentes, uma vez que estas já produzem IR LLVM. Dependendo da natureza do hardware, é possível que muitos dos aspectos de geração de IR LLVM tenham que ser alterados, mas muito código pode ser compartilhado com os back-ends existentes.

Um bom exemplo a seguir é o [back-end da GPU](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/gpu/) do XLA. O back-end da GPU tem como alvo um ISA diferente da CPU e, portanto, alguns aspectos de sua geração de código são exclusivos do domínio da GPU. Outros tipos de hardware, por exemplo, DSPs como o Hexagon (que possui um backend LLVM upstream), podem reutilizar partes da lógica de emissão IR do LLVM, mas outras partes serão exclusivas.

## Cenário 3: Hardware não similar a CPU sem um back-end LLVM existente

Caso não seja possível utilizar o LLVM, a melhor opção é implementar um novo backend para o XLA para o hardware desejado. Essa opção requer mais esforço. As classes que precisam ser implementadas são as seguintes:

- [`StreamExecutor`](https://www.tensorflow.org/code/tensorflow/compiler/xla/stream_executor/stream_executor.h): para muitos dispositivos, nem todos os métodos de `StreamExecutor` são necessários. Consulte as implementações existentes `StreamExecutor` para mais detalhes.
- [`xla::Compiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/compiler.h): esta classe encapsula a compilação de uma computação HLO em um `xla::Executable`.
- [`xla::Executable`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/executable.h): esta classe é usada para lançar uma computação compilada na plataforma.
- [`xla::TransferManager`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/transfer_manager.h): esta classe permite que back-ends forneçam mecanismos específicos de plataforma para construir dados literais XLA a partir de determinados identificadores de memória de dispositivo. Em outras palavras, ajuda a encapsular a transferência de dados do host para o dispositivo e vice-versa.
