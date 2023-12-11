# Dialetos do MLIR

## Visão geral

Para separar diferentes alvos de hardware e software, o MLIR tem "dialetos", incluindo:

- TensorFlow IR, que representa tudo o que é possível nos grafos do TensorFlow.
- XLA HLO IR, que foi criado para aproveitar as capacidades de compilação do XLA (com saída para, entre outros, TPUs).
- Um dialeto affine experimental, que foca em otimizações e [representações poliédricas](https://en.wikipedia.org/wiki/Polytope_model).
- LLVM IR, que tem um mapeamento 1:1 com a própria representação do LLVM, permitindo que o MLIR emita o código de GPU e CPU pelo LLVM.
- TensorFlow Lite, que traduzirá para código de execução em plataformas móveis.

Cada dialeto consiste em um conjunto de operações definidas que têm invariantes colocadas, como: "Isso é um operador binário, e as entradas e saídas têm os mesmos tipos".

## Somando ao MLIR

O MLIR não tem uma lista fixa/integrada de operações conhecidas globalmente (nenhum "intrínseco"). Os dialetos podem definir tipos inteiramente personalizados, que é como o MLIR pode modelar algo como o sistema de tipo IR do LLVM (que tem agregados de primeira classe), abstrações de domínio importantes para aceleradores otimizados por ML como tipos quantizados e até os sistemas de tipo Swift ou Clang (que foram criados com base em nós de declaração Swift/Clang) no futuro.

Se você quiser conectar um compilador de baixo nível, deve criar um novo dialeto e os rebaixamentos entre o dialeto do grafo do TensorFlow e seu dialeto. Isso suaviza o caminho para os fabricantes de compiladores e hardware. Você pode até segmentar dialetos em diferentes níveis no mesmo modelo; os otimizados de nível superior vão respeitar as partes desconhecidas do IR e aguardar um nível inferior lidar com elas.
