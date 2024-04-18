# MLIR

## Visão geral

O MLIR, ou Multi-Level Intermediate Representation (Representação Intermediária Multinível), é um formato de representação e uma biblioteca de utilitários de compilador que fica entre a representação do modelo e compiladores/executores de baixo nível que geram código específico de hardware.

O MLIR é, em sua essência, uma infraestrutura flexível para compiladores de otimização modernos. Isso significa que ele consiste em uma especificação para representações intermediárias (IR) e um kit de ferramentas de código para realizar transformações nessa representação. (Na linguagem do compilador, à medida que você passa de representações de nível superior para representações de nível inferior, essas transformações podem ser chamadas de "lowerings", ou rebaixamentos.)

O MLIR é altamente influenciado pelo [LLVM](https://llvm.org/) e reutiliza descaradamente muitas grandes ideias dele. Possui um sistema de tipos flexível e permite representar, analisar e transformar grafos combinando vários níveis de abstração em uma mesma unidade de compilação. Essas abstrações incluem operações do TensorFlow, regiões de loop poliédrico aninhadas e até instruções LLVM e operações e tipos de hardware fixos.

Esperamos que o MLIR seja do interesse de muitos grupos, incluindo:

- Pesquisadores e implementadores de compiladores que buscam otimizar o desempenho e o consumo de memória de modelos de aprendizado de máquina
- Fabricantes de hardware que procuram uma maneira de conectar seu hardware ao TensorFlow, como TPUs, hardware neural portátil em smartphones e outros ASICs personalizados
- Pessoas que escrevem vinculações de nomes e desejam aproveitar a otimização de compiladores e a aceleração de hardware

O ecossistema TensorFlow contém vários compiladores e otimizadores que operam em vários níveis da pilha de software e hardware. Esperamos que a adoção gradual do MLIR simplifique todos os aspectos dessa pilha.

<img src="./images/mlir-infra.svg" alt="Diagrama de visão geral do MLIR">
