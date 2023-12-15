# Arquitetura XLA

<div style="width:50%; margin:auto; margin-bottom:10px; margin-top:20px;"> <img style="width:50%" src="./images/xlalogo.png">
</div>

## Por que criamos o XLA?

Tínhamos diversos objetivos para o XLA trabalhar com o TensorFlow:

- *Melhorar a velocidade de execução.* Compilar subgrafos para reduzir o tempo de execução de operações de curta duração visando eliminar a sobrecarga do runtime do TensorFlow, fundir operações em pipeline para reduzir a sobrecarga de memória e especializar-se em formas de tensor conhecidas para permitir uma propagação constante mais agressiva.

- *Melhorar o uso da memória.* Analisar e programar o uso da memória, eliminando, em princípio, muitos buffers de armazenamento intermediários.

- *Reduzir a dependência de operações personalizadas.* Eliminar a necessidade de diversas operações personalizadas, melhorando o desempenho de operações de baixo nível fundidas automaticamente para corresponder ao desempenho de operações personalizadas que foram fundidas manualmente.

- *Reduzir o espaço ocupado em dispositivos móveis.* Eliminar o tempo de execução do TensorFlow compilando antecipadamente o subgrafo e emitindo um par de arquivo objeto/cabeçalho que pode ser vinculado diretamente a outro aplicativo. Os resultados podem reduzir o espaço ocupado pela inferência móvel em várias ordens de grandeza.

- *Melhorar a portabilidade.* Tornar relativamente fácil escrever um novo back-end para hardware novo, ponto em que uma grande parte dos programas do TensorFlow será executada sem modificações nesse hardware. Isto contrasta com a abordagem de especialização de operações monolíticas individuais para novo hardware, que exige que os programas do TensorFlow sejam reescritos para fazer uso dessas operações.

## Como funciona o XLA?

A linguagem de entrada para o XLA é chamada de "HLO IR", ou apenas HLO (High Level Operations). A semântica do HLO está descrita na página [Operation Semantics](./operation_semantics.md). É mais conveniente pensar no HLO como um [compilador IR](https://en.wikipedia.org/wiki/Intermediate_representation).

XLA recebe grafos ("computações") definidos em HLO e os compila para instruções de máquina de várias arquiteturas. O XLA é modular no sentido de que é fácil inserir um back-end alternativo visando [alguma nova arquitetura de HW](./developing_new_backend.md). O back-end da CPU para x64 e ARM64, bem como o back-end da GPU NVIDIA estão na árvore de fontes do TensorFlow.

O diagrama a seguir mostra o processo de compilação em XLA:

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">   <img src="./images/how-does-xla-work.png">
</div>

O XLA vem com diversas otimizações e passagens de análise independentes do alvo, como [CSE](https://en.wikipedia.org/wiki/Common_subexpression_elimination), fusão de operações independentes do alvo e análise de buffer para alocar memória de tempo de execução para a computação.

Depois dessa etapa independente do alvo, o XLA envia a computação HLO para um backend. O back-end pode realizar otimizações adicionais no nível do HLO, desta vez com informações e necessidades específicas do alvo. Por exemplo, o back-end da GPU XLA pode realizar fusão de operações benéfica especificamente para o modelo de programação da GPU e determinar como particionar a computação em fluxos. Neste estágio, os back-ends também podem combinar padrões de certas operações ou combinações delas com chamadas de biblioteca otimizadas.

A próxima etapa é a geração de código específico do destino. Os back-ends de CPU e GPU incluídos no XLA usam [LLVM](http://llvm.org) para IR de baixo nível, otimização e geração de código. Esses backends produzem o IR LLVM necessário para representar a computação XLA HLO de maneira eficiente e, em seguida, invocam o LLVM para emitir código nativo deste IR LLVM.

O back-end da GPU atualmente oferece suporte a GPUs NVIDIA por meio do back-end LLVM NVPTX; o back-end da CPU oferece suporte a vários ISAs de CPU.
