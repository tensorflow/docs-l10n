**Atualizado em: junho de 2021**

O Kit de ferramentas para otimização de modelos (MOT) foi amplamente usado para converter/otimizar modelos do TensorFlow em modelos do TensorFlow Lite com tamanhos menores, melhor desempenho e exatidão aceitável, para que sejam executados em dispositivos móveis e de IoT. Estamos trabalhando em ampliar as técnicas e ferramentas do MOT além do TensorFlow Lite, também oferecendo suporte ao SavedModel do TensorFlow.

Esta é uma visão geral dos desenvolvimentos futuros, que podem ser alterados a qualquer momento. A ordem abaixo não indica nenhum tipo de prioridade. Incentivamos você a comentar sobre eles e fornecer feedback no [grupo de discussão](https://groups.google.com/a/tensorflow.org/g/tflite).

## Quantização

#### TensorFlow Lite

- Quantização pós-treinamento seletiva para excluir determinadas camadas da quantização.
- Depurador de quantização para inspecionar as perdas de erro de quantização por camada.
- Aplicação de treinamento com reconhecimento de quantização em mais modelos, como o TensorFlow Model Garden.
- Melhorias de qualidade e desempenho para quantização pós-treinamento de intervalo dinâmico.

#### TensorFlow

- Quantização pós-treinamento (intervalo dinâmico de bf16 * int8).
- Treinamento consciente de quantização (bf16 * int8 somente pesos com quantização falsa).
- Quantização pós-treinamento seletiva para excluir determinadas camadas da quantização.
- Depurador de quantização para inspecionar as perdas de erro de quantização por camada.

## Esparsidade

#### TensorFlow Lite

- Suporte à execução de modelos esparsos para mais modelos.
- Criação consciente de alvo para a esparsidade.
- Ampliação do conjunto de operações esparsas com kernels x86 eficazes.

#### TensorFlow

- Suporte à esparsidade no TensorFlow.

## Técnicas de compressão em cascata

- Quantização + Compressão de tensores + Esparsidade: demonstração de todas as 3 técnicas trabalhando juntas.

## Compressão

- API de compressão de tensores para ajudar os desenvolvedores de algoritmos de compressão a implementar o próprio algoritmo de compressão de modelo (por exemplo, clustering de peso), inclusive oferecendo uma forma padrão de teste/benchmarking.
