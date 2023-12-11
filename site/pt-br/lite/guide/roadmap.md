# Desenvolvimentos futuros do TensorFlow Lite

**Atualizado em: maio de 2021**

Esta é uma visão geral dos desenvolvimentos futuros, que podem ser alterados a qualquer momento. A ordem abaixo não indica nenhum tipo de prioridade.

Detalhamos os desenvolvimentos futuros em quatro segmentos principais: usabilidade, desempenho, otimização e portabilidade. Recomendamos que faça comentários sobre os desenvolvimentos futuros e forneça feedback no [grupo de discussão do TensorFlow Lite](https://groups.google.com/a/tensorflow.org/g/tflite).

## Usabilidade

- **Expansão da cobertura de operações**
    - Inclusão de operações seletas com base em feedback dos usuários.
    - Inclusão de conjuntos seletos de operações para domínios e áreas específicos, incluindo operações aleatórias, operações de camadas do Keras base, tabelas de hash, operações seletas de treinamento.
- **Conjunto de ferramentas mais assistivas**
    - Fornecimento de ferramentas de compatibilidade e anotação de grafos ao TensorFlow para validar a compatibilidade entre o TF Lite e o acelerador de hardware durante o treinamento e após a conversão.
    - Possibilidade de escolha e otimização para aceleradores específicos durante a conversão.
- **Treinamento no dispositivo**
    - Suporte ao treinamento no dispositivo para personalização e aprendizado por transferência, incluindo um Colab demonstrando o uso do início ao fim.
    - Suporte a tipos de variável/recurso (tanto para inferência quanto treinamento).
    - Suporte à conversão e execução de grafos com diversos pontos de entrada de função (ou assinatura).
- **Integração aprimorada com o Android Studio**
    - Arraste e solte modelos do TF Lite no Android Studio para gerar interfaces do modelo.
    - Melhoria do suporte ao profiling do Android Studio, incluindo profiling de memória.
- **Model Maker (criador de modelos)**
    - Suporte a novas tarefas, incluindo detecção de objetos, recomendações e classificação de áudio, abrangendo uma grande gama de usos comuns.
    - Suporte a mais datasets para facilitar o aprendizado por transferência.
- **Biblioteca Task**
    - Suporte a mais tipos de modelo (por exemplo: áudio, NLP) com funcionalidades de pré e pós-processamento.
    - Atualização de mais exemplos de referência com APIs de tarefas.
    - Suporte integrado à aceleração para todas as tarefas.
- **Mais modelos e exemplos de última geração**
    - Inclusão de mais exemplos (como áudio, NLP e relacionados a dados estruturados) para demonstrar o uso do modelo, além de novos recursos e APIs, abrangendo diferentes plataformas.
    - Criação de modelos backbone compartilháveis para ML no dispositivo a fim de reduzir os custos de treinamento e implantação.
- **Implantação simplificada em múltiplas plataformas**
    - Execução de modelos do TensorFlow Lite na web.
- **Melhoria do suporte interplataforma**
    - Melhoria e expansão de APIs para Java no Android, Swift no iOS, Python no RPi.
    - Melhoria de suporte ao CMake (como suporte mais amplo a aceleradores).
- **Suporte melhor ao front-end**
    - Melhoria da compatibilidade com diversos front-ends de autoração, incluindo Keras e tf.numpy.

## Desempenho

- **Conjunto de ferramentas melhores**
    - Painel de ferramentas público para acompanhar os ganhos de desempenho a cada versão.
    - Conjunto de ferramentas para entender melhor a compatibilidade do grafo com os aceleradores desejados.
- **Melhor desempenho em CPUs**
    - XNNPack ativado por padrão para inferência mais rápida com ponto flutuante.
    - Suporte à meia precisão (float16) fim a fim com kernels otimizados.
- **Atualização do suporte à NNAPI**
    - Suporte completo aos recursos, operações e tipos mais novos da NNAPI para Android.
- **Otimizações para GPUs**
    - Melhoria do tempo de inicialização, com suporte à serialização de delegados.
    - Interoperabilidade de buffer de hardware para inferência sem cópias (zero-copy).
    - Maior disponibilidade de aceleração no dispositivo.
    - Maior cobertura de operações.

## Otimização

- **Quantização**

    - Quantização pós-treinamento seletiva para excluir determinadas camadas da quantização.
    - Depurador de quantização para inspecionar as perdas de erro de quantização por camada.
    - Aplicação de treinamento com reconhecimento de quantização em mais modelos, como o TensorFlow Model Garden.
    - Melhorias de qualidade e desempenho para quantização pós-treinamento de intervalo dinâmico.
    - API Tensor Compression (compressão de tensores) para permitir algoritmos de compressão, como SVD.

- **Pruning/esparsividade**

    - Combinação de APIs de tempo de treinamento configuráveis (pruning + treinamento com reconhecimento de quantização).
    - Aumento da aplicação de esparsividade nos modelos do TF Model Garden.
    - Suporte à execução de modelos esparsos no TensorFlow Lite.

## Portabilidade

- **Suporte a microcontroladores**
    - Inclusão de suporte para diversos casos de uso com arquitetura de 32 bits para classificação de fala e imagem.
    - Front-end de áudio: suporte à aceleração e ao pré-processamento de áudio no grafo.
    - Modelos e código de exemplo para dados de visão e áudio.
