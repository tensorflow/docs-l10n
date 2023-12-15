# Soluções em nuvem para o TFX

Procurando insights sobre como o TFX pode ser aplicado para construir uma solução que atenda às suas necessidades? Esses artigos e guias detalhados podem ajudar!

Observação: estes artigos discutem soluções completas nas quais o TFX é uma parte fundamental, mas não a única. Este é quase sempre o caso em implantações no mundo real. Portanto, implementar você mesmo essas soluções exigirá mais do que apenas TFX. O objetivo principal é fornecer algumas dicas sobre como outras pessoas implementaram soluções que podem atender a requisitos semelhantes aos seus, e não servir como um livro de receitas ou lista de aplicativos aprovados do TFX.

## Arquitetura de um sistema de aprendizado de máquina para correspondência de itens em tempo quase real

Use este documento para saber mais sobre a arquitetura de uma solução de aprendizado de máquina (ML - Machine Learning) que aprende e fornece embeddings de itens. Os embeddings podem ajudar você a entender quais itens seus clientes consideram semelhantes, o que permite oferecer sugestões de "itens semelhantes" em tempo real no seu aplicativo. Esta solução mostra como identificar músicas semelhantes num dataset e como usar essas informações para fazer recomendações de músicas. <a href="https://cloud.google.com/solutions/real-time-item-matching" class="external" target="_blank">Leia mais</a>

## Pré-processamento de dados para aprendizado de máquina: opções e recomendações

Este artigo em duas partes explora o tópico da engenharia de dados e engenharia de características para o aprendizado de máquina (ML). Esta primeira parte discute as práticas recomendadas de pré-processamento de dados num pipeline de aprendizado de máquina no Google Cloud. O artigo foca no uso do TensorFlow e da biblioteca de código aberto TensorFlow Transform (tf.Transform) para preparar dados, treinar o modelo e servir o modelo para previsão. Esta parte destaca os desafios do pré-processamento de dados para aprendizado de máquina e ilustra as opções e cenários para realizar a transformação de dados no Google Cloud de maneira eficaz. <a href="https://cloud.google.com/solutions/machine-learning/data-preprocessing-for-ml-with-tf-transform-pt1" class="external" target="_blank">Parte 1</a> <a href="https://cloud.google.com/solutions/machine-learning/data-preprocessing-for-ml-with-tf-transform-pt2" class="external" target="_blank">Parte 2</a>

## Arquitetura para o MLOps usando TFX, Kubeflow Pipelines e Cloud Build

Este documento descreve a arquitetura geral de um sistema de aprendizado de máquina (ML) usando bibliotecas TensorFlow Extended (TFX). Ele também discute como configurar integração contínua (CI), entrega contínua (CD) e treinamento contínuo (CT) para o sistema de ML usando Cloud Build e Kubeflow Pipelines. <a href="https://cloud.google.com/solutions/machine-learning/architecture-for-mlops-using-tfx-kubeflow-pipelines-and-cloud-build" class="external" target="_blank">Leia mais</a>

## MLOps: entrega contínua e pipelines de automação em aprendizado de máquina

Este documento discute técnicas para implementar e automatizar integração contínua (CI), entrega contínua (CD) e treinamento contínuo (CT) para sistemas de aprendizado de máquina (ML). A ciência de dados e o ML estão se tornando recursos essenciais para resolver problemas complexos do mundo real, transformar indústrias e agregar valor em todos os domínios. <a href="https://cloud.google.com/solutions/machine-learning/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning" class="external" target="_blank">Leia mais</a>

## Como configurar um ambiente MLOps no Google Cloud

Este guia de referência descreve a arquitetura de um ambiente de operações de aprendizado de máquina (MLOps) no Google Cloud. **O guia vem com laboratórios práticos** no GitHub que orientam você no processo de provisionamento e configuração do ambiente descrito aqui. Praticamente todas as indústrias estão adotando o aprendizado de máquina (ML) num ritmo cada vez mais acelerado. Um desafio importante para obter valor do ML é criar maneiras de implantar e operar sistemas de ML de maneira eficaz. Este guia destina-se a engenheiros de aprendizado de máquina (ML - Machine Learning) e DevOps. <a href="https://cloud.google.com/solutions/machine-learning/setting-up-an-mlops-environment" class="external" target="_blank">Leia mais</a>

## Requisitos principais para uma fundação MLOps

As organizações orientadas por IA estão usando dados e aprendizado de máquina para resolver seus problemas mais difíceis e já estão colhendo os frutos.

*“As empresas que absorverem totalmente a IA nos seus workflows de produção de valor até 2025 dominarão a economia mundial de 2030, com um crescimento de fluxo de caixa de +120%”,* de acordo com o McKinsey Global Institute.

Mas não é fácil agora. Os sistemas de aprendizado de máquina (ML) têm uma capacidade especial de criar dívida técnica se não forem bem geridos. <a href="https://cloud.google.com/blog/products/ai-machine-learning/key-requirements-for-an-mlops-foundation" class="external" target="_blank">Leia mais</a>

## Como criar e implantar um cartão modelo (model card) na nuvem com Scikit-Learn

Modelos de aprendizado de máquina agora estão sendo usados ​​para realizar muitas tarefas desafiadoras. Com seu vasto potencial, os modelos de ML também levantam questões sobre seu uso, construção e limitações. Documentar as respostas a essas perguntas ajuda a trazer clareza e compreensão compartilhada. Para ajudar a alcançar esses objetivos, o Google introduziu cartões modelo (model cards). <a href="https://cloud.google.com/blog/products/ai-machine-learning/create-a-model-card-with-scikit-learn" class="external" target="_blank">Leia mais</a>

## Análise e validação de dados em escala para aprendizado de máquina com o TensorFlow Data Validation

Este documento discute como usar a biblioteca TensorFlow Data Validation (TFDV) para exploração de dados e análise descritiva durante a experimentação. Cientistas de dados e engenheiros de aprendizado de máquina (ML) podem usar o TFDV num sistema de ML em produção para validar dados usados ​​num pipeline de treinamento contínuo (CT) e para detectar desvios e outliers nos dados recebidos para serviço de previsão. Inclui **laboratórios práticos**. <a href="https://cloud.google.com/solutions/machine-learning/analyzing-and-validating-data-at-scale-for-ml-using-tfx" class="external" target="_blank">Leia mais</a>
