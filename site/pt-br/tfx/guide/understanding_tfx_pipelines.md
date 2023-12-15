# Compreendendo os pipelines do TFX

MLOps é a prática de aplicação de práticas DevOps para ajudar a automatizar, gerenciar e auditar workflows de aprendizado de máquina (ML). Os workflows de ML incluem etapas para:

- Preparar, analisar e transformar dados.
- Treinar e avaliar um modelo.
- Implantar modelos treinados em produção.
- Rastrear artefatos de ML e entender suas dependências.

Gerenciar essas etapas de maneira ad hoc pode ser difícil e demorado.

O TFX facilita a implementação de MLOps, fornecendo um kit de ferramentas que ajuda a orquestrar seu processo de ML em diversos orquestradores, tais como: Apache Airflow, Apache Beam e Kubeflow Pipelines. Ao implementar seu workflow como um pipeline TFX, você pode:

- Automatizar seu processo de ML, o que permite treinar, avaliar e implantar regularmente seu modelo.
- Utilizar recursos de computação distribuídos para processar grandes datasets e workloads.
- Aumentar a velocidade da experimentação executando um pipeline com diferentes conjuntos de hiperparâmetros.

Este guia descreve os principais conceitos fundamentais para entender os pipelines do TFX.

## Artefatos

As saídas de cada etapa (passo) de um pipeline TFX são chamadas de **artefatos**. As etapas subsequentes no seu workflow podem usar esses artefatos como entradas. Dessa forma, o TFX permite transferir dados entre etapas do workflow.

Por exemplo, o componente padrão `ExampleGen` emite (produz) exemplos serializados, que componentes como o componente padrão `StatisticsGen` usam como entradas.

Os artefatos devem ser fortemente tipados com um **tipo de artefato** registrado no armazenamento [ML Metadata](mlmd.md). Saiba mais sobre os [conceitos usados no ML Metadata](mlmd.md#concepts).

Tipos de artefato têm um nome e definem um esquema de suas propriedades. Os nomes dos tipos de artefato devem ser exclusivos no seu armazenamento ML Metadata. O TFX fornece diversos [tipos padrão de artefatos](https://github.com/tensorflow/tfx/blob/master/tfx/types/standard_artifacts.py){: .external } que descrevem tipos de dados complexos e tipos de valor, como: string, inteiro e flutuante. É possível [reutilizar esses tipos de artefato](https://github.com/tensorflow/tfx/blob/master/tfx/types/standard_artifacts.py){: .external } ou definir tipos de artefato personalizados que derivam de [`Artifact`](https://github.com/tensorflow/tfx/blob/master/tfx/types/artifact.py) {: .external }.

## Parâmetros

Parâmetros são entradas para pipelines que são conhecidos antes da execução do pipeline. Os parâmetros permitem alterar o comportamento de um pipeline, ou parte de um pipeline, por meio de configuração em vez de código.

Por exemplo, você pode usar parâmetros para executar um pipeline com diferentes conjuntos de hiperparâmetros sem alterar o código do pipeline.

O uso de parâmetros permite aumentar a velocidade da experimentação, facilitando a execução do pipeline com diferentes conjuntos de parâmetros.

Saiba mais sobre a [classe RuntimeParameter](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/data_types.py) {: .external }.

## Componentes

Um **componente** é uma implementação de uma tarefa de ML que você pode usar como uma etapa no pipeline TFX. Um componente é composto por:

- Uma especificação de componente, que define os artefatos de entrada e saída do componente e seus parâmetros obrigatórios.
- Um executor, que implementa o código para executar uma etapa no workflow de ML, como ingestão e transformação de dados ou treinamento e avaliação de um modelo.
- Uma interface de componente, que empacota a especificação do componente e do executor para uso em um pipeline.

O TFX fornece diversos [componentes padrão](index.md#tfx-standard-components) que você pode usar nos seus pipelines. Se esses componentes não atenderem às suas necessidades, você poderá criar componentes personalizados. [Saiba mais sobre componentes personalizados](understanding_custom_components.md).

## Pipeline

Um pipeline TFX é uma implementação portável de um workflow de ML que pode ser executado em vários orquestradores, como: Apache Airflow, Apache Beam e Kubeflow Pipelines. Um pipeline é composto por instâncias de componentes e parâmetros de entrada.

As instâncias de componentes produzem artefatos como saídas e normalmente dependem de artefatos produzidos por instâncias de componentes upstream como entradas. A sequência de execução para instâncias de componentes é determinada pela criação de um grafo acíclico direcionado (DAG) das dependências do artefato.

Por exemplo, considere um pipeline que faz o seguinte:

- Ingere (consome) dados diretamente de um sistema proprietário usando um componente personalizado.
- Calcula estatísticas para os dados de treinamento usando o componente padrão StatisticsGen.
- Cria um esquema de dados usando o componente padrão SchemaGen.
- Verifica se há anomalias nos dados de treinamento usando o componente padrão ExampleValidator.
- Executa engenharia de recursos no dataset usando o componente padrão Transform.
- Treina um modelo usando o componente padrão Trainer.
- Avalia o modelo treinado usando o componente Evaluator.
- Se o modelo passar na avaliação, o pipeline irá enfileirar o modelo treinado num sistema de implantação (deployment) proprietário usando um componente personalizado.

![](images/tfx_pipeline_graph.svg)

Para determinar a sequência de execução das instâncias dos componentes, o TFX analisa as dependências do artefato.

- O componente de ingestão de dados não possui dependências de artefato, portanto pode ser o primeiro nó no grafo.
- StatisticsGen depende dos *exemplos* produzidos pela ingestão de dados, portanto deve ser executado após a ingestão de dados.
- SchemaGen depende das *estatísticas* criadas pelo StatisticsGen, portanto deve ser executado após o StatisticsGen.
- ExemploValidator depende das *estatísticas* criadas por StatisticsGen e do *esquema* criado por SchemaGen, portanto deve ser executado após StatisticsGen e SchemaGen.
- A transformação depende dos *exemplos* produzidos pela ingestão de dados e do *esquema* criado pelo SchemaGen, portanto deve ser executada após a ingestão de dados e o SchemaGen.
- O Trainer depende dos *exemplos* produzidos pela ingestão de dados, do *esquema* criado pelo SchemaGen e do *modelo salvo* produzido pelo Transform. O Trainer pode ser executado somente depois da ingestão de dados, do SchemaGen e do Transform.
- O Evaluator depende dos *exemplos* produzidos pela ingestão de dados e do *modelo salvo* produzido pelo Trainer, portanto deve ser executado após a ingestão de dados e do Trainer.
- O implantador (deployer) personalizado depende do *modelo salvo* produzido pelo Trainer e dos *resultados da análise* criados pelo Evaluator, portanto o implantador deve ser executado depois do Trainer e do Evaluator.

Com base nesta análise, um orquestrador executa:

- As instâncias dos componentes de ingestão de dados, StatisticsGen e SchemaGen, sequencialmente.
- Os componentes ExampleValidator e Transform podem ser executados em paralelo, pois compartilham dependências de artefatos de entrada e não dependem da saída um do outro.
- Depois que o componente Transform for concluído, as instâncias do componente Trainer, Evaluator e implementador personalizado serão executadas sequencialmente.

Saiba mais sobre como [criar um pipeline TFX](build_tfx_pipeline.md).

## TFX Pipeline Template

Os templates de pipelines TFX facilitam o início do desenvolvimento de um pipeline, fornecendo um pipeline pré-construído que você pode personalizar para seu caso de uso.

Saiba mais sobre como [personalizar um modelo de pipeline do TFX](build_tfx_pipeline.md#pipeline-templates).

## Execução do pipeline

Uma "run" corresponde à execução de toda a sequencia de etapas de um pipeline uma única vez.

## Orquestrador

Um orquestrador (orchestrator) é um sistema onde você pode executar "runs" de um pipeline. O TFX oferece suporte a orquestradores como: [Apache Airflow](airflow.md), [Apache Beam](beam.md) e [Kubeflow Pipelines](kubeflow.md). O TFX também usa o termo *DagRunner* para se referir a uma implementação que oferece suporte a um orquestrador.
