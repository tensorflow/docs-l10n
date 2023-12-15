# Construindo pipelines TFX

Observação: para uma visão conceitual dos pipelines do TFX, consulte [Noções básicas sobre pipelines do TFX](understanding_tfx_pipelines).

Observação: quer construir seu primeiro pipeline antes de se aprofundar nos detalhes? Comece com [criando um pipeline usando um template](https://www.tensorflow.org/tfx/guide/build_local_pipeline#build_a_pipeline_using_a_template).

## Usando a classe `Pipeline`

Os pipelines do TFX são definidos usando a [classe `Pipeline`](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/pipeline.py){: .external }. O exemplo a seguir demonstra como usar a classe `Pipeline`.

<pre class="devsite-click-to-copy prettyprint">
pipeline.Pipeline(
    pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;,
    pipeline_root=&lt;var&gt;pipeline-root&lt;/var&gt;,
    components=&lt;var&gt;components&lt;/var&gt;,
    enable_cache=&lt;var&gt;enable-cache&lt;/var&gt;,
    metadata_connection_config=&lt;var&gt;metadata-connection-config&lt;/var&gt;,
)
</pre>

Substitua o seguinte:

- <var>pipeline-name</var>: o nome deste pipeline. O nome do pipeline deve ser único.

    O TFX usa o nome do pipeline ao consultar o ML Metadata para artefatos de componentes de entrada. Reusar o nome de um pipeline pode resultar em comportamentos inesperados.

- <var>pipeline-root</var>: o caminho raiz das saídas deste pipeline. O caminho raiz deve ser o caminho completo para um diretório ao qual seu orquestrador tenha acesso de leitura e gravação. Em tempo de execução, o TFX usa a raiz do pipeline para gerar caminhos de saída para artefatos de componentes. Esse diretório pode ser local ou num sistema de arquivos distribuído compatível, como Google Cloud Storage ou HDFS.

- <var>components</var>: uma lista de instâncias de componentes que compõem o workflow deste pipeline.

- <var>enable-cache</var>: (opcional.) Um valor booleano que indica se este pipeline usa cache para acelerar a execução do pipeline.

- <var>metadata-connection-config</var>: (Optional.) A connection configuration for ML Metadata.

## Definindo o grafo de execução do componente

As instâncias de componentes produzem artefatos como saídas e normalmente dependem de artefatos produzidos por instâncias de componentes upstream como entradas. A sequência de execução para instâncias de componentes é determinada pela criação de um grafo acíclico direcionado (DAG) das dependências do artefato.

Por exemplo, o componente padrão `ExampleGen` pode consumir dados de um arquivo CSV e gerar registros de exemplo serializados. O componente padrão `StatisticsGen` aceita esses registros de exemplo como entrada e produz estatísticas do dataset. Neste exemplo, a instância de `StatisticsGen` deve seguir `ExampleGen` porque `SchemaGen` depende da saída de `ExampleGen`.

### Dependências baseadas em tarefas

Observação: Normalmente não é recomendado usar dependências baseadas em tarefas. Definir o grafo de execução com dependências de artefato permite aproveitar as vantagens do rastreamento automático de linhagem de artefatos e dos recursos de cache do TFX.

Você também pode definir dependências baseadas em tarefas usando os métodos [`add_upstream_node` e `add_downstream_node`](https://github.com/tensorflow/tfx/blob/master/tfx/components/base/base_node.py){: .external } do seu componente. `add_upstream_node` permite especificar que o componente atual deve ser executado após o componente especificado. `add_downstream_node` permite especificar que o componente atual deve ser executado antes do componente especificado.

## Templates de pipeline

A maneira mais fácil de configurar um pipeline rapidamente e ver como todas as peças se encaixam é usar um template. O uso de templates é abordado em [Construindo um pipeline TFX localmente](build_local_pipeline).

## Caching

O cache do pipeline TFX permite que seu pipeline ignore componentes que foram executados com o mesmo conjunto de entradas numa execução anterior do pipeline. Se o cache estiver habilitado, o pipeline tentará corresponder a assinatura de cada componente, o componente em si e o conjunto de entradas, com uma das execuções anteriores do componente desse pipeline. Se houver uma correspondência, o pipeline usará as saídas do componente da execução anterior. Se não houver correspondência, o componente é executado.

Não use caches se o pipeline usar componentes não determinísticos. Por exemplo, se você criar um componente para produzir um número aleatório para seu pipeline, ativar o cache fará com que esse componente seja executado uma vez. Neste exemplo, as execuções subsequentes reusam o número aleatório da primeira execução em vez de gerar um número aleatório novo.
