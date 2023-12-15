# Construindo um pipeline TFX localmente

O TFX facilita a orquestração do fluxo de trabalho de machine learning (ML) como um pipeline, para:

- Automatizar seu processo de ML, o que permite treinar, avaliar e implantar regularmente seu modelo.
- Criar pipelines de ML que incluem análise profunda do desempenho do modelo e validação de modelos recém-treinados para garantir desempenho e confiabilidade.
- Monitorar dados de treinamento em busca de anomalias e eliminar desvios no fornecimento de treinamento
- Aumentar a velocidade da experimentação executando um pipeline com diferentes conjuntos de hiperparâmetros.

Um processo típico de desenvolvimento de pipeline começa numa máquina local, com análise de dados e configuração de componentes, antes de ser implantado na produção. Este guia descreve duas maneiras de construir um pipeline localmente.

- Personalizando um template de pipeline do TFX para atender às necessidades do seu workflows de ML. Os templates de pipeline do TFX são workflows pré-construídos que demonstram as práticas recomendadas usando os componentes padrão do TFX.
- Criando um pipeline usando TFX. Neste caso de uso, você define um pipeline sem partir de um template.

Ao desenvolver seu pipeline, você poderá executá-lo com `LocalDagRunner`. Depois que os componentes do pipeline tiverem sido bem definidos e testados, você pode usar um orquestrador de nível de produção, como Kubeflow ou Airflow.

## Antes de começar

O TFX é um pacote Python, então você precisará configurar um ambiente de desenvolvimento Python, como um ambiente virtual ou um container do Docker. Depois disso faça:

```bash
pip install tfx
```

Se você é novato em pipelines TFX, [aprenda mais sobre os principais conceitos dos pipelines TFX](understanding_tfx_pipelines) antes de continuar.

## Crie um pipeline usando um template

Os templates de pipeline do TFX facilitam o início do desenvolvimento de um pipeline, fornecendo um conjunto pré-construído de definições de pipeline que você pode adaptar para seu caso de uso.

As seções a seguir descrevem como criar uma cópia de um template e personalizá-lo para atender às suas necessidades.

### Crie uma cópia do template do pipeline

1. Veja a lista dos templates de pipeline TFX disponíveis:

    <pre class="devsite-click-to-copy devsite-terminal">
        tfx template list
        </pre>

2. Selecione um template da lista

    <pre class="devsite-click-to-copy devsite-terminal">
        tfx template copy --model=&lt;var&gt;template&lt;/var&gt; --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
        --destination_path=&lt;var&gt;destination-path&lt;/var&gt;
        </pre>

    Substitua o seguinte:

    - <var>template</var>: O nome do template que você deseja copiar.
    - <var>pipeline-name</var>: o nome do pipeline a ser criado.
    - <var>destination-path</var>: o caminho para copiar o template.

    Saiba mais sobre o [comando `tfx template copy`](cli#copy).

3. Uma cópia do template de pipeline foi criada no caminho especificado.

Observação: O restante deste guia pressupõe que você selecionou o template `penguin`.

### Explore o template do pipeline

Esta seção fornece uma visão geral da estrutura criada por um template.

1. Explore os diretórios e arquivos que foram copiados para o diretório raiz do seu pipeline

    - Um diretório **pipeline** com

        - `pipeline.py` – define o pipeline e lista quais componentes estão sendo usados
        - `configs.py` - contém detalhes de configuração, como de onde vêm os dados ou qual orquestrador está sendo usado

    - Um diretório **data**

        - Normalmente contém um arquivo `data.csv`, que é a fonte padrão para `ExampleGen`. Você pode alterar a fonte de dados em `configs.py`.

    - Um diretório **models** com código de pré-processamento e implementações de modelo

    - O template copia executores DAG para ambiente local e Kubeflow.

    - Alguns template também incluem Notebooks Python para que você possa explorar seus dados e artefatos com metadados de aprendizado de máquina.

2. Execute os seguintes comandos no diretório do pipeline:

    <pre class="devsite-click-to-copy devsite-terminal">
        tfx pipeline create --pipeline_path local_runner.py
        </pre>

    <pre class="devsite-click-to-copy devsite-terminal">
        tfx run create --pipeline_name &lt;var&gt;pipeline_name&lt;/var&gt;
        </pre>

    O comando cria uma execução de pipeline usando `LocalDagRunner`, que adiciona os seguintes diretórios ao pipeline:

    - Um diretório **tfx_metadata** que contém o armazenamento ML Metadata usado localmente.
    - Um diretório **tfx_pipeline_output** que contém as saídas de arquivos do pipeline.

    Observação: `LocalDagRunner` é um dos vários orquestradores suportados no TFX. É especialmente adequado para executar pipelines localmente para iterações mais rápidas, possivelmente com datasets menores. `LocalDagRunner` pode não ser adequado para uso em produção, pois é executado numa única máquina, o que é mais vulnerável à perda de trabalho se o sistema ficar indisponível. O TFX também oferece suporte a orquestradores como Apache Beam, Apache Airflow e Kubeflow Pipeline. Se você estiver usando o TFX com um orquestrador diferente, use o executor DAG apropriado para esse orquestrador.

    Observação: no momento em que este artigo foi escrito, `LocalDagRunner` era usado no template `penguin`, enquanto o template `taxi` usava Apache Beam. Os arquivos de configuração do template `taxi` estão configurados para usar o Beam e o comando CLI é o mesmo.

3. Abra o arquivo `pipeline/configs.py` do seu pipeline e revise seu conteúdo. Este script define as opções de configuração usadas pelo pipeline e pelas funções do componente. É aqui que você especificaria coisas como a localização da fonte de dados ou o número de passos de treinamento numa execução.

4. Abra o arquivo `pipeline/pipeline.py` do seu pipeline e revise seu conteúdo. Este script cria o pipeline do TFX. Inicialmente, o pipeline contém apenas um componente `ExampleGen`.

    - Siga as instruções nos comentários **TODO** em `pipeline.py` para adicionar mais passos ao pipeline.

5. Abra o arquivo `local_runner.py` e revise seu conteúdo. Este script cria uma execução de pipeline e especifica os *parâmetros* da execução, como `data_path` e `preprocessing_fn`.

6. Você revisou a estrutura criado pelo template e criou uma execução de pipeline usando `LocalDagRunner`. Em seguida, você precisa personalizar o template para atender às suas necessidades.

### Personalize seu pipeline

Esta seção fornece uma visão geral de como começar a personalizar seu template.

1. Projete seu pipeline. A estrutura fornecida por um template ajuda a implementar um pipeline para dados tabulares usando os componentes padrão do TFX. Se você estiver migrando um workflow de ML existente para um pipeline, talvez seja necessário revisar seu código para aproveitar ao máximo [os componentes padrão do TFX](index#tfx_standard_components). Você também pode precisar criar [componentes personalizados](understanding_custom_components) que implementem recursos exclusivos do seu workflow ou que ainda não sejam suportados pelos componentes padrão do TFX.

2. Depois de projetar seu pipeline, personalize-o iterativamente seguindo o processo a seguir. Comece pelo componente que consome dados no seu pipeline, que geralmente é o componente `ExampleGen`.

    1. Personalize o pipeline ou um componente de acordo com seu caso de uso. Essas personalizações podem incluir alterações como:

        - Alteração de parâmetros do pipeline.
        - Adição de componentes ao pipeline ou sua remoção.
        - Substituição da fonte de dados de entrada. Essa fonte de dados pode ser um arquivo ou consultas em serviços como o BigQuery.
        - Alteração da configuração de um componente no pipeline.
        - Alteração da função de personalização de um componente.

    2. Execute o componente localmente usando o script `local_runner.py` ou outro executor DAG apropriado se você estiver usando um orquestrador diferente. Se o script falhar, depure a falha e tente executar o script novamente.

    3. Assim que essa personalização estiver funcionando, passe para a próxima personalização.

3. Trabalhando de forma iterativa, você poderá personalizar cada etapa do fluxo de trabalho do template para atender às suas necessidades.

## Crie um pipeline personalizado

Use as instruções a seguir para saber mais sobre como criar um pipeline personalizado sem usar um template.

1. Projete seu pipeline. Os componentes padrão do TFX fornecem funcionalidade comprovada para ajudá-lo a implementar um workflow de ML completo. Se você estiver migrando um workflow de ML existente para um pipeline, talvez seja necessário revisar seu código para aproveitar ao máximo os componentes padrão do TFX. Também pode ser necessário criar [componentes personalizados](understanding_custom_components) que implementem recursos como ampliação de dados.

    - Saiba mais sobre os [componentes padrão do TFX](index#tfx_standard_components).
    - Saiba mais sobre [componentes personalizados](understanding_custom_components).

2. Crie um arquivo de script para definir seu pipeline usando o exemplo a seguir. Este guia refere-se a este arquivo como `my_pipeline.py`.

    <pre class="devsite-click-to-copy prettyprint">
        import os
        from typing import Optional, Text, List
        from absl import logging
        from ml_metadata.proto import metadata_store_pb2
        import tfx.v1 as tfx

        PIPELINE_NAME = 'my_pipeline'
        PIPELINE_ROOT = os.path.join('.', 'my_pipeline_output')
        METADATA_PATH = os.path.join('.', 'tfx_metadata', PIPELINE_NAME, 'metadata.db')
        ENABLE_CACHE = True

        def create_pipeline(
          pipeline_name: Text,
          pipeline_root:Text,
          enable_cache: bool,
          metadata_connection_config: Optional[
            metadata_store_pb2.ConnectionConfig] = None,
          beam_pipeline_args: Optional[List[Text]] = None
        ):
          components = []

          return tfx.dsl.Pipeline(
                pipeline_name=pipeline_name,
                pipeline_root=pipeline_root,
                components=components,
                enable_cache=enable_cache,
                metadata_connection_config=metadata_connection_config,
                beam_pipeline_args=beam_pipeline_args, &lt;!-- needed? --&gt;
            )

        def run_pipeline():
          my_pipeline = create_pipeline(
              pipeline_name=PIPELINE_NAME,
              pipeline_root=PIPELINE_ROOT,
              enable_cache=ENABLE_CACHE,
              metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH)
              )

          tfx.orchestration.LocalDagRunner().run(my_pipeline)

        if __name__ == '__main__':
          logging.set_verbosity(logging.INFO)
          run_pipeline()
        </pre>

    Nos próximos passos, você definirá seu pipeline em `create_pipeline` e executará seu pipeline localmente usando o executor local.

    Construa seu pipeline iterativamente usando o processo a seguir.

    1. Personalize o pipeline ou um componente de acordo com seu caso de uso. Essas personalizações podem incluir alterações como:

        - Alteração de parâmetros do pipeline.
        - Adição de componentes ao pipeline ou sua remoção.
        - Substituição de um arquivo de entrada de dados.
        - Alteração da configuração de um componente no pipeline.
        - Alteração da função de personalização de um componente.

    2. Execute o componente localmente usando o executor local ou executando o script diretamente. Se o script falhar, depure a falha e tente executar o script novamente.

    3. Assim que essa personalização estiver funcionando, passe para a próxima personalização.

    Comece no primeiro nó do workflow do seu pipeline, normalmente o primeiro nó consome dados para o seu pipeline.

3. Adicione o primeiro nó do workflow ao pipeline. Neste exemplo, o pipeline usa o componente padrão `ExampleGen` para carregar um CSV de um diretório em `./data`.

    <pre class="devsite-click-to-copy prettyprint">
        from tfx.components import CsvExampleGen

        DATA_PATH = os.path.join('.', 'data')

        def create_pipeline(
          pipeline_name: Text,
          pipeline_root:Text,
          data_path: Text,
          enable_cache: bool,
          metadata_connection_config: Optional[
            metadata_store_pb2.ConnectionConfig] = None,
          beam_pipeline_args: Optional[List[Text]] = None
        ):
          components = []

          example_gen = tfx.components.CsvExampleGen(input_base=data_path)
          components.append(example_gen)

          return tfx.dsl.Pipeline(
                pipeline_name=pipeline_name,
                pipeline_root=pipeline_root,
                components=components,
                enable_cache=enable_cache,
                metadata_connection_config=metadata_connection_config,
                beam_pipeline_args=beam_pipeline_args, &lt;!-- needed? --&gt;
            )

        def run_pipeline():
          my_pipeline = create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_path=DATA_PATH,
            enable_cache=ENABLE_CACHE,
            metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH)
            )

          tfx.orchestration.LocalDagRunner().run(my_pipeline)
        </pre>

    `CsvExampleGen` cria registros de exemplo serializados usando os dados do CSV no caminho de dados especificado. Definindo o parâmetro `input_base` do componente `CsvExampleGen` com a raiz de dados.

4. Crie um diretório `data` no mesmo diretório que `my_pipeline.py`. Adicione um pequeno arquivo CSV ao diretório `data`.

5. Use o seguinte comando para executar seu script `my_pipeline.py`.

    <pre class="devsite-click-to-copy devsite-terminal">
        python my_pipeline.py
        </pre>

    O resultado deve ser algo similar ao mostrado abaixo:

    <pre>
        INFO:absl:Component CsvExampleGen depends on [].
        INFO:absl:Component CsvExampleGen is scheduled.
        INFO:absl:Component CsvExampleGen is running.
        INFO:absl:Running driver for CsvExampleGen
        INFO:absl:MetadataStore with DB connection initialized
        INFO:absl:Running executor for CsvExampleGen
        INFO:absl:Generating examples.
        INFO:absl:Using 1 process(es) for Local pipeline execution.
        INFO:absl:Processing input csv data ./data/* to TFExample.
        WARNING:root:Couldn't find python-snappy so the implementation of _TFRecordUtil._masked_crc32c is not as fast as it could be.
        INFO:absl:Examples generated.
        INFO:absl:Running publisher for CsvExampleGen
        INFO:absl:MetadataStore with DB connection initialized
        INFO:absl:Component CsvExampleGen is finished.
        </pre>

6. Continue a adicionar componentes iterativamente ao seu pipeline.
