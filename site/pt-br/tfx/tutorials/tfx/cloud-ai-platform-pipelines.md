# TFX em Cloud AI Platform Pipelines

## Introdução

Este tutorial foi desenvolvido para introduzir o [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx) e [AIPlatform Pipelines] (https://cloud.google.com/ai-platform/pipelines/docs/introduction) e ajudar você a aprender a criar seus próprios pipelines de aprendizado de máquina (ML) no Google Cloud. Ele mostra integração com TFX, AI Platform Pipelines e Kubeflow, bem como interação com TFX em notebooks Jupyter.

Ao final deste tutorial, você terá criado e executado um pipeline de ML hospedado no Google Cloud. Você poderá visualizar os resultados de cada execução e visualizar a linhagem dos artefatos criados.

Termo chave: um pipeline TFX é um grafo acíclico direcionado ou "DAG". Freqüentemente nos referiremos aos pipelines como DAGs.

Você seguirá um processo típico de desenvolvimento de ML, começando com a análise do dataset e terminando com um pipeline completo e funcional. Ao longo do caminho, você explorará maneiras de depurar e atualizar seu pipeline e medir seu desempenho.

Observação: a conclusão deste tutorial pode levar de 45 a 60 minutos.

### Dataset Chicago Taxi

<!-- Image free for commercial use, does not require attribution:
https://pixabay.com/photos/new-york-cab-cabs-taxi-urban-city-2087998/ -->

![Taxi](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/taxi.jpg?raw=true) ![Chicago taxi](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/chicago.png?raw=true)

Você usará o [dataset Taxi Trips](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew) disponibilizado pela cidade de Chicago.

Observação: Este site fornece aplicativos que utilizam dados que foram modificados para uso a partir de sua fonte original obtida em www.cityofchicago.org, site oficial da cidade de Chicago. A cidade de Chicago não faz nenhuma reivindicação quanto ao conteúdo, precisão, atualidade ou integridade de qualquer um dos dados fornecidos neste site. Os dados fornecidos neste site estão sujeitos a alterações a qualquer momento. Entende-se que os dados fornecidos neste site são utilizados por sua conta e risco.

[Leia mais](https://cloud.google.com/bigquery/public-data/chicago-taxi) sobre o dataset no [Google BigQuery](https://cloud.google.com/bigquery/). Explore o dataset completo no [BigQuery UI](https://bigquery.cloud.google.com/dataset/bigquery-public-data:chicago_taxi_trips).

#### Objetivo do modelo – Classificação binária

O cliente dará uma gorjeta maior ou menor que 20%?

## 1. Configure um projeto do Google Cloud

### 1.a Configure seu ambiente no Google Cloud

Para começar, você precisa ter uma conta no Google Cloud. Se você já tiver uma, vá para [Criar novo projeto](#create_project).

Aviso: esta demonstração foi projetada para não exceder os limites do [nível gratuito do Google Cloud](https://cloud.google.com/free). Se você já possui uma Conta do Google, pode já ter atingido os limites do nível gratuito ou esgotado todos os créditos gratuitos do Google Cloud concedidos a novos usuários. **Se for esse o caso, seguir esta demonstração resultará em cobranças em sua conta do Google Cloud**.

1. Acesse o [Console do Google Cloud](https://console.cloud.google.com/) .

2. Concorde com os termos e condições do Google Cloud

    <img style="width: 50%;" src="images/cloud-ai-platform-pipelines/welcome-popup.png">

3. Se quiser começar com uma conta de avaliação gratuita, clique em [**Try For Free**](https://console.cloud.google.com/freetrial) (ou [**Get started for free**](https://console.cloud.google.com/freetrial)).

    1. Escolha o seu país.

    2. Concorde com os termos de serviço.

    3. Informe seus dados de cobrança.

        Você não será cobrado neste momento. Se você não tiver outros projetos do Google Cloud, poderá concluir este tutorial sem exceder os limites do [nível gratuito do Google Cloud](https://cloud.google.com/free), que inclui no máximo oito núcleos (cores) rodando ao mesmo tempo.

Observação: neste momento você também pode optar por se tornar um usuário pago em vez de contar com a avaliação gratuita. Como este tutorial permanece dentro dos limites do nível gratuito, você ainda não será cobrado se este for seu único projeto e permanecer dentro desses limites. Para mais detalhes, consulte a [Calculadora de custos do Google Cloud](https://cloud.google.com/products/calculator/) e [Nível gratuito do Google Cloud Platform](https://cloud.google.com/free).

### 1.b Crie um novo projeto.<a name="create_project"></a>

Observação: este tutorial pressupõe que você deseja trabalhar nesta demonstração em um novo projeto. Você pode, se quiser, trabalhar em um projeto existente.

Observação: você precisa ter registrado um cartão de crédito verificado antes de criar o projeto.

1. No [painel principal do Google Cloud](https://console.cloud.google.com/home/dashboard), clique no menu suspenso do projeto próximo ao cabeçalho do **Google Cloud Platform** e selecione **New project**.
2. Dê um nome ao seu projeto e insira outros detalhes do projeto
3. **Depois de criar um projeto, selecione-o no menu suspenso do projeto.**

## 2. Configure e implante um AI Platform Pipeline em um novo cluster Kubernetes

Observação: isso poderá levar até 10 minutos, pois exige a espera em vários pontos para que os recursos sejam provisionados.

1. Acesse a página [AI Platform Pipelines Clusters](https://console.cloud.google.com/ai-platform/pipelines).

    No menu principal: ≡ &gt; AI Platform &gt; Pipelines

2. Clique em **+ New Instance** para criar um novo cluster.

    <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/new-instance.png">

3. Na página de visão geral do **Kubeflow Pipelines**, clique em **Configure**.

    <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/configure.png">

4. Clique em "Enable" para ativar a API do Kubernetes Engine

    <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/enable_api.png">

    Observação: talvez seja necessário vários alguns minutos antes de prosseguir, enquanto as APIs do Kubernetes Engine estão sendo ativadas para você.

5. Na página **Deploy Kubeflow Pipelines**:

    1. Selecione uma [zona](https://cloud.google.com/compute/docs/regions-zones) (ou "região") para o seu cluster. A rede e a sub-rede podem ser definidas, mas para os fins deste tutorial vamos deixá-las com os valores padrão.

    2. **IMPORTANTE** Marque a caixa *Allow access to the following cloud APIs* (Permitir acesso às seguintes APIs de nuvem). Isto é necessário para que este cluster acesse as outras partes do seu projeto. Se você perder esta etapa, corrigi-la mais tarde será um pouco complicado.

        <img style="width: 50%;" src="images/cloud-ai-platform-pipelines/check-the-box.png">

    3. Clique em **Create New Cluster** e aguarde alguns minutos até que o cluster seja criado. Isto levará alguns minutos. Quando terminar, você verá uma mensagem como:

        > Cluster "cluster-1" successfully created in zone "us-central1-a".

    4. Selecione um namespace e um nome de instância (usar os padrões é ok). Para os fins deste tutorial, não marque *executor.emissary* ou *managedstorage.enabled*.

    5. Clique em **Deploy** e aguarde alguns instantes até que o pipeline seja implantado. Ao implantar o Kubeflow Pipelines, você aceita os Termos de Serviço.

## 3. Configure uma instância do Cloud AI Platform Notebook.

1. Acesse a página do [Vertex AI Workbench](https://console.cloud.google.com/vertex-ai/workbench). Na primeira vez que você executar o Workbench, será necessário ativar a API Notebooks.

    No menu de navegação principal: ≡ -&gt; Vertex AI -&gt; Workbench

2. Se solicitado, ative a API Compute Engine.

3. Crie um **New Notebook** com o TensorFlow Enterprise 2.7 (ou superior) instalado.

    <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/new-notebook.png">

    New Notebook -&gt; TensorFlow Enterprise 2.7 -&gt; Without GPU

    Selecione uma região e zona e dê um nome à instância do notebook.

    Para permanecer dentro dos limites do nível gratuito, talvez seja necessário alterar as configurações padrão aqui para reduzir o número de vCPUs disponíveis para esta instância de 4 para 2:

    1. Selecione **Advanced Options** na parte inferior do formulário **New notebook**.

    2. Em **Machine configuration** você talvez queira selecionar uma configuração com 1 ou 2 vCPUs para permanecer no nível gratuito.

        <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/two-cpus.png">

    3. Aguarde a criação do novo notebook e clique em **Enable Notebooks API**

Observação: você poderá enfrentar um desempenho lento em seu notebook se usar 1 ou 2 vCPUs em vez do padrão ou superior. Isto não deve prejudicar seriamente a conclusão deste tutorial. Se desejar usar as configurações padrão, [atualize sua conta](https://cloud.google.com/free/docs/gcp-free-tier#to_upgrade_your_account) para pelo menos 12 vCPUs. Isto vai resultar em cobrança adicional. Veja [Google Kubernetes Engine Pricing](https://cloud.google.com/kubernetes-engine/pricing/) para mais detalhes sobre preços, incluindo uma [calculadora de preços](https://cloud.google.com/products/calculator) e informações sobre o [nível gratuito do Google Cloud](https://cloud.google.com/free).

## 4. Inicie o Notebook

1. Acesse a página [**AI Platform Pipelines Clusters**] (https://console.cloud.google.com/ai-platform/pipelines).

    No menu principal: ≡ &gt; AI Platform &gt; Pipelines

2. Na linha do cluster que você está usando neste tutorial, clique em **Open Pipelines Dashboard**.

    <img src="images/cloud-ai-platform-pipelines/open-dashboard.png">

3. Na página **Getting Started** cliqie em **Open a Cloud AI Platform Notebook on Google Cloud**.

    <img src="images/cloud-ai-platform-pipelines/open-template.png">

4. Selecione a instância do Notebook que você está usando para este tutorial e **Continue** e, em seguida, **Confirm**.

    ![select-notebook](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/cloud-ai-platform-pipelines/select-notebook.png?raw=true)

## 5. Continue trabalhando no Notebook

Importante: o restante deste tutorial deve ser concluído no Jupyter Lab Notebook que você abriu na etapa anterior. As instruções e explicações estão disponíveis aqui como referência.

### Instalação

O notebook Getting Started começa com a instalação do [TFX](https://www.tensorflow.org/tfx) e do [Kubeflow Pipelines (KFP)](https://www.kubeflow.org/docs/pipelines/) na VM em que o Jupyter Lab está sendo executado.

Em seguida, ele verifica qual versão do TFX está instalada, faz uma importação, define e imprime o ID do projeto:

![check python version and import](https://gitlocalize.com/repo/4592/pt-br/site/en-snapshot/tfx/tutorials/tfx/images/cloud-ai-platform-pipelines/check-version-nb-cell.png)

### Conexão aos serviços do Google Cloud

A configuração do pipeline precisa do ID do seu projeto, que você pode obter por meio do notebook e definir como uma variável de ambiente.

```python
# Read GCP project id from env.
shell_output=!gcloud config list --format 'value(core.project)' 2>/dev/null
GCP_PROJECT_ID=shell_output[0]
print("GCP project ID:" + GCP_PROJECT_ID)
```

Agora defina o endpoint do cluster KFP.

Isso pode ser encontrado na URL do painel Pipelines. Vá para o painel do Kubeflow Pipeline e veja a URL. O endpoint é tudo na URL *, começando com *`https://`, *até e incluindo* `googleusercontent.com`.

```python
ENDPOINT='' # Enter YOUR ENDPOINT here.
```

O notebook então define um nome exclusivo para a imagem personalizada do Docker:

```python
# Docker image name for the pipeline image
CUSTOM_TFX_IMAGE='gcr.io/' + GCP_PROJECT_ID + '/tfx-pipeline'
```

## 6. Copie um template para o diretório do seu projeto

Edite a próxima célula do notebook para definir um nome para seu pipeline. Neste tutorial usaremos `my_pipeline`.

```python
PIPELINE_NAME="my_pipeline"
PROJECT_DIR=os.path.join(os.path.expanduser("~"),"imported",PIPELINE_NAME)
```

O notebook então usa a CLI `tfx` para copiar o template de pipeline. Este tutorial usa o dataset Chicago Taxi para realizar a classificação binária, portanto, o template define o modelo como `taxi`:

```python
!tfx template copy \
  --pipeline-name={PIPELINE_NAME} \
  --destination-path={PROJECT_DIR} \
  --model=taxi
```

O notebook então muda seu contexto CWD para o diretório do projeto:

```
%cd {PROJECT_DIR}
```

### Explore os arquivos do pipeline

No lado esquerdo do Cloud AI Platform Notebook, você verá um navegador de arquivos. Deve haver um diretório com o nome do seu pipeline (`my_pipeline`). Abra-o e visualize os arquivos. (Você também poderá abri-los e editá-los no ambiente do notebook.)

```
# You can also list the files from the shell
! ls
```

O comando `tfx template copy` acima criou uma estrutura básica de arquivos que constroem um pipeline. Isto inclui códigos-fonte Python, dados de amostra e notebooks Jupyter. Eles se destinam a este exemplo específico. Para seus próprios pipelines, esses seriam os arquivos de suporte exigidos pelo pipeline.

Aqui está uma breve introdução a cada um dos arquivos Python.

- `pipeline` - este diretório contém a definição do pipeline
    - `configs.py` — define constantes comuns para executores de pipeline
    - `pipeline.py` — define componentes TFX e um pipeline
- `models` - este diretório contém definições de modelo de ML.
    - `features.py`/`features_test.py` — define características para o modelo
    - `preprocessing.py`/`preprocessing_test.py` — define jobs de pré-processamento usando `tf::Transform`
    - `estimator` — este diretório contém um modelo baseado em Estimator.
        - `constants.py` — define constantes do modelo
        - `model.py`/`model_test.py` — define o modelo DNN usando o estimador TF
    - `keras` — este diretório contém um modelo baseado em Keras.
        - `constants.py` — define constantes do modelo
        - `model.py`/`model_test.py` — define o modelo DNN usando Keras
- `local_runner.py`/ `kubeflow_runner.py` — define executores para cada mecanismo de orquestração

## 7. Execute seu primeiro pipeline TFX no Kubeflow

O notebook executará o pipeline usando o comando CLI `tfx run`.

### Conecte-se ao armazenamento

A execução de pipelines cria artefatos que devem ser armazenados em [ML-Metadata](https://github.com/google/ml-metadata). Os artefatos referem-se a payloads, que são arquivos que devem ser armazenados em sistema de arquivos ou armazenamento em bloco. Neste tutorial, usaremos o GCS para armazenar nossos payloads de metadados, usando o bucket que foi criado automaticamente durante a configuração. Seu nome será `<your-project-id>-kubeflowpipelines-default`.

### Crie o pipeline

O notebook fará upload de nossos dados de amostra para o bucket do GCS para que possamos usá-los em nosso pipeline posteriormente.

```python
!gsutil cp data/data.csv gs://{GOOGLE_CLOUD_PROJECT}-kubeflowpipelines-default/tfx-template/data/taxi/data.csv
```

O notebook depois usa o comando `tfx pipeline create` para criar o pipeline.

```python
!tfx pipeline create  \
--pipeline-path=kubeflow_runner.py \
--endpoint={ENDPOINT} \
--build-image
```

Ao criar um pipeline, um `Dockerfile` será gerado para construir uma imagem Docker. Não esqueça de adicioná-lo ao sistema de controle de fontes (por exemplo, git) junto com outros arquivos-fonte.

### Execute o pipeline

O notebook depois usa o comando `tfx run create` para iniciar uma execução do seu pipeline. Você também verá esta execução listada em Experiments no painel do Kubeflow Pipelines.

```python
!tfx run create --pipeline-name={PIPELINE_NAME} --endpoint={ENDPOINT}
```

Você pode visualizar seu pipeline no painel Kubeflow Pipelines.

Observação: se a execução do seu pipeline falhar, você poderá ver logs detalhados no painel do KFP. Uma das principais fontes de falha são os problemas relacionados à permissões. Verifique se o cluster KFP tem as permissões necessárias para acessar as APIs do Google Cloud. Isto pode ser configurado [quando você cria um cluster KFP no GCP](https://cloud.google.com/ai-platform/pipelines/docs/setting-up) ou veja o [documento sobre solução de problemas no GCP](https://cloud.google.com/ai-platform/pipelines/docs/troubleshooting).

## 8. Valide seus dados

A primeira tarefa em qualquer projeto de ciência de dados ou ML é compreender e limpar os dados, o que inclui:

- Compreender os tipos de dados de cada característica
- Procurar anomalias e valores ausentes
- Compreender as distribuições de cada característica

### Componentes

![Data Components](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/examplegen1.png?raw=true) ![Data Components](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/examplegen2.png?raw=true)

- [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen) ingere e divide o dataset de entrada.
- [StatisticsGen](https://www.tensorflow.org/tfx/guide/statsgen) calcula estatísticas para o dataset.
- [SchemaGen](https://www.tensorflow.org/tfx/guide/schemagen) examina as estatísticas e cria um esquema de dados.
- [ExampleValidator](https://www.tensorflow.org/tfx/guide/exampleval) procura anomalias e valores ausentes no dataset.

### No editor de arquivos do Jupyter Lab:

Em `pipeline`/`pipeline.py`, descomente as linhas que acrescentam esses componentes ao seu pipeline:

```python
# components.append(statistics_gen)
# components.append(schema_gen)
# components.append(example_validator)
```

(`ExampleGen` já estava ativado quando os arquivos de template foram copiados.)

### Atualize o pipeline e execute-o novamente

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### Verifique o pipeline

Para o Kubeflow Orchestrator, acesse o painel do KFP para encontrar as saídas do pipeline na página da execução do seu pipeline. Clique na aba <em>Experiments</em> à esquerda e <em>All runs</em> na página Experiments. Você deverá conseguir encontrar a execução mais recente com o nome do seu pipeline.

### Exemplo mais avançado

O exemplo apresentado aqui serve apenas para você começar. Para obter um exemplo mais avançado, veja o [TensorFlow Data Validation Colab](https://www.tensorflow.org/tfx/tutorials/data_validation/chicago_taxi).

Para obter mais informações sobre como usar o TFDV para explorar e validar um dataset, [veja exemplos em tensorflow.org](https://www.tensorflow.org/tfx/data_validation).

## 9. Engenharia de características

Você pode aumentar a qualidade preditiva de seus dados e/ou reduzir a dimensionalidade com engenharia de características.

- Cruzamentos de características
- Vocabulários
- Embeddings
- PCA
- Codificação categórica

Um dos benefícios de usar o TFX é que você escreverá seu código de transformação uma vez e as transformações resultantes serão consistentes entre o treinamento e o serviço.

### Componentes

![Transform](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/images/airflow_workshop/transform.png?raw=true)

- O [Transform](https://www.tensorflow.org/tfx/guide/transform) realiza engenharia de características no dataset.

### No editor de arquivos do Jupyter Lab:

Em `pipeline`/`pipeline.py`, encontre e descomente as linhas que acrescentam [Transform](https://www.tensorflow.org/tfx/guide/transform) ao pipeline:

```python
# components.append(transform)
```

### Atualize o pipeline e execute-o novamente

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### Verifique as saídas do pipeline

Para o Kubeflow Orchestrator, acesse o painel do KFP para encontrar as saídas do pipeline na página da execução do seu pipeline. Clique na aba <em>Experiments</em> à esquerda e <em>All runs</em> na página Experiments. Você deverá conseguir encontrar a execução mais recente com o nome do seu pipeline.

### Exemplo mais avançado

O exemplo apresentado aqui serve apenas para você começar. Para obter um exemplo mais avançado, veja o [TensorFlow Transform Colab](https://www.tensorflow.org/tfx/tutorials/transform/census).

## 10. Treinamento

Treine um modelo do TensorFlow com seus dados bonitos, limpos e transformados.

- Inclua as transformações da etapa anterior para que sejam aplicadas de forma consistente
- Salve os resultados como SavedModel para produção
- Visualize e explore o processo de treinamento usando o TensorBoard
- Salve também um EvalSavedModel para análise do desempenho do modelo

### Componentes

- O [Trainer](https://www.tensorflow.org/tfx/guide/trainer) treina um modelo do TensorFlow.

### No editor de arquivos do Jupyter Lab:

Em `pipeline`/`pipeline.py`, encontre e descomente a linha que acrescenta Trainer ao pipeline:

```python
# components.append(trainer)
```

### Atualize o pipeline e execute-o novamente

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### Verifique as saídas do pipeline

Para o Kubeflow Orchestrator, acesse o painel do KFP para encontrar as saídas do pipeline na página da execução do seu pipeline. Clique na aba <em>Experiments</em> à esquerda e <em>All runs</em> na página Experiments. Você deverá conseguir encontrar a execução mais recente com o nome do seu pipeline.

### Exemplo mais avançado

O exemplo apresentado aqui serve apenas para você começar. Para obter um exemplo mais avançado, veja o [TensorBoard Tutorial](https://www.tensorflow.org/tensorboard/get_started).

## 11. Analisando o desempenho de modelos

Compreender mais do que apenas as métricas de nível superior.

- Os usuários experimentam o desempenho do modelo apenas nas consultas deles
- O baixo desempenho em fatias de dados pode ser ocultado por métricas de nível superior
- A equidade do modelo é importante
- Muitas vezes, os principais subconjuntos de usuários ou dados são muito importantes e podem ser pequenos
    - Desempenho em condições críticas, mas incomuns
    - Desempenho para públicos-chave, como influenciadores
- Se você estiver substituindo um modelo que está atualmente em produção, primeiro certifique-se de que o novo seja melhor

### Componentes

- O [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator) realiza uma análise profunda dos resultados do treinamento.

### No editor de arquivos do Jupyter Lab:

Em `pipeline`/`pipeline.py`, encontre e descomente a linha que acrescenta Evaluator ao pipeline:

```python
components.append(evaluator)
```

### Atualize o pipeline e execute-o novamente

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### Verifique as saídas do pipeline

Para o Kubeflow Orchestrator, acesse o painel do KFP para encontrar as saídas do pipeline na página da execução do seu pipeline. Clique na aba <em>Experiments</em> à esquerda e <em>All runs</em> na página Experiments. Você deverá conseguir encontrar a execução mais recente com o nome do seu pipeline.

## 12. Servindo o modelo

Se o novo modelo estiver pronto, disponibilize-o.

- O Pusher implanta SavedModels em locais bem conhecidos

Os alvos de implantação recebem novos modelos de locais conhecidos

- TensorFlow Serving
- TensorFlow Lite
- TensorFlow JS
- TensorFlow Hub

### Componentes

- O [Pusher](https://www.tensorflow.org/tfx/guide/pusher) implanta o modelo numa infraestrutura de serviço.

### No editor de arquivos do Jupyter Lab:

Em `pipeline`/`pipeline.py`, encontre e descomente a linha que acrescenta Pusher ao pipeline:

```python
# components.append(pusher)
```

### Verifique as saídas do pipeline

Para o Kubeflow Orchestrator, acesse o painel do KFP para encontrar as saídas do pipeline na página da execução do seu pipeline. Clique na aba <em>Experiments</em> à esquerda e <em>All runs</em> na página Experiments. Você deverá conseguir encontrar a execução mais recente com o nome do seu pipeline.

### Destinos de implantação disponíveis

Agora você treinou e validou seu modelo, e ele está pronto para entrar em produção. Agora você pode implantar seu modelo em qualquer um dos destinos de implantação do TensorFlow, incluindo:

- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving), para servir seu modelo em servidor ou farm de servidores e processar solicitações de inferência REST e/ou gRPC.
- [TensorFlow Lite](https://www.tensorflow.org/lite), para incluir seu modelo num aplicativo móvel nativo Android ou iOS ou num aplicativo Raspberry Pi, IoT ou microcontrolador.
- [TensorFlow.js](https://www.tensorflow.org/js), para executar seu modelo num navegador da web ou aplicativo Node.JS.

## Exemplos mais avançados

O exemplo apresentado acima serve apenas para você começar. Abaixo estão alguns exemplos de integração com outros serviços Cloud.

### Considerações sobre recursos do Kubeflow Pipelines

Dependendo dos requisitos da sua carga de trabalho, a configuração padrão para a implantação do Kubeflow Pipelines pode ou não atender às suas necessidades. Você pode personalizar suas configurações de recursos usando `pipeline_operator_funcs` em sua chamada para `KubeflowDagRunnerConfig`.

`pipeline_operator_funcs` é uma lista de itens `OpFunc`, que transforma todas as instâncias `ContainerOp` geradas na especificação de pipeline KFP que é compilada de `KubeflowDagRunner`.

Por exemplo, para configurar a memória podemos usar [`set_memory_request`](https://github.com/kubeflow/pipelines/blob/646f2fa18f857d782117a078d626006ca7bde06d/sdk/python/kfp/dsl/_container_op.py#L249) para declarar a quantidade de memória necessária. Uma maneira típica de fazer isso é criar um wrapper para `set_memory_request` e usá-lo para adicionar à lista de `OpFunc` s do pipeline:

```python
def request_more_memory():
  def _set_memory_spec(container_op):
    container_op.set_memory_request('32G')
  return _set_memory_spec

# Then use this opfunc in KubeflowDagRunner
pipeline_op_funcs = kubeflow_dag_runner.get_default_pipeline_operator_funcs()
pipeline_op_funcs.append(request_more_memory())
config = KubeflowDagRunnerConfig(
    pipeline_operator_funcs=pipeline_op_funcs,
    ...
)
kubeflow_dag_runner.KubeflowDagRunner(config=config).run(pipeline)
```

Funções semelhantes de configuração de recursos incluem:

- `set_memory_limit`
- `set_cpu_request`
- `set_cpu_limit`
- `set_gpu_limit`

### Experimente o `BigQueryExampleGen`

[BigQuery](https://cloud.google.com/bigquery) é um data warehouse em nuvem sem servidor, altamente escalonável e econômico. O BigQuery pode ser usado como fonte para exemplos de treinamento no TFX. Nesta etapa, adicionaremos <code>BigQueryExampleGen</code> ao pipeline.

#### No editor de arquivos do Jupyter Lab:

**Dê um duplo-clique para abrir `pipeline/pipeline.py`**. Comente `CsvExampleGen` e descomente a linha que cria uma instância de `BigQueryExampleGen`. Você também precisa descomentar o argumento `query` da função `create_pipeline`.

Precisamos especificar qual projeto GCP usar para o BigQuery, e isso é feito definindo `--project` em `beam_pipeline_args` ao criar um pipeline.

**Dê um duplo-clique para abrir `configs.py`**. Descomente a definição de `BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS` e <code>BIG_QUERY_QUERY</code>. Você deve substituir o valor da região neste arquivo pelos valores corretos para o seu projeto GCP.

> **Observação: você PRECISA definir sua região e ID do projeto GCP no arquivo `configs.py` antes de continuar.**

**Mude o diretório para um nível acima.** Clique no nome do diretório acima da lista de arquivos. O nome do diretório é o nome do pipeline que é `my_pipeline` se você não mudou.

**Dê um duplo-clique para abrir `kubeflow_runner.py`**. Descomente os dois argumentos, `query` e `beam_pipeline_args`, para a função `create_pipeline`.

Agora o pipeline está pronto para usar o BigQuery como fonte de exemplos. Atualize o pipeline e crie uma nova execução como fizemos nas etapas 5 e 6.

#### Atualize o pipeline e execute-o novamente

```python
# Update the pipeline
!tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

!tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}
```

### Experimente o Dataflow

Vários [componentes TFX usam o Apache Beam](https://www.tensorflow.org/tfx/guide/beam) para implementar pipelines paralelos de dados, o que significa que você poderá distribuir cargas de trabalho de processamento de dados usando o [Google Cloud Dataflow](https://cloud.google.com/dataflow/). Nesta etapa, configuraremos o orquestrador Kubeflow para usar o dataflow como back-end de processamento de dados para o Apache Beam.

> **Observação:** se a API Dataflow ainda não estiver ativada, você poderá ativá-la usando o console ou na CLI usando este comando (por exemplo, no Cloud Shell):

```bash
# Select your project:
gcloud config set project YOUR_PROJECT_ID

# Get a list of services that you can enable in your project:
gcloud services list --available | grep Dataflow

# If you don't see dataflow.googleapis.com listed, that means you haven't been
# granted access to enable the Dataflow API.  See your account adminstrator.

# Enable the Dataflow service:

gcloud services enable dataflow.googleapis.com
```

> **Observação:** a velocidade de execução pode ser limitada pela cota padrão [do Google Compute Engine (GCE)](https://cloud.google.com/compute). Recomendamos definir uma cota suficiente para aproximadamente 250 VMs do Dataflow: **250 CPUs, 250 endereços IP e 62.500 GB de disco permanente**. Para mais detalhes, consulte a documentação [Cota do GCE](https://cloud.google.com/compute/quotas) e [Cota do Dataflow](https://cloud.google.com/dataflow/quotas). Se você estiver bloqueado pela cota de endereço IP, usar um [`worker_type`](https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#setting-other-cloud-dataflow-pipeline-options) maior reduzirá o número de IPs necessários.

**Dê um duplo-clique em `pipeline` para alterar o diretório e dê outro duplo-clique para abrir `configs.py`**. Descomente a definição de `GOOGLE_CLOUD_REGION` e `DATAFLOW_BEAM_PIPELINE_ARGS`.

**Mude o diretório para um nível acima.** Clique no nome do diretório acima da lista de arquivos. O nome do diretório é o nome do pipeline que é `my_pipeline` se você não mudou.

**Dê um duplo-clique para abrir `kubeflow_runner.py`**. Descomente `beam_pipeline_args`. (Certifique-se também de comentar o `beam_pipeline_args` atual que você adicionou na Etapa 7.)

#### Atualize o pipeline e execute-o novamente

```python
# Update the pipeline
!tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

!tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}
```

Você pode encontrar seus jobs de Dataflow no [Dataflow no Cloud Console](http://console.cloud.google.com/dataflow).

### Experimente o Cloud AI Platform Training and Prediction com o KFP

O TFX interopera com vários serviços gerenciados do GCP, como o [Cloud AI Platform for Training and Prediction](https://cloud.google.com/ai-platform/). Você pode configurar seu componente `Trainer` para usar o Cloud AI Platform Training, um serviço gerenciado para treinamento de modelos de ML. Além disso, quando seu modelo estiver criado e pronto para ser servido, você poderá *enviá-lo* para o Cloud AI Platform Prediction para disponibilizar o serviço. Nesta etapa, configuraremos nosso componente `Trainer` e `Pusher` para usar os serviços Cloud AI Platform.

Antes de editar arquivos, talvez seja necessário ativar a *API AI Platform Training &amp; Prediction*.

**Dê um duplo-clique em `pipeline` para alterar o diretório e dê outro duplo-clique para abrir `configs.py`**. Descomente a definição de `GOOGLE_CLOUD_REGION`, `GCP_AI_PLATFORM_TRAINING_ARGS` e `GCP_AI_PLATFORM_SERVING_ARGS`. Usaremos nossa imagem de container personalizada para treinar um modelo no Cloud AI Platform Training, portanto, precisamos definir `masterConfig.imageUri` em `GCP_AI_PLATFORM_TRAINING_ARGS` com o mesmo valor de `CUSTOM_TFX_IMAGE` acima.

**Mude o diretório um nível acima e dê um duplo-clique para abrir `kubeflow_runner.py`**. Descomente `ai_platform_training_args` e `ai_platform_serving_args`.

> Observação: se você receber um erro de permissão na etapa de treinamento, talvez seja necessário fornecer permissões de visualização de objetos de armazenamento para a conta de serviço do Cloud Machine Learning Engine (AI Platform Prediction &amp; Training). Mais informações estão disponíveis na [Documentação do Container Registry](https://cloud.google.com/container-registry/docs/access-control#grant).

#### Atualize o pipeline e execute-o novamente

```python
# Update the pipeline
!tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

!tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}
```

Você pode encontrar seus jobs de treinamento em [Cloud AI Platform Jobs](https://console.cloud.google.com/ai-platform/jobs). Se o pipeline for concluído com sucesso, você encontrará seu modelo em [Cloud AI Platform Models](https://console.cloud.google.com/ai-platform/models).

## 14. Use seus próprios dados

Neste tutorial, você criou um pipeline para um modelo usando o dataset Chicago Taxi. Agora tente colocar seus próprios dados no pipeline. Seus dados podem ser armazenados em qualquer lugar onde o pipeline possa acessá-los, incluindo Google Cloud Storage, BigQuery ou arquivos CSV.

Você precisará modificar a definição do pipeline para acomodar seus dados.

### Se seus dados estiverem armazenados em arquivos

1. Modifique `DATA_PATH` em `kubeflow_runner.py`, indicando o local.

### Se seus dados estiverem armazenados no BigQuery

1. Modifique `BIG_QUERY_QUERY` em configs.py para sua instrução de pesquisa.
2. Adicione características em `models/features.py`.
3. Modifique `models/preprocessing.py` para <a>transformar os dados de entrada para treinamento</a>.
4. Modifique `models/keras/model.py` e `models/keras/constants.py` para <a>descrever seu modelo de ML</a>.

### Saiba mais sobre o Trainer

Consulte o [guia do componente Trainer](https://www.tensorflow.org/tfx/guide/trainer) para mais detalhes sobre pipelines de treinamento.

## Limpeza

Para limpar todos os recursos do Google Cloud usados ​​neste projeto, [exclua o projeto do Google Cloud](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects) usado no tutorial.

Alternativamente, você pode limpar recursos individualmente acessando cada console: - [Google Cloud Storage](https://console.cloud.google.com/storage) - [Google Container Registry](https://console.cloud.google.com/gcr) - [Google Kubernetes Engine](https://console.cloud.google.com/kubernetes)
