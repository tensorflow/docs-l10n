## Testando o pipeline usando Stub Executors

### Introdução

**Você deve concluir o tutorial [template.ipynb](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/tutorials/tfx/template.ipynb) até o *Passo 6* para prosseguir com este tutorial.**

Este documento fornecerá instruções para testar um pipeline TensorFlow Extended (TFX) usando o `BaseStubExecuctor`, que gera artefatos falsos usando os dados de teste dourados. O objetivo é que os usuários substituam os executores que não desejam testar, para que possam economizar tempo na execução dos executores reais. O stub executor é fornecido com o pacote TFX Python em `tfx.experimental.pipeline_testing.base_stub_executor`.

Este tutorial serve como uma extensão do tutorial `template.ipynb`, portanto, você também usará [o dataset Taxi Trips](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew) lançado pela cidade de Chicago. Recomendamos fortemente que você tente modificar os componentes antes de utilizar executores stub.

### 1. Registre as saídas do pipeline no Google Cloud Storage

Primeiro precisamos registrar as saídas do pipeline para que os stub executors possam copiar os artefatos das saídas registradas.

Como este tutorial pressupõe que você concluiu `template.ipynb` até o passo 6, uma execução bem-sucedida do pipeline deve ter sido salva no [MLMD](https://www.tensorflow.org/tfx/guide/mlmd). As informações de execução no MLMD podem ser acessadas usando o servidor gRPC.

Abra um Terminal e execute os seguintes comandos:

1. Gere um arquivo kubeconfig com credenciais apropriadas: `bash gcloud container clusters get-credentials $cluster_name --zone $compute_zone --project $gcp_project_id{/code0} {code1}$compute_zone` é a região do engine gcp e `$gcp_project_id` é o ID de projeto do seu projeto GCP.

2. Configure o encaminhamento de portas para a conexão com o MLMD: `bash nohup kubectl port-forward deployment/metadata-grpc-deployment -n $namespace $port:8080 &` `$namespace{/code1} é o namespace do cluster e {code2}$port` é qualquer porta não utilizada que será usada para encaminhamento de porta.

3. Clone o repositório tfx GitHub. Dentro do diretório tfx, execute o seguinte comando:

```bash
python tfx/experimental/pipeline_testing/pipeline_recorder.py \
--output_dir=gs://<gcp_project_id>-kubeflowpipelines-default/testdata \
--host=$host \
--port=$port \
--pipeline_name=$pipeline_name
```

`$output_dir` deve ser definido como um caminho no Google Cloud Storage onde as saídas do pipeline serão registradas. Portanto, substitua `<gcp_project_id>` pelo ID do projeto do GCP.

`$host{/code0} e {code1}$port` são o nome do host e a porta do servidor grpc de metadados para conectar-se ao MLMD. `$port` deve ser definido como o número da porta que você usou para encaminhamento de porta e você pode definir "localhost" para o nome do host.

No tutorial `template.ipynb`, o nome do pipeline é definido como "my_pipeline" por padrão, então defina `pipeline_name="my_pipeline"`. Se você modificou o nome do pipeline ao executar o tutorial do modelo, deverá modificar `--pipeline_name` de acordo.

### 2. Habilite Stub Executors no Kubeflow DAG Runner

Primeiro, certifique-se de que o modelo predefinido foi copiado para o diretório do projeto usando o comando CLI `tfx template copy`. É necessário editar os dois arquivos a seguir nos arquivos-fonte copiados.

1. Crie um arquivo chamado `stub_component_launcher.py` no diretório onde o kubeflow_dag_runner.py está localizado e coloque o seguinte conteúdo nele.

    ```python
    from tfx.experimental.pipeline_testing import base_stub_component_launcher
    from pipeline import configs

    class StubComponentLauncher(
        base_stub_component_launcher.BaseStubComponentLauncher):
      pass

    # GCS directory where KFP outputs are recorded
    test_data_dir = "gs://{}/testdata".format(configs.GCS_BUCKET_NAME)
    # TODO: customize self.test_component_ids to test components, replacing other
    # component executors with a BaseStubExecutor.
    test_component_ids = ['Trainer']
    StubComponentLauncher.initialize(
        test_data_dir=test_data_dir,
        test_component_ids=test_component_ids)
    ```

    OBSERVAÇÃO: Este launcher de componente stub não pode ser definido em `kubeflow_dag_runner.py` porque a classe do launcher é importada pelo path do módulo.

2. Defina os IDs dos componentes como uma lista de IDs dos componentes que serão testados (em outras palavras, os executores de outros componentes são substituídos por BaseStubExecutor).

3. Abra `kubeflow_dag_runner.py`. Adicione a seguinte instrução import na parte superior para usar a classe `StubComponentLauncher` que acabamos de adicionar.

    ```python
    import stub_component_launcher
    ```

4. Em `kubeflow_dag_runner.py`, adicione a classe `StubComponentLauncher` a `supported_launcher_class` de `KubeflowDagRunnerConfig` para permitir o lançamento dos stub executors:

    ```python
    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        supported_launcher_classes=[
            stub_component_launcher.StubComponentLauncher
        ],
    ```

### 3. Atualize e execute o pipeline com stub executors

Atualize o pipeline existente com a definição de pipeline modificada com os stub executors.

```bash
tfx pipeline update --pipeline-path=kubeflow_dag_runner.py \
  --endpoint=$endpoint --engine=kubeflow
```

`$endpoint` deve ser definido como o endpoint do seu cluster KFP.

Execute o comando a seguir para criar uma nova execução do pipeline atualizado.

```bash
tfx run create --pipeline-name $pipeline_name --endpoint=$endpoint \
  --engine=kubeflow
```

## Limpeza

Use o comando `fg` para acessar o encaminhamento de porta em segundo plano e depois ctrl-C para encerrá-lo. Você pode excluir o diretório contendo as saídas de pipeline gravadas usando `gsutil -m rm -R $output_dir`.

Para limpar todos os recursos do Google Cloud usados ​​neste projeto, [exclua o projeto do Google Cloud](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects) usado no tutorial.

Alternativamente, você pode limpar recursos individualmente acessando cada console: - [Google Cloud Storage](https://console.cloud.google.com/storage) - [Google Container Registry](https://console.cloud.google.com/gcr) - [Google Kubernetes Engine](https://console.cloud.google.com/kubernetes)
