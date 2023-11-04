# Usando a interface de linha de comando do TFX

A interface de linha de comando (CLI) do TFX executa uma gama completa de ações de pipeline usando orquestradores de pipeline, como Kubeflow Pipelines e Vertex Pipelines. O orquestrador local também pode ser usado para desenvolvimento ou depuração mais rápidos. Apache Beam e Apache Airflow são suportados como recursos experimentais. Por exemplo, você pode usar a CLI para:

- Criar, atualizar e excluir pipelines.
- Executar um pipeline e monitorar a execução em vários orquestradores.
- Listar pipelines e execuções de pipeline.

Observação: o TFX CLI atualmente não oferece garantias de compatibilidade. A interface CLI pode mudar à medida que novas versões forem lançadas.

## Sobre a CLI do TFX

A CLI do TFX é instalada como parte do pacote TFX. Todos os comandos CLI seguem a estrutura abaixo:

<pre class="devsite-terminal">
tfx &lt;var&gt;command-group&lt;/var&gt; &lt;var&gt;command&lt;/var&gt; &lt;var&gt;flags&lt;/var&gt;
</pre>

As seguintes opções de <var>command-group</var> são atualmente suportadas:

- [tfx pipeline](#tfx-pipeline) - Criar e gerenciar pipelines TFX.
- [tfx run](#tfx-run) - Criar e gerenciar execuções de pipelines TFX em várias plataformas de orquestração.
- [tfx template](#tfx-template-experimental) - Comandos experimentais para listar e copiar templates de pipelines TFX.

Cada grupo de comandos fornece um conjunto de <var>comandos</var>. Siga as instruções nas seções [comandos de pipeline](#tfx-pipeline), [comandos de execução](#tfx-run) e [comandos de template](#tfx-template-experimental) para saber mais sobre como usar esses comandos.

Aviso: atualmente nem todos os comandos são suportados em todos os orquestradores. Esses comandos mencionam explicitamente os mecanismos suportados.

Os sinalizadores permitem que você passe argumentos para os comandos da CLI. As palavras nos sinalizadores são separadas ou por um hífen (`-`) ou por sublinhado (`_`). Por exemplo, o sinalizador do nome do pipeline pode ser especificado como `--pipeline-name` ou `--pipeline_name`. Este documento especifica sinalizadores com sublinhados por questões de brevidade. Saiba mais sobre os [<var>sinalizadores</var> usados ​​na CLI do TFX](#understanding-tfx-cli-flags).

## tfx pipeline

A estrutura dos comandos no grupo de comandos `tfx pipeline` é a seguinte:

<pre class="devsite-terminal">
tfx pipeline &lt;var&gt;command&lt;/var&gt; &lt;var&gt;required-flags&lt;/var&gt; [&lt;var&gt;optional-flags&lt;/var&gt;]
</pre>

Use as seções a seguir para saber mais sobre os comandos no grupo de comandos do `tfx pipeline`.

### create

Cria um novo pipeline no orquestrador fornecido.

Uso

<pre class="devsite-click-to-copy devsite-terminal">
tfx pipeline create --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; \
--build_image --build_base_image=&lt;var&gt;build-base-image&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>O caminho para o arquivo de configuração do pipeline.</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       (Opcional.) Endpoint do serviço da API Kubeflow Pipelines. O endpoint do seu serviço de API do Kubeflow Pipelines é o mesmo que a URL do painel do Kubeflow Pipelines. O valor do seu endpoint deve ser algo como:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional.) O orquestrador a ser usado para o pipeline. O valor de engine deve corresponder a um dos seguintes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: define o motor como Kubeflow</li>
      <li>
<strong>local</strong>: define o motor como orquestrador local</li>
      <li>
<strong>vertex</strong>: define o motor como Vertex Pipelines</li>
      <li>
<strong>airflow</strong>: (experimental) define o motor como Apache Airflow</li>
      <li>
<strong>beam</strong>: (experimental) define o motor como Apache Beam</li>
    </ul>
    <p>       Se o engine não estiver configurado, ele será detectado automaticamente com base no ambiente.</p>
    <p>       ** Nota importante: O orquestrador exigido pelo DagRunner no arquivo de configuração do pipeline deve corresponder ao motor selecionado ou detectado automaticamente. A detecção automática do motor é baseada no ambiente do usuário. Se o Apache Airflow e o Kubeflow Pipelines não estiverem instalados, o orquestrador local será usado por padrão.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>     (Opcional.) ID do cliente para endpoint protegido por IAP ao usar Kubeflow Pipelines.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>     (Opcional.) Namespace Kubernetes para conectar-se à API Kubeflow Pipelines. Se o namespace não for especificado, o valor padrão será <code>kubeflow</code>.</dd>


  <dt>--build_image</dt>
  <dd>
    <p>       (Opcional.) Quando o <var>engine</var> é <strong>kubeflow</strong> ou <strong>vertex</strong>, o TFX cria uma imagem de container para seu pipeline, se especificado. `Dockerfile` no diretório atual será usado e o TFX irá gerar um automaticamente se não existir.</p>
    <p>       A imagem construída será enviada para o registro remoto especificado em `KubeflowDagRunnerConfig` ou `KubeflowV2DagRunnerConfig`.</p>
  </dd>
  <dt>--build_base_image=<var>build-base-image</var>
</dt>
  <dd>
    <p>       (Opcional.) Quando o <var>engine</var> é <strong>kubeflow</strong>, o TFX cria uma imagem de container para seu pipeline. A imagem base do build especifica a imagem de container base a ser usada ao criar a imagem do container do pipeline.</p>
  </dd>



#### Exemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx pipeline create --engine=kubeflow --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--build_image
</pre>

Local:

<pre class="devsite-terminal">
tfx pipeline create --engine=local --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

Vertex:

<pre class="devsite-terminal">
tfx pipeline create --engine=vertex --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; \
--build_image
</pre>

Para detectar automaticamente o motor do ambiente do usuário, simplesmente evite usar o sinalizador engine como no exemplo abaixo. Para mais detalhes, veja a seção sobre sinalizadores.

<pre class="devsite-terminal">
tfx pipeline create --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

### update

Atualiza um pipeline existente no orquestrador fornecido.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx pipeline update --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --build_image]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>O caminho para o arquivo de configuração do pipeline.</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       (Opcional.) Endpoint do serviço da API Kubeflow Pipelines. O endpoint do seu serviço de API do Kubeflow Pipelines é o mesmo que a URL do painel do Kubeflow Pipelines. O valor do seu endpoint deve ser algo como:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional.) O orquestrador a ser usado para o pipeline. O valor de engine deve corresponder a um dos seguintes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: define o motor como Kubeflow</li>
      <li>
<strong>local</strong>: define o motor como orquestrador local</li>
      <li>
<strong>vertex</strong>: define o motor como Vertex Pipelines</li>
      <li>
<strong>airflow</strong>: (experimental) define o motor como Apache Airflow</li>
      <li>
<strong>beam</strong>: (experimental) define o motor como Apache Beam</li>
    </ul>
    <p>       Se o engine não estiver configurado, ele será detectado automaticamente com base no ambiente.</p>
    <p>       ** Nota importante: O orquestrador exigido pelo DagRunner no arquivo de configuração do pipeline deve corresponder ao motor selecionado ou detectado automaticamente. A detecção automática do motor é baseada no ambiente do usuário. Se o Apache Airflow e o Kubeflow Pipelines não estiverem instalados, o orquestrador local será usado por padrão.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>     (Opcional.) ID do cliente para endpoint protegido por IAP.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>     (Opcional.) Namespace Kubernetes para conectar-se à API Kubeflow Pipelines. Se o namespace não for especificado, o valor padrão será <code>kubeflow</code>.</dd>
  <dt>--build_image</dt>
  <dd>
    <p>       (Opcional.) Quando o <var>engine</var> é <strong>kubeflow</strong> ou <strong>vertex</strong>, o TFX cria uma imagem de container para seu pipeline, se especificado. `Dockerfile` no diretório atual será usado.</p>
    <p>       A imagem construída será enviada para o registro remoto especificado em `KubeflowDagRunnerConfig` ou `KubeflowV2DagRunnerConfig`.</p>
  </dd>



#### Exemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx pipeline update --engine=kubeflow --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--build_image
</pre>

Local:

<pre class="devsite-terminal">
tfx pipeline update --engine=local --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

Vertex:

<pre class="devsite-terminal">
tfx pipeline update --engine=vertex --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; \
--build_image
</pre>

### compile

Compila o arquivo de configuração do pipeline para criar um arquivo de workflow no Kubeflow e executa as seguintes verificações durante a compilação:

1. Verifica se o caminho do pipeline é válido.
2. Verifica se os dados do pipeline foram extraídos com sucesso do arquivo de configuração do pipeline.
3. Verifica se o DagRunner na configuração do pipeline corresponde ao motor.
4. Verifica se o arquivo de workflow foi criado com sucesso no caminho do pacote fornecido (somente para Kubeflow).

Recomenda-se usar antes de criar ou atualizar um pipeline.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx pipeline compile --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--engine=&lt;var&gt;engine&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>O caminho para o arquivo de configuração do pipeline.</dd>
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional.) O orquestrador a ser usado para o pipeline. O valor de engine deve corresponder a um dos seguintes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: define o motor como Kubeflow</li>
      <li>
<strong>local</strong>: define o motor como orquestrador local</li>
      <li>
<strong>vertex</strong>: define o motor como Vertex Pipelines</li>
      <li>
<strong>airflow</strong>: (experimental) define o motor como Apache Airflow</li>
      <li>
<strong>beam</strong>: (experimental) define o motor como Apache Beam</li>
    </ul>
    <p>       Se o engine não estiver configurado, ele será detectado automaticamente com base no ambiente.</p>
    <p>       ** Nota importante: O orquestrador exigido pelo DagRunner no arquivo de configuração do pipeline deve corresponder ao motor selecionado ou detectado automaticamente. A detecção automática do motor é baseada no ambiente do usuário. Se o Apache Airflow e o Kubeflow Pipelines não estiverem instalados, o orquestrador local será usado por padrão.</p>
  </dd>
</dl>

#### Exemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx pipeline compile --engine=kubeflow --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

Local:

<pre class="devsite-terminal">
tfx pipeline compile --engine=local --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

Vertex:

<pre class="devsite-terminal">
tfx pipeline compile --engine=vertex --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

### delete

Exclui um pipeline do orquestrador fornecido.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx pipeline delete --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>O caminho para o arquivo de configuração do pipeline.</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       (Opcional.) Endpoint do serviço da API Kubeflow Pipelines. O endpoint do seu serviço de API do Kubeflow Pipelines é o mesmo que a URL do painel do Kubeflow Pipelines. O valor do seu endpoint deve ser algo como:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional.) O orquestrador a ser usado para o pipeline. O valor de engine deve corresponder a um dos seguintes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: define o motor como Kubeflow</li>
      <li>
<strong>local</strong>: define o motor como orquestrador local</li>
      <li>
<strong>vertex</strong>: define o motor como Vertex Pipelines</li>
      <li>
<strong>airflow</strong>: (experimental) define o motor como Apache Airflow</li>
      <li>
<strong>beam</strong>: (experimental) define o motor como Apache Beam</li>
    </ul>
    <p>       Se o engine não estiver configurado, ele será detectado automaticamente com base no ambiente.</p>
    <p>       ** Nota importante: O orquestrador exigido pelo DagRunner no arquivo de configuração do pipeline deve corresponder ao motor selecionado ou detectado automaticamente. A detecção automática do motor é baseada no ambiente do usuário. Se o Apache Airflow e o Kubeflow Pipelines não estiverem instalados, o orquestrador local será usado por padrão.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>     (Opcional.) ID do cliente para endpoint protegido por IAP.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>     (Opcional.) Namespace Kubernetes para conectar-se à API Kubeflow Pipelines. Se o namespace não for especificado, o valor padrão será <code>kubeflow</code>.</dd>



#### Exemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx pipeline delete --engine=kubeflow --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

Local:

<pre class="devsite-terminal">
tfx pipeline delete --engine=local --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;
</pre>

Vertex:

<pre class="devsite-terminal">
tfx pipeline delete --engine=vertex --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;
</pre>

### list

Lista todos os pipelines no orquestrador fornecido.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx pipeline list [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       (Opcional.) Endpoint do serviço da API Kubeflow Pipelines. O endpoint do seu serviço de API do Kubeflow Pipelines é o mesmo que a URL do painel do Kubeflow Pipelines. O valor do seu endpoint deve ser algo como:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional.) O orquestrador a ser usado para o pipeline. O valor de engine deve corresponder a um dos seguintes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: define o motor como Kubeflow</li>
      <li>
<strong>local</strong>: define o motor como orquestrador local</li>
      <li>
<strong>vertex</strong>: define o motor como Vertex Pipelines</li>
      <li>
<strong>airflow</strong>: (experimental) define o motor como Apache Airflow</li>
      <li>
<strong>beam</strong>: (experimental) define o motor como Apache Beam</li>
    </ul>
    <p>       Se o engine não estiver configurado, ele será detectado automaticamente com base no ambiente.</p>
    <p>       ** Nota importante: O orquestrador exigido pelo DagRunner no arquivo de configuração do pipeline deve corresponder ao motor selecionado ou detectado automaticamente. A detecção automática do motor é baseada no ambiente do usuário. Se o Apache Airflow e o Kubeflow Pipelines não estiverem instalados, o orquestrador local será usado por padrão.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>     (Opcional.) ID do cliente para endpoint protegido por IAP.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>     (Opcional.) Namespace Kubernetes para conectar-se à API Kubeflow Pipelines. Se o namespace não for especificado, o valor padrão será <code>kubeflow</code>.</dd>



#### Exemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx pipeline list --engine=kubeflow --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

Local:

<pre class="devsite-terminal">
tfx pipeline list --engine=local
</pre>

Vertex:

<pre class="devsite-terminal">
tfx pipeline list --engine=vertex
</pre>

## tfx run

A estrutura dos comandos no grupo de comandos `tfx run` é a seguinte:

<pre class="devsite-terminal">
tfx run &lt;var&gt;command&lt;/var&gt; &lt;var&gt;required-flags&lt;/var&gt; [&lt;var&gt;optional-flags&lt;/var&gt;]
</pre>

Use as seções a seguir para saber mais sobre os comandos do grupo de comandos `tfx run`.

### create

Cria uma nova instância de execução para um pipeline no orquestrador. Para o Kubeflow, é usada a versão mais recente do pipeline no cluster.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx run create --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>O nome do pipeline.</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       (Opcional.) Endpoint do serviço da API Kubeflow Pipelines. O endpoint do seu serviço de API do Kubeflow Pipelines é o mesmo que a URL do painel do Kubeflow Pipelines. O valor do seu endpoint deve ser algo como:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional.) O orquestrador a ser usado para o pipeline. O valor de engine deve corresponder a um dos seguintes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: define o motor como Kubeflow</li>
      <li>
<strong>local</strong>: define o motor como orquestrador local</li>
      <li>
<strong>vertex</strong>: define o motor como Vertex Pipelines</li>
      <li>
<strong>airflow</strong>: (experimental) define o motor como Apache Airflow</li>
      <li>
<strong>beam</strong>: (experimental) define o motor como Apache Beam</li>
    </ul>
    <p>       Se o engine não estiver configurado, ele será detectado automaticamente com base no ambiente.</p>
    <p>       ** Nota importante: O orquestrador exigido pelo DagRunner no arquivo de configuração do pipeline deve corresponder ao motor selecionado ou detectado automaticamente. A detecção automática do motor é baseada no ambiente do usuário. Se o Apache Airflow e o Kubeflow Pipelines não estiverem instalados, o orquestrador local será usado por padrão.</p>
  </dd>


  <dt>--runtime_parameter=<var>parameter-name</var>=<var>parameter-value</var>
</dt>
  <dd>     (Opcional.) Define um valor de parâmetro de tempo de execução. Pode ser definido várias vezes para configurar os valores de múltiplas variáveis. Aplicável apenas aos motores `airflow`, `kubeflow` e `vertex`.</dd>


  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>     (Opcional.) ID do cliente para endpoint protegido por IAP.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
  <dd>     (Opcional.) Namespace Kubernetes para conectar-se à API Kubeflow Pipelines. Se o namespace não for especificado, o valor padrão será <code>kubeflow</code>.</dd>


  <dt>--project=<var>GCP-project-id</var>
</dt>
  <dd>     (Obrigatório para Vertex.) ID do projeto GCP para o pipeline vertex.</dd>


  <dt>--region=<var>GCP-region</var>
</dt>
  <dd>     (Obrigatório para Vertex.) Nome da região GCP, como us-central1. Veja a [documentação da Vertex](https://cloud.google.com/vertex-ai/docs/general/locations) para saber quais são as regiões disponíveis.</dd>





#### Exemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx run create --engine=kubeflow --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

Local:

<pre class="devsite-terminal">
tfx run create --engine=local --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;
</pre>

Vertex:

<pre class="devsite-terminal">
tfx run create --engine=vertex --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
  --runtime_parameter=&lt;var&gt;var_name&lt;/var&gt;=&lt;var&gt;var_value&lt;/var&gt; \
  --project=&lt;var&gt;gcp-project-id&lt;/var&gt; --region=&lt;var&gt;gcp-region&lt;/var&gt;
</pre>

### terminate

Interrompe a execução de um determinado pipeline.

** Observação importante: atualmente suportado apenas no Kubeflow.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx run terminate --run_id=&lt;var&gt;run-id&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--run_id=<var>run-id</var>
</dt>
  <dd>Identificador exclusivo para uma execução de pipeline.</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       (Opcional.) Endpoint do serviço da API Kubeflow Pipelines. O endpoint do seu serviço de API do Kubeflow Pipelines é o mesmo que a URL do painel do Kubeflow Pipelines. O valor do seu endpoint deve ser algo como:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional.) O orquestrador a ser usado para o pipeline. O valor de engine deve corresponder a um dos seguintes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: define o motor como Kubeflow</li>
    </ul>
    <p>       Se o engine não estiver configurado, ele será detectado automaticamente com base no ambiente.</p>
    <p>       ** Nota importante: O orquestrador exigido pelo DagRunner no arquivo de configuração do pipeline deve corresponder ao motor selecionado ou detectado automaticamente. A detecção automática do motor é baseada no ambiente do usuário. Se o Apache Airflow e o Kubeflow Pipelines não estiverem instalados, o orquestrador local será usado por padrão.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>     (Opcional.) ID do cliente para endpoint protegido por IAP.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>     (Opcional.) Namespace Kubernetes para conectar-se à API Kubeflow Pipelines. Se o namespace não for especificado, o valor padrão será <code>kubeflow</code>.</dd>



#### Exemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx run delete --engine=kubeflow --run_id=&lt;var&gt;run-id&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

### list

Lista todas as execuções de um pipeline.

** Observação importante: atualmente não é suportado em motores Local e Apache Beam.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx run list --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>O nome do pipeline.</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       (Opcional.) Endpoint do serviço da API Kubeflow Pipelines. O endpoint do seu serviço de API do Kubeflow Pipelines é o mesmo que a URL do painel do Kubeflow Pipelines. O valor do seu endpoint deve ser algo como:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional.) O orquestrador a ser usado para o pipeline. O valor de engine deve corresponder a um dos seguintes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: define o motor como Kubeflow</li>
      <li>
<strong>airflow</strong>: (experimental) define o motor como Apache Airflow</li>
    </ul>
    <p>       Se o engine não estiver configurado, ele será detectado automaticamente com base no ambiente.</p>
    <p>       ** Nota importante: O orquestrador exigido pelo DagRunner no arquivo de configuração do pipeline deve corresponder ao motor selecionado ou detectado automaticamente. A detecção automática do motor é baseada no ambiente do usuário. Se o Apache Airflow e o Kubeflow Pipelines não estiverem instalados, o orquestrador local será usado por padrão.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>     (Opcional.) ID do cliente para endpoint protegido por IAP.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>     (Opcional.) Namespace Kubernetes para conectar-se à API Kubeflow Pipelines. Se o namespace não for especificado, o valor padrão será <code>kubeflow</code>.</dd>



#### Exemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx run list --engine=kubeflow --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

### status

Retorna o status atual de uma execução.

** Observação importante: atualmente não é suportado em motores Local e Apache Beam.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx run status --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; --run_id=&lt;var&gt;run-id&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>O nome do pipeline.</dd>
  <dt>--run_id=<var>run-id</var>
</dt>
  <dd>Identificador exclusivo para uma execução de pipeline.</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       (Opcional.) Endpoint do serviço da API Kubeflow Pipelines. O endpoint do seu serviço de API do Kubeflow Pipelines é o mesmo que a URL do painel do Kubeflow Pipelines. O valor do seu endpoint deve ser algo como:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional.) O orquestrador a ser usado para o pipeline. O valor de engine deve corresponder a um dos seguintes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: define o motor como Kubeflow</li>
      <li>
<strong>airflow</strong>: (experimental) define o motor como Apache Airflow</li>
    </ul>
    <p>       Se o engine não estiver configurado, ele será detectado automaticamente com base no ambiente.</p>
    <p>       ** Nota importante: O orquestrador exigido pelo DagRunner no arquivo de configuração do pipeline deve corresponder ao motor selecionado ou detectado automaticamente. A detecção automática do motor é baseada no ambiente do usuário. Se o Apache Airflow e o Kubeflow Pipelines não estiverem instalados, o orquestrador local será usado por padrão.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>     (Opcional.) ID do cliente para endpoint protegido por IAP.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>     (Opcional.) Namespace Kubernetes para conectar-se à API Kubeflow Pipelines. Se o namespace não for especificado, o valor padrão será <code>kubeflow</code>.</dd>



#### Exemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx run status --engine=kubeflow --run_id=&lt;var&gt;run-id&lt;/var&gt; --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

### delete

Exclui uma execução de um determinado pipeline.

** Observação importante: atualmente suportado apenas no Kubeflow.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx run delete --run_id=&lt;var&gt;run-id&lt;/var&gt; [--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;]
</pre>

<dl>
  <dt>--run_id=<var>run-id</var>
</dt>
  <dd>Identificador exclusivo para uma execução de pipeline.</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       (Opcional.) Endpoint do serviço da API Kubeflow Pipelines. O endpoint do seu serviço de API do Kubeflow Pipelines é o mesmo que a URL do painel do Kubeflow Pipelines. O valor do seu endpoint deve ser algo como:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional.) O orquestrador a ser usado para o pipeline. O valor de engine deve corresponder a um dos seguintes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: define o motor como Kubeflow</li>
    </ul>
    <p>       Se o engine não estiver configurado, ele será detectado automaticamente com base no ambiente.</p>
    <p>       ** Nota importante: O orquestrador exigido pelo DagRunner no arquivo de configuração do pipeline deve corresponder ao motor selecionado ou detectado automaticamente. A detecção automática do motor é baseada no ambiente do usuário. Se o Apache Airflow e o Kubeflow Pipelines não estiverem instalados, o orquestrador local será usado por padrão.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>     (Opcional.) ID do cliente para endpoint protegido por IAP.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>     (Opcional.) Namespace Kubernetes para conectar-se à API Kubeflow Pipelines. Se o namespace não for especificado, o valor padrão será <code>kubeflow</code>.</dd>



#### Exemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx run delete --engine=kubeflow --run_id=&lt;var&gt;run-id&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

## tfx template [Experimental]

A estrutura dos comandos no grupo de comandos `tfx template` é a seguinte:

<pre class="devsite-terminal">
tfx template &lt;var&gt;command&lt;/var&gt; &lt;var&gt;required-flags&lt;/var&gt; [&lt;var&gt;optional-flags&lt;/var&gt;]
</pre>

Use as seções a seguir para saber mais sobre os comandos no grupo de comandos `tfx template`. O template é um recurso experimental e está sujeito a alterações a qualquer momento.

### list

Veja a lista dos templates de pipeline TFX disponíveis:

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx template list
</pre>

### copy

Copia um template para o diretório de destino.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx template copy --model=&lt;var&gt;model&lt;/var&gt; --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
--destination_path=&lt;var&gt;destination-path&lt;/var&gt;
</pre>

<dl>
  <dt>--model=<var>model</var>
</dt>
  <dd>O nome do modelo criado pelo template do pipeline.</dd>
  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>O nome do pipeline.</dd>
  <dt>--destination_path=<var>destination-path</var>
</dt>
  <dd>O caminho para onde copiar o template.</dd>
</dl>

## Understanding TFX CLI Flags

### Common flags

<dl>
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       O orquestrador a ser usado para o pipeline. O valor de engine deve corresponder a um dos seguintes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: define o motor como Kubeflow</li>
      <li>
<strong>local</strong>: define o motor como orquestrador local</li>
      <li>
<strong>vertex</strong>: define o motor como Vertex Pipelines</li>
      <li>
<strong>airflow</strong>: (experimental) define o motor como Apache Airflow</li>
      <li>
<strong>beam</strong>: (experimental) define o motor como Apache Beam</li>
    </ul>
    <p>       Se o engine não estiver configurado, ele será detectado automaticamente com base no ambiente.</p>
    <p>       ** Nota importante: O orquestrador exigido pelo DagRunner no arquivo de configuração do pipeline deve corresponder ao motor selecionado ou detectado automaticamente. A detecção automática do motor é baseada no ambiente do usuário. Se o Apache Airflow e o Kubeflow Pipelines não estiverem instalados, o orquestrador local será usado por padrão.</p>
  </dd>
</dl>

  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>O nome do pipeline.</dd>


  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>O caminho para o arquivo de configuração do pipeline.</dd>


  <dt>--run_id=<var>run-id</var>
</dt>
  <dd>Identificador exclusivo para uma execução de pipeline.</dd>





### Kubeflow specific flags

<dl>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>Endpoint do serviço da API Kubeflow Pipelines. O endpoint do seu serviço de API do Kubeflow Pipelines é o mesmo que a URL do painel do Kubeflow Pipelines. O valor do seu endpoint deve ser algo como:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  


  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>ID do cliente para endpoint protegido por IAP.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>Namespace Kubernetes para conectar-se à API Kubeflow Pipelines. Se o namespace não for especificado, o valor padrão será <code>kubeflow</code>.</dd>



## Generated files by TFX CLI

Quando pipelines são criados e executados, vários arquivos são gerados para gerenciamento do pipeline.

- ${HOME}/tfx/local, beam, airflow, vertex
    - Os metadados do pipeline lidos da configuração são armazenados em `${HOME}/tfx/${ORCHESTRATION_ENGINE}/${PIPELINE_NAME}`. Este local pode ser personalizado definindo uma variável de ambiente como `AIRFLOW_HOME` ou `KUBEFLOW_HOME`. Este comportamento poderá ser alterado em versões futuras. Este diretório é usado para armazenar informações de pipeline, incluindo IDs de pipeline no cluster Kubeflow Pipelines, que são necessários para criar execuções ou atualizar pipelines.
    - Antes do TFX 0.25, esses arquivos estavam localizados em `${HOME}/${ORCHESTRATION_ENGINE}`. No TFX 0.25, os arquivos no local antigo serão movidos automaticamente para o novo local para uma migração tranquila.
    - A partir do TFX 0.27, o kubeflow não cria esses arquivos de metadados no sistema de arquivos local. No entanto, veja abaixo outros arquivos que o kubeflow cria.
- (somente Kubeflow) Dockerfile e uma imagem de container
    - O Kubeflow Pipelines requer dois tipos de entrada para um pipeline. Esses arquivos são gerados pelo TFX no diretório atual.
    - Um é uma imagem de container que será usada para executar componentes no pipeline. Esta imagem de container é criada quando um pipeline para Kubeflow Pipelines é criado ou atualizado com o sinalizador `--build-image`. A CLI do TFX gerará `Dockerfile` se não existir e criará e enviará uma imagem de container para o registro especificado em KubeflowDagRunnerConfig.
