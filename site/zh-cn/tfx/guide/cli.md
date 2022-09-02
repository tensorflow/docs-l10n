# 使用 TFX 命令行接口

TFX 命令行接口 (CLI) 使用流水线编排器（例如 Kubeflow Pipelines、Vertex Pipelines）执行各种流水线操作。本地编排器还可以用于加快开发或调试速度。支持 Apache Beam 和 Apache Airflow 作为实验性功能。例如，您可以使用 CLI 执行以下操作：

- 创建、更新和删除流水线。
- 运行流水线并监视在各种编排器上的运行。
- 列出流水线和流水线运行。

注：TFX CLI 目前不提供兼容性保证。随着新版本的发布，CLI 接口可能会更改。

## 关于 TFX CLI

TFX CLI 作为 TFX 软件包的一部分进行安装。所有 CLI 命令都遵循以下结构：

<pre class="devsite-terminal">
tfx &lt;var&gt;command-group&lt;/var&gt; &lt;var&gt;command&lt;/var&gt; &lt;var&gt;flags&lt;/var&gt;
</pre>

目前支持以下 <var>command-group</var> 选项：

- [tfx pipeline](#tfx-pipeline) - 创建并管理 TFX 流水线。
- [tfx run](#tfx-run) - 在各种编排平台上创建和管理 TFX 流水线的运行。
- [tfx template](#tfx-template-experimental) - 用于列出和复制 TFX 流水线模板的实验性命令。

每个命令组都提供一组 <var>commands</var>。请遵循[流水线命令](#tfx-pipeline)、[运行命令](#tfx-run)和[模板命令](#tfx-template-experimental)部分中的说明，详细了解这些命令的用法。

警告：目前并非每个编排器都支持所有命令。这些命令明确提到了支持的引擎。

您可以通过标记将参数传递到 CLI 命令中。标记中的单词用连字符 (`-`) 或下划线 (`_`) 分隔。例如，流水线名称标记可以指定为 `--pipeline-name` 或 `--pipeline_name`。为了简洁起见，本文档将使用下划线指定标记。详细了解[在 TFX CLI 中使用的 <var>flags</var>](#understanding-tfx-cli-flags)。

## tfx pipeline

`tfx pipeline` 命令组中的命令结构如下：

<pre class="devsite-terminal">tfx pipeline &lt;var&gt;command&lt;/var&gt; &lt;var&gt;required-flags&lt;/var&gt; [&lt;var&gt;optional-flags&lt;/var&gt;]
</pre>

使用以下各个部分详细了解 `tfx pipeline` 命令组中的命令。

### create

在给定的编排器中创建新的流水线。

用法：

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline create --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; \
--build_image --build_base_image=&lt;var&gt;build-base-image&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var> </dt>
  <dd>流水线配置文件的路径。</dd>
  <dt>--endpoint=<var>endpoint</var> </dt>
  <dd>
    <p>（可选）Kubeflow Pipelines API 服务的端点。Kubeflow Pipelines API 服务的端点与 Kubeflow Pipelines 信息中心的网址相同。您的端点值应类似于：</p>
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

  
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>（可选）用于流水线的编排器。引擎的值必须与以下值匹配：</p>
    <ul>
      <li> <strong>kubeflow</strong>：将引擎设置为 Kubeflow</li>
      <li> <strong>local</strong>：将引擎设置为本地编排器</li>
      <li> <strong>vertex</strong>：将引擎设置为 Vertex Pipelines</li>
      <li> <strong>airflow</strong>：（实验性）将引擎设置为 Apache Airflow</li>
      <li> <strong>beam</strong> ：（实验性）将引擎设置为 Apache Beam</li>
    </ul>
    <p>如果未设置引擎，则会根据环境自动检测引擎。</p>
    <p>** 重要说明：DagRunner 在流水线配置文件中所需的编排器必须与所选或自动检测到的引擎匹配。引擎自动检测基于用户环境。如果未安装 Apache Airflow 和 Kubeflow Pipelines，则默认使用本地编排器。</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var> </dt>
  <dd>（可选）使用 Kubeflow Pipelines 时受 IAP 保护的端点的客户端 ID。</dd>


  <dt>--namespace=<var>namespace</var>   </dt>
<dd>     (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API.     If the namespace is not specified, the value defaults to     <code>kubeflow</code>.   </dd>


  <dt>--build_image</dt>
  <dd>
    <p>（可选）当 <var>engine</var> 为 <strong>kubeflow</strong> 或 <strong>vertex</strong> 时，TFX 会为您的流水线创建容器镜像（如果已指定）。将使用当前目录下的 `Dockerfile`，如果不存在，则 TFX 会自动生成。</p>
    <p>构建的镜像将被推送到在 `KubeflowDagRunnerConfig` 或 `KubeflowV2DagRunnerConfig` 中指定的远程注册表。</p>
  </dd>
  <dt>--build_base_image=<var>build-base-image</var> </dt>
  <dd>
    <p>（可选）当 <var>engine</var> 为 <strong>kubeflow</strong> 时，TFX 会为您的流水线创建容器镜像。build-base-image 指定要在构建流水线容器镜像时使用的基础容器镜像。</p>
  </dd>



#### 示例：

Kubeflow：

<pre class="devsite-terminal">tfx pipeline create --engine=kubeflow --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--build_image
</pre>

本地：

<pre class="devsite-terminal">tfx pipeline create --engine=local --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

Vertex：

<pre class="devsite-terminal">tfx pipeline create --engine=vertex --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; \
--build_image
</pre>

要从用户环境自动检测引擎，只需避免使用引擎标记（如下面的示例所示）。有关更多详细信息，请查看标记部分。

<pre class="devsite-terminal">tfx pipeline create --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

### update

更新给定编排器中的现有流水线。

用法：

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline update --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --build_image]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var> </dt>
  <dd>流水线配置文件的路径。</dd>
  <dt>--endpoint=<var>endpoint</var> </dt>
  <dd>
    <p>（可选）Kubeflow Pipelines API 服务的端点。Kubeflow Pipelines API 服务的端点与 Kubeflow Pipelines 信息中心的网址相同。您的端点值应类似于：</p>
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

  
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>（可选）用于流水线的编排器。引擎的值必须与以下值匹配：</p>
    <ul>
      <li> <strong>kubeflow</strong>：将引擎设置为 Kubeflow</li>
      <li> <strong>local</strong>：将引擎设置为本地编排器</li>
      <li> <strong>vertex</strong>：将引擎设置为 Vertex Pipelines</li>
      <li> <strong>airflow</strong>：（实验性）将引擎设置为 Apache Airflow</li>
      <li> <strong>beam</strong> ：（实验性）将引擎设置为 Apache Beam</li>
    </ul>
    <p>如果未设置引擎，则会根据环境自动检测引擎。</p>
    <p>** 重要说明：DagRunner 在流水线配置文件中所需的编排器必须与所选或自动检测到的引擎匹配。引擎自动检测基于用户环境。如果未安装 Apache Airflow 和 Kubeflow Pipelines，则默认使用本地编排器。</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var> </dt>
  <dd>（可选）受 IAP 保护的端点的客户端 ID。</dd>


  <dt>--namespace=<var>namespace</var> </dt>
<dd>（可选）要连接到 Kubeflow Pipelines API 的 Kubernetes 命名空间。如果未指定命名空间，则值将默认为 <code>kubeflow</code>。</dd>
  <dt>--build_image</dt>
  <dd>
    <p>（可选）当 <var>engine</var> 为 <strong>kubeflow</strong> 或 <strong>vertex</strong> 时，TFX 会为您的流水线创建容器镜像（如果已指定）。将使用当前目录中的 `Dockerfile`。</p>
    <p>构建的镜像将被推送到在 `KubeflowDagRunnerConfig` 或 `KubeflowV2DagRunnerConfig` 中指定的远程注册表。</p>
  </dd>



#### 示例：

Kubeflow：

<pre class="devsite-terminal">tfx pipeline update --engine=kubeflow --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--build_image
</pre>

本地：

<pre class="devsite-terminal">tfx pipeline update --engine=local --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

Vertex：

<pre class="devsite-terminal">tfx pipeline update --engine=vertex --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; \
--build_image
</pre>

### compile

编译流水线配置文件，以在 Kubeflow 中创建工作流文件并在编译时执行以下检查：

1. 检查流水线路径是否有效。
2. 检查是否从流水线配置文件中成功提取了流水线详细信息。
3. 检查流水线配置中的 DagRunner 是否与引擎匹配。
4. 检查是否在提供的软件包路径中成功创建了工作流文件（仅适用于 Kubeflow）。

建议在创建或更新流水线之前使用。

用法：

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline compile --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--engine=&lt;var&gt;engine&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var> </dt>
  <dd>流水线配置文件的路径。</dd>
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>（可选）用于流水线的编排器。引擎的值必须与以下值匹配：</p>
    <ul>
      <li> <strong>kubeflow</strong>：将引擎设置为 Kubeflow</li>
      <li> <strong>local</strong>：将引擎设置为本地编排器</li>
      <li> <strong>vertex</strong>：将引擎设置为 Vertex Pipelines</li>
      <li> <strong>airflow</strong>：（实验性）将引擎设置为 Apache Airflow</li>
      <li> <strong>beam</strong> ：（实验性）将引擎设置为 Apache Beam</li>
    </ul>
    <p>如果未设置引擎，则会根据环境自动检测引擎。</p>
    <p>** 重要说明：DagRunner 在流水线配置文件中所需的编排器必须与所选或自动检测到的引擎匹配。引擎自动检测基于用户环境。如果未安装 Apache Airflow 和 Kubeflow Pipelines，则默认使用本地编排器。</p>
  </dd>
</dl>

#### 示例：

Kubeflow：

<pre class="devsite-terminal">tfx pipeline compile --engine=kubeflow --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

本地：

<pre class="devsite-terminal">tfx pipeline compile --engine=local --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

Vertex：

<pre class="devsite-terminal">tfx pipeline compile --engine=vertex --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

### delete

从给定的编排器中删除流水线。

用法：

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline delete --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var> </dt>
  <dd>流水线配置文件的路径。</dd>
  <dt>--endpoint=<var>endpoint</var> </dt>
  <dd>
    <p>（可选）Kubeflow Pipelines API 服务的端点。Kubeflow Pipelines API 服务的端点与 Kubeflow Pipelines 信息中心的网址相同。您的端点值应类似于：</p>
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

  
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>（可选）用于流水线的编排器。引擎的值必须与以下值匹配：</p>
    <ul>
      <li> <strong>kubeflow</strong>：将引擎设置为 Kubeflow</li>
      <li> <strong>local</strong>：将引擎设置为本地编排器</li>
      <li> <strong>vertex</strong>：将引擎设置为 Vertex Pipelines</li>
      <li> <strong>airflow</strong>：（实验性）将引擎设置为 Apache Airflow</li>
      <li> <strong>beam</strong> ：（实验性）将引擎设置为 Apache Beam</li>
    </ul>
    <p>如果未设置引擎，则会根据环境自动检测引擎。</p>
    <p>** 重要说明：DagRunner 在流水线配置文件中所需的编排器必须与所选或自动检测到的引擎匹配。引擎自动检测基于用户环境。如果未安装 Apache Airflow 和 Kubeflow Pipelines，则默认使用本地编排器。</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var> </dt>
  <dd>（可选）受 IAP 保护的端点的客户端 ID。</dd>


  <dt>--namespace=<var>namespace</var> </dt>
<dd>（可选）要连接到 Kubeflow Pipelines API 的 Kubernetes 命名空间。如果未指定命名空间，则值将默认为 <code>kubeflow</code>。</dd>



#### 示例：

Kubeflow：

<pre class="devsite-terminal">tfx pipeline delete --engine=kubeflow --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

本地：

<pre class="devsite-terminal">tfx pipeline delete --engine=local --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;
</pre>

Vertex：

<pre class="devsite-terminal">tfx pipeline delete --engine=vertex --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;
</pre>

### list

列出给定编排器中的所有流水线。

用法：

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline list [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--endpoint=<var>endpoint</var> </dt>
  <dd>
    <p>（可选）Kubeflow Pipelines API 服务的端点。Kubeflow Pipelines API 服务的端点与 Kubeflow Pipelines 信息中心的网址相同。您的端点值应类似于：</p>
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

  
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>（可选）用于流水线的编排器。引擎的值必须与以下值匹配：</p>
    <ul>
      <li> <strong>kubeflow</strong>：将引擎设置为 Kubeflow</li>
      <li> <strong>local</strong>：将引擎设置为本地编排器</li>
      <li> <strong>vertex</strong>：将引擎设置为 Vertex Pipelines</li>
      <li> <strong>airflow</strong>：（实验性）将引擎设置为 Apache Airflow</li>
      <li> <strong>beam</strong> ：（实验性）将引擎设置为 Apache Beam</li>
    </ul>
    <p>如果未设置引擎，则会根据环境自动检测引擎。</p>
    <p>** 重要说明：DagRunner 在流水线配置文件中所需的编排器必须与所选或自动检测到的引擎匹配。引擎自动检测基于用户环境。如果未安装 Apache Airflow 和 Kubeflow Pipelines，则默认使用本地编排器。</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var> </dt>
  <dd>     (Optional.) Client ID for IAP protected endpoint.   </dd>


  <dt>--namespace=<var>namespace</var> </dt>
<dd>（可选）要连接到 Kubeflow Pipelines API 的 Kubernetes 命名空间。如果未指定命名空间，则值将默认为 <code>kubeflow</code>。</dd>



#### 示例：

Kubeflow：

<pre class="devsite-terminal">tfx pipeline list --engine=kubeflow --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

本地：

<pre class="devsite-terminal">tfx pipeline list --engine=local
</pre>

Vertex：

<pre class="devsite-terminal">tfx pipeline list --engine=vertex
</pre>

## tfx run

`tfx run` 命令组中的命令结构如下：

<pre class="devsite-terminal">tfx run &lt;var&gt;command&lt;/var&gt; &lt;var&gt;required-flags&lt;/var&gt; [&lt;var&gt;optional-flags&lt;/var&gt;]
</pre>

使用以下各个部分详细了解 `tfx run` 命令组中的命令。

### create

在编排器中为流水线创建新的运行实例。对于 Kubeflow，会使用集群中流水线的最新版本。

用法：

<pre class="devsite-click-to-copy devsite-terminal">tfx run create --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var> </dt>
  <dd>流水线的名称。</dd>
  <dt>--endpoint=<var>endpoint</var> </dt>
  <dd>
    <p>（可选）Kubeflow Pipelines API 服务的端点。Kubeflow Pipelines API 服务的端点与 Kubeflow Pipelines 信息中心的网址相同。您的端点值应类似于：</p>
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

  
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>（可选）用于流水线的编排器。引擎的值必须与以下值匹配：</p>
    <ul>
      <li> <strong>kubeflow</strong>：将引擎设置为 Kubeflow</li>
      <li> <strong>local</strong>：将引擎设置为本地编排器</li>
      <li> <strong>vertex</strong>：将引擎设置为 Vertex Pipelines</li>
      <li> <strong>airflow</strong>：（实验性）将引擎设置为 Apache Airflow</li>
      <li> <strong>beam</strong> ：（实验性）将引擎设置为 Apache Beam</li>
    </ul>
    <p>如果未设置引擎，则会根据环境自动检测引擎。</p>
    <p>** 重要说明：DagRunner 在流水线配置文件中所需的编排器必须与所选或自动检测到的引擎匹配。引擎自动检测基于用户环境。如果未安装 Apache Airflow 和 Kubeflow Pipelines，则默认使用本地编排器。</p>
  </dd>


  <dt>--runtime_parameter=<var>parameter-name</var>=<var>parameter-value</var> </dt>
  <dd>（可选）设置运行时参数值。可以多次设置来设置多个变量的值。仅适用于 `airflow`、`kubeflow` 和 `vertex` 引擎。</dd>


  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>（可选）受 IAP 保护的端点的客户端 ID。</dd>


  <dt>--namespace=<var>namespace</var> </dt>
  <dd>（可选）要连接到 Kubeflow Pipelines API 的 Kubernetes 命名空间。如果未指定命名空间，则值将默认为 <code>kubeflow</code>。</dd>


  <dt>--project=<var>GCP-project-id</var> </dt>
  <dd>（Vertex 必需）Vertex 流水线的 GCP 项目 ID。</dd>


  <dt>--region=<var>GCP-region</var> </dt>
  <dd>（Vertex 必需）GCP 区域名称，例如 us-central1。有关可用区域，请参阅 [Vertex 文档] (https://cloud.google.com/vertex-ai/docs/general/locations)。</dd>





#### 示例：

Kubeflow：

<pre class="devsite-terminal">tfx run create --engine=kubeflow --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

本地：

<pre class="devsite-terminal">tfx run create --engine=local --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;
</pre>

Vertex：

<pre class="devsite-terminal">tfx run create --engine=vertex --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
  --runtime_parameter=&lt;var&gt;var_name&lt;/var&gt;=&lt;var&gt;var_value&lt;/var&gt; \
  --project=&lt;var&gt;gcp-project-id&lt;/var&gt; --region=&lt;var&gt;gcp-region&lt;/var&gt;
</pre>

### terminate

停止给定流水线的运行。

** 重要说明：目前仅在 Kubeflow 中受支持。

用法：

<pre class="devsite-click-to-copy devsite-terminal">tfx run terminate --run_id=&lt;var&gt;run-id&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--run_id=<var>run-id</var> </dt>
  <dd>流水线运行的唯一标识符。</dd>
  <dt>--endpoint=<var>endpoint</var> </dt>
  <dd>
    <p>（可选）Kubeflow Pipelines API 服务的端点。Kubeflow Pipelines API 服务的端点与 Kubeflow Pipelines 信息中心的网址相同。您的端点值应类似于：</p>
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

  
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>（可选）用于流水线的编排器。引擎的值必须与以下值匹配：</p>
    <ul>
      <li> <strong>kubeflow</strong>：将引擎设置为 Kubeflow</li>
    </ul>
    <p>如果未设置引擎，则会根据环境自动检测引擎。</p>
    <p>** 重要说明：DagRunner 在流水线配置文件中所需的编排器必须与所选或自动检测到的引擎匹配。引擎自动检测基于用户环境。如果未安装 Apache Airflow 和 Kubeflow Pipelines，则默认使用本地编排器。</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var> </dt>
  <dd>（可选）受 IAP 保护的端点的客户端 ID。</dd>


  <dt>--namespace=<var>namespace</var> </dt>
<dd>（可选）要连接到 Kubeflow Pipelines API 的 Kubernetes 命名空间。如果未指定命名空间，则值默认为 <code>kubeflow</code>。</dd>



#### 示例：

Kubeflow：

<pre class="devsite-terminal">tfx run delete --engine=kubeflow --run_id=&lt;var&gt;run-id&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

### list

列出流水线的所有运行。

** 重要说明：目前在本地和 Apache Beam 中不受支持。

用法：

<pre class="devsite-click-to-copy devsite-terminal">tfx run list --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var> </dt>
  <dd>流水线的名称。</dd>
  <dt>--endpoint=<var>endpoint</var> </dt>
  <dd>
    <p>（可选）Kubeflow Pipelines API 服务的端点。Kubeflow Pipelines API 服务的端点与 Kubeflow Pipelines 信息中心的网址相同。您的端点值应类似于：</p>
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
    <p>（可选）用于流水线的编排器。引擎的值必须与以下值匹配：</p>
    <ul>
      <li> <strong>kubeflow</strong>：将引擎设置为 Kubeflow</li>
      <li> <strong>airflow</strong>：（实验性）将引擎设置为 Apache Airflow</li>
    </ul>
    <p>如果未设置引擎，则会根据环境自动检测引擎。</p>
    <p>** 重要说明：DagRunner 在流水线配置文件中所需的编排器必须与所选或自动检测到的引擎匹配。引擎自动检测基于用户环境。如果未安装 Apache Airflow 和 Kubeflow Pipelines，则默认使用本地编排器。</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var> </dt>
  <dd>     (Optional.) Client ID for IAP protected endpoint.   </dd>


  <dt>--namespace=<var>namespace</var> </dt>
<dd>（可选）要连接到 Kubeflow Pipelines API 的 Kubernetes 命名空间。如果未指定命名空间，则值将默认为 <code>kubeflow</code>。</dd>



#### 示例：

Kubeflow：

<pre class="devsite-terminal">tfx run list --engine=kubeflow --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

### status

返回运行的当前状态。

** 重要说明：目前在本地和 Apache Beam 中不受支持。

用法：

<pre class="devsite-click-to-copy devsite-terminal">tfx run status --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; --run_id=&lt;var&gt;run-id&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var> </dt>
  <dd>流水线的名称。</dd>
  <dt>--run_id=<var>run-id</var> </dt>
  <dd>流水线运行的唯一标识符。</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint       of your Kubeflow Pipelines API service is the same as URL of the Kubeflow       Pipelines dashboard. Your endpoint value should be something like:     </p>
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

  
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>（可选）用于流水线的编排器。引擎的值必须与以下值匹配：</p>
    <ul>
      <li> <strong>kubeflow</strong>：将引擎设置为 Kubeflow</li>
      <li> <strong>airflow</strong>：（实验性）将引擎设置为 Apache Airflow</li>
    </ul>
    <p>如果未设置引擎，则会根据环境自动检测引擎。</p>
    <p>** 重要说明：DagRunner 在流水线配置文件中所需的编排器必须与所选或自动检测到的引擎匹配。引擎自动检测基于用户环境。如果未安装 Apache Airflow 和 Kubeflow Pipelines，则默认使用本地编排器。</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var> </dt>
  <dd>（可选）受 IAP 保护的端点的客户端 ID。</dd>


  <dt>--namespace=<var>namespace</var> </dt>
<dd>     (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API.     If the namespace is not specified, the value defaults to     <code>kubeflow</code>.   </dd>



#### 示例：

Kubeflow：

<pre class="devsite-terminal">tfx run status --engine=kubeflow --run_id=&lt;var&gt;run-id&lt;/var&gt; --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

### delete

删除给定流水线的运行。

** 重要说明：目前仅在 Kubeflow 中受支持。

用法：

<pre class="devsite-click-to-copy devsite-terminal">tfx run delete --run_id=&lt;var&gt;run-id&lt;/var&gt; [--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;]
</pre>

<dl>
  <dt>--run_id=<var>run-id</var> </dt>
  <dd>流水线运行的唯一标识符。</dd>
  <dt>--endpoint=<var>endpoint</var> </dt>
  <dd>
    <p>（可选）Kubeflow Pipelines API 服务的端点。Kubeflow Pipelines API 服务的端点与 Kubeflow Pipelines 信息中心的网址相同。您的端点值应类似于：</p>
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

  
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>（可选）用于流水线的编排器。引擎的值必须与以下值匹配：</p>
    <ul>
      <li> <strong>kubeflow</strong>：将引擎设置为 Kubeflow</li>
    </ul>
    <p>如果未设置引擎，则会根据环境自动检测引擎。</p>
    <p>** 重要说明：DagRunner 在流水线配置文件中所需的编排器必须与所选或自动检测到的引擎匹配。引擎自动检测基于用户环境。如果未安装 Apache Airflow 和 Kubeflow Pipelines，则默认使用本地编排器。</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var> </dt>
  <dd>（可选）受 IAP 保护的端点的客户端 ID。</dd>


  <dt>--namespace=<var>namespace</var> </dt>
<dd>（可选）要连接到 Kubeflow Pipelines API 的 Kubernetes 命名空间。如果未指定命名空间，则值将默认为 <code>kubeflow</code>。</dd>



#### 示例：

Kubeflow：

<pre class="devsite-terminal">tfx run delete --engine=kubeflow --run_id=&lt;var&gt;run-id&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

## tfx template [实验性]

`tfx template` 命令组中的命令结构如下：

<pre class="devsite-terminal">tfx template &lt;var&gt;command&lt;/var&gt; &lt;var&gt;required-flags&lt;/var&gt; [&lt;var&gt;optional-flags&lt;/var&gt;]
</pre>

使用以下部分详细了解 `tfx template` 命令组中的命令。模板是一项实验性功能，随时可能更改。

### list

列出可用的 TFX 流水线模板。

用法：

<pre class="devsite-click-to-copy devsite-terminal">tfx template list</pre>

### copy

将模板复制到目标目录。

用法：

<pre class="devsite-click-to-copy devsite-terminal">tfx template copy --model=&lt;var&gt;model&lt;/var&gt; --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
--destination_path=&lt;var&gt;destination-path&lt;/var&gt;
</pre>

<dl>
  <dt>--model=<var>model</var> </dt>
  <dd>由流水线模板构建的模型的名称。</dd>
  <dt>--pipeline_name=<var>pipeline-name</var> </dt>
  <dd>流水线的名称。</dd>
  <dt>--destination_path=<var>destination-path</var> </dt>
  <dd>要将模板复制到的路径。</dd>
</dl>

## 了解 TFX CLI 标记

### 通用标记

<dl>
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>用于流水线的编排器。引擎的值必须与以下值匹配：</p>
    <ul>
      <li> <strong>kubeflow</strong>：将引擎设置为 Kubeflow</li>
      <li> <strong>local</strong>：将引擎设置为本地编排器</li>
      <li> <strong>vertex</strong>：将引擎设置为 Vertex Pipelines</li>
      <li> <strong>airflow</strong>：（实验性）将引擎设置为 Apache Airflow</li>
      <li> <strong>beam</strong> ：（实验性）将引擎设置为 Apache Beam</li>
    </ul>
    <p>如果未设置引擎，则会根据环境自动检测引擎。</p>
    <p>** 重要说明：DagRunner 在流水线配置文件中所需的编排器必须与所选或自动检测到的引擎匹配。引擎自动检测基于用户环境。如果未安装 Apache Airflow 和 Kubeflow Pipelines，则默认使用本地编排器。</p>
  </dd>
</dl>

  <dt>--pipeline_name=<var>pipeline-name</var> </dt>
  <dd>流水线的名称。</dd>


  <dt>--pipeline_path=<var>pipeline-path</var> </dt>
  <dd>流水线配置文件的路径。</dd>


  <dt>--run_id=<var>run-id</var> </dt>
  <dd>流水线运行的唯一标识符。</dd>





### Kubeflow 专用标记

<dl>
  <dt>--endpoint=<var>endpoint</var> </dt>
  <dd>
    <p>Kubeflow Pipelines API 服务的端点。Kubeflow Pipelines API 服务的端点与 Kubeflow Pipelines 信息中心的网址相同。您的端点值应类似于：</p>
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

  


  <dt>--iap_client_id=<var>iap-client-id</var> </dt>
  <dd>受 IAP 保护的端点的客户端 ID。</dd>


  <dt>--namespace=<var>namespace</var> </dt>
<dd>要连接到 Kubeflow Pipelines API 的 Kubernetes 命名空间。如果未指定命名空间，则值将默认为 <code>kubeflow</code>。</dd>



## 由 TFX CLI 生成的文件

创建并运行流水线后，会生成几个文件用于流水线管理。

- ${HOME}/tfx/local、beam、airflow、vertex
    - 从配置中读取的流水线元数据存储在 `${HOME}/tfx/${ORCHESTRATION_ENGINE}/${PIPELINE_NAME}` 下。可以通过设置环境变量（如 `AIRFLOW_HOME` 或 `KUBEFLOW_HOME`）来自定义此位置。在未来的版本中可能会改变此行为。此目录用于存储流水线信息，包括创建运行或更新流水线所需的 Kubeflow Pipelines 集群中的流水线 ID。
    - 在 TFX 0.25 之前，这些文件位于 `${HOME}/${ORCHESTRATION_ENGINE}` 下。在 TFX 0.25 中，旧位置中的文件将自动移动到新位置，以便顺利迁移。
    - 从 TFX 0.27 开始，Kubeflow 不会在本地文件系统中创建这些元数据文件。但是，请参阅以下内容，了解 Kubeflow 创建的其他文件。
- （仅限 Kubeflow）Dockerfile 和容器镜像
    - Kubeflow Pipelines 需要两种流水线输入。这些文件由 TFX 在当前目录下生成。
    - 一种是容器镜像，用于在流水线中运行组件。此容器镜像在使用 `--build-image` 标志创建或更新 Kubeflow Pipelines 的流水线时构建。如果不存在，TFX CLI 将生成 `Dockerfile`，并构建容器镜像并将其推送到 KubeflowDagRunnerConfig 中指定的注册表。
