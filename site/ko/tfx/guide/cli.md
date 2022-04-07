# TFX 명령줄 인터페이스 사용하기

TFX 명령줄 인터페이스(CLI)는 Kubeflow Pipelines, Vertex Pipelines와 같은 파이프라인 오케스트레이터를 사용하여 모든 범위의 파이프라인 작업을 수행합니다. 더 빠른 개발 또는 디버깅을 위해 로컬 오케스트레이터를 사용할 수도 있습니다. Apache Beam 및 Apache airflow는 실험적 기능으로 지원됩니다. 예를 들어 CLI를 사용하여 다음을 수행할 수 있습니다.

- 파이프라인을 생성, 업데이트 및 삭제합니다.
- 파이프라인을 실행하고 다양한 오케스트레이터에서 실행을 모니터링합니다.
- 파이프라인 및 파이프라인 실행을 나열합니다.

참고: TFX CLI는 현재 호환성 보장을 제공하지 않습니다. 새 버전이 출시되면 CLI 인터페이스가 변경될 수 있습니다.

## TFX CLI 정보

TFX CLI는 TFX 패키지의 일부로 설치됩니다. 모든 CLI 명령은 아래 구조를 따릅니다.

<pre class="devsite-terminal">tfx &lt;var&gt;command-group&lt;/var&gt; &lt;var&gt;command&lt;/var&gt; &lt;var&gt;flags&lt;/var&gt;</pre>

현재 지원되는 <var>명령 그룹</var> 옵션은 다음과 같습니다.

- [tfx pipeline](#tfx-pipeline) - TFX 파이프라인을 만들고 관리합니다.
- [tfx run](#tfx-run) - 다양한 조정 플랫폼에서 TFX 파이프라인 실행을 생성하고 관리합니다.
- [tfx template](#tfx-template-experimental) - TFX 파이프라인 템플릿을 나열하고 복사하기 위한 실험 단계에 있는 명령입니다.

각 명령 그룹은 <var>commands</var> 세트를 제공합니다. [파이프라인 명령](#tfx-pipeline), [실행 명령](#tfx-run) 및 [템플릿 명령](#tfx-template-experimental) 섹션의 지침에 따라 이들 명령을 사용하는 방법에 대해 자세히 알아보세요.

경고: 현재, 모든 오케스트레이터에서 모든 명령이 지원되는 것은 아닙니다. 이들 명령은 지원되는 엔진을 명시적으로 언급합니다.

플래그를 사용하면 CLI 명령에 인수를 전달할 수 있습니다. 플래그의 단어는 하이픈(`-`) 또는 밑줄(`_`)로 구분됩니다. 예를 들어, 파이프라인 이름 플래그는 `--pipeline-name` 또는 `--pipeline_name`으로 지정할 수 있습니다. 이 문서에서는 간단히 하기 위해 밑줄로 플래그를 지정합니다. [TFX CLI에서 사용되는 <var>플래그</var>](#understanding-tfx-cli-flags)에 대해 자세히 알아보세요.

## tfx pipeline

`tfx pipeline` 명령 그룹의 명령 구조는 다음과 같습니다.

<pre class="devsite-terminal">tfx pipeline &lt;var&gt;command&lt;/var&gt; &lt;var&gt;required-flags&lt;/var&gt; [&lt;var&gt;optional-flags&lt;/var&gt;]</pre>

다음 섹션을 사용하여 `tfx pipeline` 명령 그룹의 명령에 대해 자세히 알아보세요.

### create

지정된 오케스트레이터에서 새 파이프라인을 만듭니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline create --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; \
--build_image --build_base_image=&lt;var&gt;build-base-image&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var> </dt>
  <dd>파이프라인 구성 파일의 경로입니다.</dd>
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
  <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li> <strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
      <li> <strong>local</strong>: 엔진을 로컬 오케스트레이터로 설정합니다.</li>
      <li>
<strong>vertex</strong>: 엔진을 정점 파이프라인으로 설정합니다.</li>
      <li>
<strong>airflow</strong>: (실험적) 엔진을 Apache Airflow로 설정합니다.</li>
      <li>
<strong>beam</strong>: (실험적) 엔진을 Apache Beam으로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, 기본적으로 로컬 오케스트레이터가 사용됩니다.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var> </dt>
  <dd>(선택 사항) Kubeflow Pipelines를 사용할 때 IAP로 보호되는 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var> </dt>
<dd>(선택 사항) Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>


  <dt>--build_image</dt>
  <dd>
    <p>(선택 사항) <var>engine</var>이 <strong>kubeflow</strong> 또는 <strong>vertex</strong>인 경우 TFX는 지정된 경우 파이프라인에 대한 컨테이너 이미지를 생성합니다. 현재 디렉터리에 있는 'Dockerfile'이 사용되며, TFX가 없으면 자동으로 생성합니다.</p>
    <p>빌드된 이미지는 `KubeflowDagRunnerConfig` 또는 `KubeflowV2DagRunnerConfig`에 지정된 원격 레지스트리로 푸시됩니다.</p>
  </dd>
  <dt>--build_base_image=<var>build-base-image</var> </dt>
  <dd>
    <p>(선택 사항) <var>engine</var>이 <strong>kubeflow</strong>이면 TFX는 파이프라인에 대한 컨테이너 이미지를 생성합니다. 빌드 기본 이미지는 파이프라인 컨테이너 이미지를 빌드할 때 사용할 기본 컨테이너 이미지를 지정합니다.</p>
  </dd>



#### 예:

Kubeflow:

<pre class="devsite-terminal">tfx pipeline create --engine=kubeflow --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--build_image
</pre>

Local:

<pre class="devsite-terminal">tfx pipeline create --engine=local --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;</pre>

Vertex:

<pre class="devsite-terminal">tfx pipeline create --engine=vertex --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; \
--build_image
</pre>

사용자 환경에서 엔진을 자동 감지하려면 아래 예와 같이 engine 플래그를 사용하지 않으면 됩니다. 자세한 내용은 플래그 섹션을 확인하세요.

<pre class="devsite-terminal">tfx pipeline create --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

### update

지정된 오케스트레이터에서 기존 파이프라인을 업데이트합니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline update --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --build_image]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var> </dt>
  <dd>파이프라인 구성 파일의 경로입니다.</dd>
  <dt>--endpoint=<var>endpoint</var> </dt>
  <dd>
    <p>(선택 사항) Kubeflow Pipelines API 서비스의 끝점입니다. Kubeflow Pipelines API 서비스의 끝점은 Kubeflow Pipelines 대시보드의 URL과 동일합니다. 끝점 값은 다음과 같아야 합니다.</p>
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
  <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
      <li> <strong>local</strong>: 엔진을 로컬 오케스트레이터로 설정합니다.</li>
      <li>
<strong>vertex</strong>: 엔진을 정점 파이프라인으로 설정합니다.</li>
      <li>
<strong>airflow</strong>: (실험적) 엔진을 Apache Airflow로 설정합니다.</li>
      <li>
<strong>beam</strong>: (실험적) 엔진을 Apache Beam으로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, 기본적으로 로컬 오케스트레이터가 사용됩니다.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var> </dt>
  <dd>(선택 사항) IAP 보호 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var> </dt>
<dd>(선택 사항) Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>
  <dt>--build_image</dt>
  <dd>
    <p>(선택 사항) <var>engine</var>이 <strong>kubeflow</strong> 또는 <strong>vertex</strong>이면 TFX는 지정된 경우 파이프라인에 대한 컨테이너 이미지를 생성합니다. 현재 디렉터리의 'Dockerfile'이 사용됩니다.</p>
    <p>빌드된 이미지는 `KubeflowDagRunnerConfig` 또는 `KubeflowV2DagRunnerConfig`에 지정된 원격 레지스트리로 푸시됩니다.</p>
  </dd>



#### 예:

Kubeflow:

<pre class="devsite-terminal">tfx pipeline update --engine=kubeflow --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--build_image
</pre>

Local:

<pre class="devsite-terminal">tfx pipeline update --engine=local --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;</pre>

Vertex:

<pre class="devsite-terminal">tfx pipeline update --engine=vertex --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; \
--build_image
</pre>

### compile

파이프라인 구성 파일을 컴파일하여 Kubeflow에서 워크플로 파일을 만들고 컴파일하는 동안 다음 검사를 수행합니다.

1. 파이프라인 경로가 유효한지 확인합니다.
2. Checks if the pipeline details are extracted successfully from the pipeline config file.
3. 파이프라인 구성의 DagRunner가 엔진과 일치하는지 확인합니다.
4. Checks if the workflow file is created successfully in the package path provided (only for Kubeflow).

파이프라인을 생성하거나 업데이트하기 전에 사용하는 것이 좋습니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline compile --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--engine=&lt;var&gt;engine&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var> </dt>
  <dd>파이프라인 구성 파일의 경로입니다.</dd>
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
      <li> <strong>local</strong>: 엔진을 로컬 오케스트레이터로 설정합니다.</li>
      <li>
<strong>vertex</strong>: 엔진을 정점 파이프라인으로 설정합니다.</li>
      <li>
<strong>airflow</strong>: (실험적) 엔진을 Apache Airflow로 설정합니다.</li>
      <li>
<strong>beam</strong>: (실험적) 엔진을 Apache Beam으로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, 기본적으로 로컬 오케스트레이터가 사용됩니다.</p>
  </dd>
</dl>

#### 예:

Kubeflow:

<pre class="devsite-terminal">tfx pipeline compile --engine=kubeflow --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

Local:

<pre class="devsite-terminal">tfx pipeline compile --engine=local --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;</pre>

Vertex:

<pre class="devsite-terminal">tfx pipeline compile --engine=vertex --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

### delete

지정된 오케스트레이터에서 파이프라인을 삭제합니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline delete --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \&lt;br&gt;--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var> </dt>
  <dd>The path to the pipeline configuration file.</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>(선택 사항) Kubeflow Pipelines API 서비스의 끝점입니다. Kubeflow Pipelines API 서비스의 끝점은 Kubeflow Pipelines 대시보드의 URL과 동일합니다. 끝점 값은 다음과 같아야 합니다.</p>
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
  <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
      <li> <strong>local</strong>: 엔진을 로컬 오케스트레이터로 설정합니다.</li>
      <li>
<strong>vertex</strong>: 엔진을 정점 파이프라인으로 설정합니다.</li>
      <li>
<strong>airflow</strong>: (실험적) 엔진을 Apache Airflow로 설정합니다.</li>
      <li>
<strong>beam</strong>: (실험적) 엔진을 Apache Beam으로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, 기본적으로 로컬 오케스트레이터가 사용됩니다.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>(선택 사항) IAP 보호 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var> </dt>
<dd>(선택 사항) Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>



#### 예:

Kubeflow:

<pre class="devsite-terminal">tfx pipeline delete --engine=kubeflow --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \&lt;br&gt;--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;</pre>

Local:

<pre class="devsite-terminal">tfx pipeline delete --engine=local --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;</pre>

Vertex:

<pre class="devsite-terminal">tfx pipeline delete --engine=vertex --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;
</pre>

### list

지정된 오케스트레이터의 모든 파이프라인을 나열합니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline list [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \&lt;br&gt;--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]</pre>

<dl>
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
  <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
      <li> <strong>local</strong>: 엔진을 로컬 오케스트레이터로 설정합니다.</li>
      <li>
<strong>vertex</strong>: 엔진을 정점 파이프라인으로 설정합니다.</li>
      <li>
<strong>airflow</strong>: (실험적) 엔진을 Apache Airflow로 설정합니다.</li>
      <li>
<strong>beam</strong>: (실험적) 엔진을 Apache Beam으로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, 기본적으로 로컬 오케스트레이터가 사용됩니다.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>(선택 사항) IAP 보호 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var> </dt>
<dd>(선택 사항) Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>



#### 예:

Kubeflow:

<pre class="devsite-terminal">tfx pipeline list --engine=kubeflow --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \&lt;br&gt;--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;</pre>

Local:

<pre class="devsite-terminal">tfx pipeline list --engine=local
</pre>

Vertex:

<pre class="devsite-terminal">tfx pipeline list --engine=vertex
</pre>

## tfx run

`tfx run` 명령 그룹의 명령 구조는 다음과 같습니다.

<pre class="devsite-terminal">tfx run &lt;var&gt;command&lt;/var&gt; &lt;var&gt;required-flags&lt;/var&gt; [&lt;var&gt;optional-flags&lt;/var&gt;]</pre>

다음 섹션을 통해 `tfx run` 명령 그룹의 명령에 대해 자세히 알아보세요.

### create

오케스트레이터에서 파이프라인에 대한 새 실행 인스턴스를 만듭니다. Kubeflow의 경우 클러스터에 있는 파이프라인의 최신 파이프라인 버전이 사용됩니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx run create --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; \&lt;br&gt;--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var> </dt>
  <dd>The name of the pipeline.</dd>
  <dt>--endpoint=<var>endpoint</var> </dt>
  <dd>
    <p>(선택 사항) Kubeflow Pipelines API 서비스의 끝점입니다. Kubeflow Pipelines API 서비스의 끝점은 Kubeflow Pipelines 대시보드의 URL과 동일합니다. 끝점 값은 다음과 같아야 합니다.</p>
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
  <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
      <li> <strong>local</strong>: 엔진을 로컬 오케스트레이터로 설정합니다.</li>
      <li>
<strong>vertex</strong>: 엔진을 정점 파이프라인으로 설정합니다.</li>
      <li>
<strong>airflow</strong>: (실험적) 엔진을 Apache Airflow로 설정합니다.</li>
      <li>
<strong>beam</strong>: (실험적) 엔진을 Apache Beam으로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, 기본적으로 로컬 오케스트레이터가 사용됩니다.</p>
  </dd>


  <dt>--runtime_parameter=<var>parameter-name</var>=<var>parameter-value</var>
</dt>
  <dd>(선택 사항) 런타임 매개변수 값을 설정합니다. 여러 변수의 값을 설정하기 위해 여러 번 설정할 수 있습니다. 'airflow', 'kubeflow', 'vertex' 엔진에만 적용됩니다.</dd>


  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>(선택 사항) IAP 보호 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
  <dd>(선택 사항) Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>


  <dt>--project=<var>GCP-project-id</var>
</dt>
  <dd>(Vertex에 필수) Vertex 파이프라인의 GCP 프로젝트 ID입니다.</dd>


  <dt>--region=<var>GCP-region</var>
</dt>
  <dd>(Vertex에 필수) us-central1과 같은 GCP 영역 이름입니다. 사용 가능한 영역은 [Vertex 문서](https://cloud.google.com/vertex-ai/docs/general/locations)를 참조하세요.</dd>





#### 예:

Kubeflow:

<pre class="devsite-terminal">tfx run create --engine=kubeflow --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \&lt;br&gt;--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;</pre>

Local:

<pre class="devsite-terminal">tfx run create --engine=local --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;
</pre>

Vertex:

<pre class="devsite-terminal">tfx run create --engine=vertex --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
  --runtime_parameter=&lt;var&gt;var_name&lt;/var&gt;=&lt;var&gt;var_value&lt;/var&gt; \
  --project=&lt;var&gt;gcp-project-id&lt;/var&gt; --region=&lt;var&gt;gcp-region&lt;/var&gt;
</pre>

### terminate

주어진 파이프라인의 실행을 중지합니다.

** 중요 참고: 현재, Kubeflow에서만 지원됩니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx run terminate --run_id=&lt;var&gt;run-id&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \&lt;br&gt;--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]</pre>

<dl>
  <dt>--run_id=<var>run-id</var> </dt>
  <dd>Unique identifier for a pipeline run.</dd>
  <dt>--endpoint=<var>endpoint</var> </dt>
  <dd>
    <p>(선택 사항) Kubeflow Pipelines API 서비스의 끝점입니다. Kubeflow Pipelines API 서비스의 끝점은 Kubeflow Pipelines 대시보드의 URL과 동일합니다. 끝점 값은 다음과 같아야 합니다.</p>
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
  <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li> <strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, 기본적으로 로컬 오케스트레이터가 사용됩니다.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>(선택 사항) IAP 보호 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var> </dt>
<dd>(선택 사항) Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>



#### 예:

Kubeflow:

<pre class="devsite-terminal">tfx run delete --engine=kubeflow --run_id=&lt;var&gt;run-id&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \&lt;br&gt;--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;</pre>

### list

파이프라인의 모든 실행을 나열합니다.

** 중요 참고 사항: 현재 로컬 및 Apache Beam에서는 지원되지 않습니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx run list --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; \&lt;br&gt;--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var> </dt>
  <dd>파이프라인의 이름입니다.</dd>
  <dt>--endpoint=<var>endpoint</var> </dt>
  <dd>
    <p>(선택 사항) Kubeflow Pipelines API 서비스의 끝점입니다. Kubeflow Pipelines API 서비스의 끝점은 Kubeflow Pipelines 대시보드의 URL과 동일합니다. 끝점 값은 다음과 같아야 합니다.</p>
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
  <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
      <li>
<strong>airflow</strong>: (실험적) 엔진을 Apache Airflow로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, 기본적으로 로컬 오케스트레이터가 사용됩니다.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>(선택 사항) IAP 보호 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var> </dt>
<dd>(선택 사항) Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>



#### 예:

Kubeflow:

<pre class="devsite-terminal">tfx run list --engine=kubeflow --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \&lt;br&gt;--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;</pre>

### status

실행의 현재 상태를 반환합니다.

** 중요 참고 사항: 현재 로컬 및 Apache Beam에서는 지원되지 않습니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx run status --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; --run_id=&lt;var&gt;run-id&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; \&lt;br&gt;--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var> </dt>
  <dd>파이프라인의 이름입니다.</dd>
  <dt>--run_id=<var>run-id</var> </dt>
  <dd>파이프라인 실행의 고유 식별자입니다.</dd>
  <dt>--endpoint=<var>endpoint</var> </dt>
  <dd>
    <p>(선택 사항) Kubeflow Pipelines API 서비스의 끝점입니다. Kubeflow Pipelines API 서비스의 끝점은 Kubeflow Pipelines 대시보드의 URL과 동일합니다. 끝점 값은 다음과 같아야 합니다.</p>
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
  <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
      <li>
<strong>airflow</strong>: (실험적) 엔진을 Apache Airflow로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, 기본적으로 로컬 오케스트레이터가 사용됩니다.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>(선택 사항) IAP 보호 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var> </dt>
<dd>(선택 사항) Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>



#### 예:

Kubeflow:

<pre class="devsite-terminal">tfx run status --engine=kubeflow --run_id=&lt;var&gt;run-id&lt;/var&gt; --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \&lt;br&gt;--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;</pre>

### delete

주어진 파이프라인의 실행을 삭제합니다.

** 중요 참고: 현재, Kubeflow에서만 지원됩니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx run delete --run_id=&lt;var&gt;run-id&lt;/var&gt; [--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \&lt;br&gt;--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;]</pre>

<dl>
  <dt>--run_id=<var>run-id</var> </dt>
  <dd>파이프라인 실행의 고유 식별자입니다.</dd>
  <dt>--endpoint=<var>endpoint</var> </dt>
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
  <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, 기본적으로 로컬 오케스트레이터가 사용됩니다.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>(선택 사항) IAP 보호 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var> </dt>
<dd>(선택 사항) Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>



#### 예:

Kubeflow:

<pre class="devsite-terminal">tfx run delete --engine=kubeflow --run_id=&lt;var&gt;run-id&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \&lt;br&gt;--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;</pre>

## tfx template [실험적]

`tfx template` 명령 그룹의 명령 구조는 다음과 같습니다.

<pre class="devsite-terminal">tfx template &lt;var&gt;command&lt;/var&gt; &lt;var&gt;required-flags&lt;/var&gt; [&lt;var&gt;optional-flags&lt;/var&gt;]</pre>

다음 섹션을 통해 `tfx template` 명령 그룹의 명령에 대해 자세히 알아보세요. 템플릿은 실험적인 기능이며 언제든지 변경될 수 있습니다.

### list

사용 가능한 TFX 파이프라인 템플릿을 나열합니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx template list</pre>

### copy

템플릿을 대상 디렉토리에 복사합니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx template copy --model=&lt;var&gt;model&lt;/var&gt; --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \&lt;br&gt;--destination_path=&lt;var&gt;destination-path&lt;/var&gt;</pre>

<dl>
  <dt>--model=<var>model</var> </dt>
  <dd>The name of the model built by the pipeline template.</dd>
  <dt>--pipeline_name=<var>pipeline-name</var> </dt>
  <dd>The name of the pipeline.</dd>
  <dt>--destination_path=<var>destination-path</var> </dt>
  <dd>The path to copy the template to.</dd>
</dl>

## TFX CLI 플래그 이해하기

### 공통 플래그

<dl>
  <dt>--engine=<var>engine</var> </dt>
  <dd>
    <p>파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
      <li> <strong>local</strong>: 엔진을 로컬 오케스트레이터로 설정합니다.</li>
      <li>
<strong>vertex</strong>: 엔진을 정점 파이프라인으로 설정합니다.</li>
      <li>
<strong>airflow</strong>: (실험적) 엔진을 Apache Airflow로 설정합니다.</li>
      <li>
<strong>beam</strong>: (실험적) 엔진을 Apache Beam으로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, 기본적으로 로컬 오케스트레이터가 사용됩니다.</p>
  </dd>
</dl>

  <dt>--pipeline_name=<var>pipeline-name</var> </dt>
  <dd>The name of the pipeline.</dd>


  <dt>--pipeline_path=<var>pipeline-path</var> </dt>
  <dd>The path to the pipeline configuration file.</dd>


  <dt>--run_id=<var>run-id</var> </dt>
  <dd>파이프라인 실행의 고유 식별자입니다.</dd>





### Kubeflow 특정 플래그

<dl>
  <dt>--endpoint=<var>endpoint</var> </dt>
  <dd>
    <p>Kubeflow Pipelines API 서비스의 끝점입니다. Kubeflow Pipelines API 서비스의 끝점은 Kubeflow Pipelines 대시보드의 URL과 동일합니다. 끝점 값은 다음과 같아야 합니다.</p>
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
  <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  


  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>IAP 보호 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var> </dt>
<dd>Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>



## TFX CLI로 생성된 파일

파이프라인이 생성되고 실행되면 파이프라인 관리를 위해 여러 파일이 생성됩니다.

- ${HOME}/tfx/local, beam, airflow, vertex
    - 구성에서 읽은 파이프라인 메타데이터는 `${HOME}/tfx/${ORCHESTRATION_ENGINE}/${PIPELINE_NAME}` 아래에 저장됩니다. `AIRFLOW_HOME` 또는 `KUBEFLOW_HOME`과 같은 환경 변수를 설정하여 이 위치를 사용자 정의할 수 있습니다. 이 동작은 향후 릴리스에서 변경될 수 있습니다. 이 디렉터리는 파이프라인을 실행하거나 업데이트하는 데 필요한 Kubeflow Pipelines 클러스터에 파이프라인 ID를 포함한 파이프라인 정보를 저장하는 데 사용됩니다.
    - TFX 0.25 이전에는 이러한 파일이 `${HOME}/${ORCHESTRATION_ENGINE}`에 있었습니다. TFX 0.25에서는 원활한 마이그레이션을 위해 이전 위치의 파일이 자동으로 새 위치로 이동됩니다.
    - TFX 0.27부터 kubeflow는 로컬 파일 시스템에 이러한 메타데이터 파일을 생성하지 않습니다. 그러나 kubeflow가 생성하는 다른 파일은 아래를 참조하세요.
- (Kubeflow만 해당) Dockerfile 및 컨테이너 이미지
    - Kubeflow Pipelines에는 파이프라인에 대해 두 가지 종류의 입력이 필요합니다. 이러한 파일은 현재 디렉터리에서 TFX에 의해 생성됩니다.
    - 하나는 파이프라인에서 구성 요소를 실행하는 데 사용되는 컨테이너 이미지입니다. 이 컨테이너 이미지는 Kubeflow Pipelines용 파이프라인이 생성되거나 `--build-image` 플래그로 업데이트될 때 빌드됩니다. TFX CLI는 있는 경우 `Dockerfile`을 생성하고 컨테이너 이미지를 빌드하고 KubeflowDagRunnerConfig에 지정된 레지스트리에 푸시합니다.
