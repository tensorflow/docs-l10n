# TFX 명령줄 인터페이스 사용하기

TFX 명령줄 인터페이스(CLI)는 Apache Airflow, Apache Beam 및 Kubeflow Pipelines와 같은 파이프라인 오케스트레이터를 사용하여 전 범위의 파이프라인 작업을 수행합니다. 예를 들어, CLI를 사용하여 다음을 수행할 수 있습니다.

- 파이프라인을 생성, 업데이트 및 삭제합니다.
- 파이프라인을 실행하고 다양한 오케스트레이터에서 실행을 모니터링합니다.
- 파이프라인 및 파이프라인 실행을 나열합니다.

참고: TFX CLI는 현재 호환성 보장을 제공하지 않습니다. 새 버전이 출시되면 CLI 인터페이스가 변경될 수 있습니다.

## About the TFX CLI

The TFX CLI is installed as a part of the TFX package. All CLI commands follow the structure below:

<pre class="devsite-terminal">tfx <var>command-group</var> <var>command</var> <var>flags</var>
</pre>

The following <var>command-group</var> options are currently supported:

- [tfx pipeline](#tfx-pipeline) - TFX 파이프라인을 만들고 관리합니다.
- [tfx run](#tfx-run) - 다양한 조정 플랫폼에서 TFX 파이프라인 실행을 생성하고 관리합니다.
- [tfx template](#tfx-template-experimental) - TFX 파이프라인 템플릿을 나열하고 복사하기 위한 실험 단계에 있는 명령입니다.

각 명령 그룹은 <var>commands</var> 세트를 제공합니다. [파이프라인 명령](#tfx-pipeline), [실행 명령](#tfx-run) 및 [템플릿 명령](#tfx-template-experimental) 섹션의 지침에 따라 이들 명령을 사용하는 방법에 대해 자세히 알아보세요.

경고: 현재, 모든 오케스트레이터에서 모든 명령이 지원되는 것은 아닙니다. 이들 명령은 지원되는 엔진을 명시적으로 언급합니다.

플래그를 사용하면 CLI 명령에 인수를 전달할 수 있습니다. 플래그의 단어는 하이픈(`-`) 또는 밑줄(`_`)로 구분됩니다. 예를 들어, 파이프라인 이름 플래그는 `--pipeline-name` 또는 `--pipeline_name`으로 지정할 수 있습니다. 이 문서에서는 간단히 하기 위해 밑줄로 플래그를 지정합니다. [TFX CLI에서 사용되는 <var>플래그</var>](#understanding-tfx-cli-flags)에 대해 자세히 알아보세요.

## tfx pipeline

The structure for commands in the `tfx pipeline` command group is as follows:

<pre class="devsite-terminal">tfx pipeline <var>command</var> <var>required-flags</var> [<var>optional-flags</var>]
</pre>

다음 섹션을 사용하여 `tfx pipeline` 명령 그룹의 명령에 대해 자세히 알아보세요.

### create

지정된 오케스트레이터에서 새 파이프라인을 만듭니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline create --pipeline_path=<var>pipeline-path</var> [--endpoint=<var>endpoint</var> --engine=<var>engine</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var> --package_path=<var>package-path</var> \
--build_target_image=<var>build-target-image</var> --build_base_image=<var>build-base-image</var> \
--skaffold_cmd=<var>skaffold-command</var>]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>파이프라인 구성 파일의 경로입니다.</dd>
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

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li> <strong>airflow</strong>: 엔진을 Apache Airflow로 설정합니다.</li>
      <li>
<strong>beam</strong>: 엔진을 Apache Beam으로 설정합니다.</li>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, Apache Beam이 기본적으로 사용됩니다.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>(선택 사항) IAP 보호 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>(선택 사항) Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>


  <dt>--package_path=<var>package-path</var>
</dt>
  <dd>
    <p>(선택 사항) 파일로 컴파일된 파이프라인의 경로입니다. 컴파일된 파이프라인은 압축 파일(<code>.tar.gz</code>, <code>.tgz</code> 또는 <code>.zip</code>) 또는 YAML 파일(<code>.yaml</code> 또는 <code>.yml</code>)이어야 합니다.</p>
    <p><var>package-path</var>가 지정되지 않으면 TFX가 <code>current_directory/pipeline_name.tar.gz</code>를 기본 경로로 사용합니다.</p>
  </dd>
  <dt>--build_target_image=<var>build-target-image</var>
</dt>
  <dd>
    <p>(선택 사항) <var>engine</var>이 <strong>kubeflow</strong>이면 TFX는 파이프라인에 대한 컨테이너 이미지를 생성합니다. 빌드 대상 이미지는 파이프라인 컨테이너 이미지를 만들 때 사용할 이름, 컨테이너 이미지 리포지토리 및 태그를 지정합니다. 태그를 지정하지 않으면 컨테이너 이미지가 <code>latest</code>로 태그 지정됩니다.</p>
    <p>Kubeflow Pipelines 클러스터가 파이프라인을 실행하려면 클러스터가 지정된 컨테이너 이미지 리포지토리에 액세스할 수 있어야 합니다.</p>
  </dd>
  <dt>--build_base_image=<var>build-base-image</var>
</dt>
  <dd>
    <p>(선택 사항) <var>engine</var>이 <strong>kubeflow</strong>이면 TFX는 파이프라인에 대한 컨테이너 이미지를 생성합니다. 빌드 기본 이미지는 파이프라인 컨테이너 이미지를 빌드할 때 사용할 기본 컨테이너 이미지를 지정합니다.</p>
  </dd>
  <dt>--skaffold_cmd=<var>skaffold-cmd</var>
</dt>
  <dd>
    <p>       (Optional.) The path to <a href="https://skaffold.dev/" class="external">       Skaffold</a> on your computer.     </p>
  </dd>



#### 예:

Apache Airflow:

<pre class="devsite-terminal">tfx pipeline create --engine=airflow --pipeline_path=<var>pipeline-path</var>
</pre>

Apache Beam:

<pre class="devsite-terminal">tfx pipeline create --engine=beam --pipeline_path=<var>pipeline-path</var>
</pre>

Kubeflow:

<pre class="devsite-terminal">tfx pipeline create --engine=kubeflow --pipeline_path=<var>pipeline-path</var> --package_path=<var>package-path</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var> --endpoint=<var>endpoint</var> \
--skaffold_cmd=<var>skaffold-cmd</var>
</pre>

사용자 환경에서 엔진을 자동 감지하려면 아래 예와 같이 engine 플래그를 사용하지 않으면 됩니다. 자세한 내용은 플래그 섹션을 확인하세요.

<pre class="devsite-terminal">tfx pipeline create --pipeline_path=<var>pipeline-path</var> --endpoint --iap_client_id --namespace \
--package_path --skaffold_cmd
</pre>

### update

지정된 오케스트레이터에서 기존 파이프라인을 업데이트합니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline update --pipeline_path=<var>pipeline-path</var> [--endpoint=<var>endpoint</var> --engine=<var>engine</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var> --package_path=<var>package-path</var> \
--skaffold_cmd=<var>skaffold-command</var>]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>파이프라인 구성 파일의 경로입니다.</dd>
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

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li> <strong>airflow</strong>: 엔진을 Apache Airflow로 설정합니다.</li>
      <li>
<strong>beam</strong>: 엔진을 Apache Beam으로 설정합니다.</li>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, Apache Beam이 기본적으로 사용됩니다.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>(선택 사항) IAP 보호 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>(선택 사항) Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>


  <dt>--package_path=<var>package-path</var>
</dt>
  <dd>
    <p>(선택 사항) 파일로 컴파일된 파이프라인의 경로입니다. 컴파일된 파이프라인은 압축 파일(<code>.tar.gz</code>, <code>.tgz</code> 또는 <code>.zip</code>) 또는 YAML 파일(<code>.yaml</code> 또는 <code>.yml</code>)이어야 합니다.</p>
    <p><var>package-path</var>가 지정되지 않으면 TFX가 <code>current_directory/pipeline_name.tar.gz</code>를 기본 경로로 사용합니다.</p>
  </dd>
  <dt>--skaffold_cmd=<var>skaffold-cmd</var>
</dt>
  <dd>
    <p>       (Optional.) The path to <a href="https://skaffold.dev/" class="external">       Skaffold</a> on your computer.     </p>
  </dd>



#### 예:

Apache Airflow:

<pre class="devsite-terminal">tfx pipeline update --engine=airflow --pipeline_path=<var>pipeline-path</var>
</pre>

Apache Beam:

<pre class="devsite-terminal">tfx pipeline update --engine=beam --pipeline_path=<var>pipeline-path</var>
</pre>

Kubeflow:

<pre class="devsite-terminal">tfx pipeline update --engine=kubeflow --pipeline_path=<var>pipeline-path</var> --package_path=<var>package-path</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var> --endpoint=<var>endpoint</var> \
--skaffold_cmd=<var>skaffold-cmd</var>
</pre>

### compile

파이프라인 구성 파일을 컴파일하여 Kubeflow에서 워크플로 파일을 만들고 컴파일하는 동안 다음 검사를 수행합니다.

1. 파이프라인 경로가 유효한지 확인합니다.
2. 파이프라인 구성 파일에서 파이프라인 세부 정보가 성공적으로 추출되었는지 확인합니다.
3. 파이프라인 구성의 DagRunner가 엔진과 일치하는지 확인합니다.
4. 제공된 패키지 경로에 워크플로 파일이 성공적으로 생성되었는지 확인합니다(Kubflow에만 해당).

파이프라인을 생성하거나 업데이트하기 전에 사용하는 것이 좋습니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline compile --pipeline_path=<var>pipeline-path</var> [--endpoint=<var>endpoint</var> --engine=<var>engine</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var> --package_path=<var>package-path</var>]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>파이프라인 구성 파일의 경로입니다.</dd>
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

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li> <strong>airflow</strong>: 엔진을 Apache Airflow로 설정합니다.</li>
      <li>
<strong>beam</strong>: 엔진을 Apache Beam으로 설정합니다.</li>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, Apache Beam이 기본적으로 사용됩니다.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>(선택 사항) IAP 보호 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>(선택 사항) Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>


  <dt>--package_path=<var>package-path</var>
</dt>
  <dd>
    <p>(선택 사항) 파일로 컴파일된 파이프라인의 경로입니다. 컴파일된 파이프라인은 압축 파일(<code>.tar.gz</code>, <code>.tgz</code> 또는 <code>.zip</code>) 또는 YAML 파일(<code>.yaml</code> 또는 <code>.yml</code>)이어야 합니다.</p>
    <p><var>package-path</var>가 지정되지 않으면 TFX가 <code>current_directory/pipeline_name.tar.gz</code>를 기본 경로로 사용합니다.</p>
  </dd>



#### 예:

Apache Airflow:

<pre class="devsite-terminal">tfx pipeline compile --engine=airflow --pipeline_path=<var>pipeline-path</var>
</pre>

Apache Beam:

<pre class="devsite-terminal">tfx pipeline compile --engine=beam --pipeline_path=<var>pipeline-path</var>
</pre>

Kubeflow:

<pre class="devsite-terminal">tfx pipeline compile --engine=kubeflow --pipeline_path=<var>pipeline-path</var> --package_path=<var>package-path</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var> --endpoint=<var>endpoint</var>
</pre>

### delete

지정된 오케스트레이터에서 파이프라인을 삭제합니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline delete --pipeline_path=<var>pipeline-path</var> [--endpoint=<var>endpoint</var> --engine=<var>engine</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var>]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>파이프라인 구성 파일의 경로입니다.</dd>
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

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li> <strong>airflow</strong>: 엔진을 Apache Airflow로 설정합니다.</li>
      <li>
<strong>beam</strong>: 엔진을 Apache Beam으로 설정합니다.</li>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, Apache Beam이 기본적으로 사용됩니다.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>(선택 사항) IAP 보호 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>(선택 사항) Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>



#### 예:

Apache Airflow:

<pre class="devsite-terminal">tfx pipeline delete --engine=airflow --pipeline_name=<var>pipeline-name</var>
</pre>

Apache Beam:

<pre class="devsite-terminal">tfx pipeline delete --engine=beam --pipeline_name=<var>pipeline-name</var>
</pre>

Kubeflow:

<pre class="devsite-terminal">tfx pipeline delete --engine=kubeflow --pipeline_name=<var>pipeline-name</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var> --endpoint=<var>endpoint</var>
</pre>

### list

지정된 오케스트레이터의 모든 파이프라인을 나열합니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline list [--endpoint=<var>endpoint</var> --engine=<var>engine</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var>]
</pre>

<dl>
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

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li> <strong>airflow</strong>: 엔진을 Apache Airflow로 설정합니다.</li>
      <li>
<strong>beam</strong>: 엔진을 Apache Beam으로 설정합니다.</li>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, Apache Beam이 기본적으로 사용됩니다.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>(선택 사항) IAP 보호 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>(선택 사항) Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>



#### 예:

Apache Airflow:

<pre class="devsite-terminal">tfx pipeline list --engine=airflow
</pre>

Apache Beam:

<pre class="devsite-terminal">tfx pipeline list --engine=beam
</pre>

Kubeflow:

<pre class="devsite-terminal">tfx pipeline list --engine=kubeflow --iap_client_id=<var>iap-client-id</var> \
--namespace=<var>namespace</var> --endpoint=<var>endpoint</var>
</pre>

## tfx run

The structure for commands in the `tfx run` command group is as follows:

<pre class="devsite-terminal">tfx run <var>command</var> <var>required-flags</var> [<var>optional-flags</var>]
</pre>

다음 섹션을 통해 `tfx run` 명령 그룹의 명령에 대해 자세히 알아보세요.

### create

오케스트레이터에서 파이프라인에 대한 새 실행 인스턴스를 만듭니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx run create --pipeline_name=<var>pipeline-name</var> [--endpoint=<var>endpoint</var> \
--engine=<var>engine</var> --iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var>]
</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>파이프라인의 이름입니다.</dd>
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

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li> <strong>airflow</strong>: 엔진을 Apache Airflow로 설정합니다.</li>
      <li>
<strong>beam</strong>: 엔진을 Apache Beam으로 설정합니다.</li>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, Apache Beam이 기본적으로 사용됩니다.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>(선택 사항) IAP 보호 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>(선택 사항) Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>



#### 예:

Apache Airflow:

<pre class="devsite-terminal">tfx run create --engine=airflow --pipeline_name=<var>pipeline-name</var>
</pre>

Apache Beam:

<pre class="devsite-terminal">tfx run create --engine=beam --pipeline_name=<var>pipeline-name</var>
</pre>

Kubeflow:

<pre class="devsite-terminal">tfx run create --engine=kubeflow --pipeline_name=<var>pipeline-name</var> --iap_client_id=<var>iap-client-id</var> \
--namespace=<var>namespace</var> --endpoint=<var>endpoint</var>
</pre>

### terminate

주어진 파이프라인의 실행을 중지합니다.

** 중요 참고: 현재, Kubeflow에서만 지원됩니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx run terminate --run_id=<var>run-id</var> [--endpoint=<var>endpoint</var> --engine=<var>engine</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var>]
</pre>

<dl>
  <dt>--run_id=<var>run-id</var>
</dt>
  <dd>파이프라인 실행의 고유 식별자입니다.</dd>
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

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li> <strong>airflow</strong>: 엔진을 Apache Airflow로 설정합니다.</li>
      <li>
<strong>beam</strong>: 엔진을 Apache Beam으로 설정합니다.</li>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, Apache Beam이 기본적으로 사용됩니다.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>(선택 사항) IAP 보호 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>(선택 사항) Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>



#### 예:

Kubeflow:

<pre class="devsite-terminal">tfx run delete --engine=kubeflow --run_id=<var>run-id</var> --iap_client_id=<var>iap-client-id</var> \
--namespace=<var>namespace</var> --endpoint=<var>endpoint</var>
</pre>

### list

파이프라인의 모든 실행을 나열합니다.

** 중요 참고: 현재, Apache Beam에서는 지원되지 않습니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx run list --pipeline_name=<var>pipeline-name</var> [--endpoint=<var>endpoint</var> \
--engine=<var>engine</var> --iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var>]
</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>파이프라인의 이름입니다.</dd>
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

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li> <strong>airflow</strong>: 엔진을 Apache Airflow로 설정합니다.</li>
      <li>
<strong>beam</strong>: 엔진을 Apache Beam으로 설정합니다.</li>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, Apache Beam이 기본적으로 사용됩니다.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>(선택 사항) IAP 보호 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>(선택 사항) Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>



#### 예:

Apache Airflow:

<pre class="devsite-terminal">tfx run list --engine=airflow --pipeline_name=<var>pipeline-name</var>
</pre>

Kubeflow:

<pre class="devsite-terminal">tfx run list --engine=kubeflow --pipeline_name=<var>pipeline-name</var> --iap_client_id=<var>iap-client-id</var> \
--namespace=<var>namespace</var> --endpoint=<var>endpoint</var>
</pre>

### status

Returns the current status of a run.

** 중요 참고: 현재, Apache Beam에서는 지원되지 않습니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx run status --pipeline_name=<var>pipeline-name</var> --run_id=<var>run-id</var> [--endpoint=<var>endpoint</var> \
--engine=<var>engine</var> --iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var>]
</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>파이프라인의 이름입니다.</dd>
  <dt>--run_id=<var>run-id</var>
</dt>
  <dd>파이프라인 실행의 고유 식별자입니다.</dd>
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

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li> <strong>airflow</strong>: 엔진을 Apache Airflow로 설정합니다.</li>
      <li>
<strong>beam</strong>: 엔진을 Apache Beam으로 설정합니다.</li>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, Apache Beam이 기본적으로 사용됩니다.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>(선택 사항) IAP 보호 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>(선택 사항) Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>



#### 예:

Apache Airflow:

<pre class="devsite-terminal">tfx run status --engine=airflow --run_id=<var>run-id</var> --pipeline_name=<var>pipeline-name</var>
</pre>

Kubeflow:

<pre class="devsite-terminal">tfx run status --engine=kubeflow --run_id=<var>run-id</var> --pipeline_name=<var>pipeline-name</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var> --endpoint=<var>endpoint</var>
</pre>

### delete

주어진 파이프라인의 실행을 삭제합니다.

** 중요 참고: 현재, Kubeflow에서만 지원됩니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx run delete --run_id=<var>run-id</var> [--engine=<var>engine</var> --iap_client_id=<var>iap-client-id</var> \
--namespace=<var>namespace</var> --endpoint=<var>endpoint</var>]
</pre>

<dl>
  <dt>--run_id=<var>run-id</var>
</dt>
  <dd>파이프라인 실행의 고유 식별자입니다.</dd>
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

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>(선택 사항) 파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li> <strong>airflow</strong>: 엔진을 Apache Airflow로 설정합니다.</li>
      <li>
<strong>beam</strong>: 엔진을 Apache Beam으로 설정합니다.</li>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, Apache Beam이 기본적으로 사용됩니다.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>(선택 사항) IAP 보호 끝점의 클라이언트 ID입니다.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>(선택 사항) Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>



#### 예:

Kubeflow:

<pre class="devsite-terminal">tfx run delete --engine=kubeflow --run_id=<var>run-id</var> --iap_client_id=<var>iap-client-id</var> \
--namespace=<var>namespace</var> --endpoint=<var>endpoint</var>
</pre>

## tfx template [실험적]

The structure for commands in the `tfx template` command group is as follows:

<pre class="devsite-terminal">tfx template <var>command</var> <var>required-flags</var> [<var>optional-flags</var>]
</pre>

다음 섹션을 통해 `tfx template` 명령 그룹의 명령에 대해 자세히 알아보세요. 템플릿은 실험적인 기능이며 언제든지 변경될 수 있습니다.

### list

사용 가능한 TFX 파이프라인 템플릿을 나열합니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx template list
</pre>

### copy

템플릿을 대상 디렉토리에 복사합니다.

사용법:

<pre class="devsite-click-to-copy devsite-terminal">tfx template copy --model=<var>model</var> --pipeline_name=<var>pipeline-name</var> \
--destination_path=<var>destination-path</var>
</pre>

<dl>
  <dt>--model=<var>model</var>
</dt>
  <dd>파이프라인 템플릿으로 빌드된 모델의 이름입니다.</dd>
  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>파이프라인의 이름입니다.</dd>
  <dt>--destination_path=<var>destination-path</var>
</dt>
  <dd>템플릿을 복사할 경로입니다.</dd>
</dl>

## TFX CLI 플래그 이해하기

### Common flags

<dl>
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>파이프라인에 사용할 오케스트레이터입니다. 엔진 값은 다음 값 중 하나와 일치해야 합니다.</p>
    <ul>
      <li> <strong>airflow</strong>: 엔진을 Apache Airflow로 설정합니다.</li>
      <li>
<strong>beam</strong>: 엔진을 Apache Beam으로 설정합니다.</li>
      <li>
<strong>kubeflow</strong>: 엔진을 Kubeflow로 설정합니다.</li>
    </ul>
    <p>엔진이 설정되지 않으면, 환경에 따라 엔진이 자동 감지됩니다.</p>
    <p>** 중요 참고 사항: 파이프라인 구성 파일에서 DagRunner에 필요한 오케스트레이터는 선택되거나 자동 감지된 엔진과 일치해야 합니다. 엔진 자동 감지는 사용자 환경을 기반으로 합니다. Apache Airflow 및 Kubeflow Pipelines가 설치되지 않은 경우, Apache Beam이 기본적으로 사용됩니다.</p>
  </dd>
</dl>

  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>파이프라인의 이름입니다.</dd>


  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>파이프라인 구성 파일의 경로입니다.</dd>


  <dt>--run_id=<var>run-id</var>
</dt>
  <dd>파이프라인 실행의 고유 식별자입니다.</dd>





### Kubeflow specific flags

<dl>
  <dt>--endpoint=<var>endpoint</var>
</dt>
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


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>Kubeflow Pipelines API에 연결하기 위한 Kubernetes 네임스페이스입니다. 네임스페이스가 지정되지 않으면, <code>kubeflow</code>가 기본값으로 사용됩니다.</dd>


  <dt>--package_path=<var>package-path</var>
</dt>
  <dd>
    <p>파일로 컴파일된 파이프라인의 경로입니다. 컴파일된 파이프라인은 압축 파일(<code>.tar.gz</code>, <code>.tgz</code> 또는 <code>.zip</code>) 또는 YAML 파일(<code>.yaml</code> 또는 <code>.yml</code>)이어야 합니다.</p>
    <p>       <var>package-path</var>가 지정되지 않으면 TFX가 <code><var>current_directory</var>/<var>pipeline_name</var>.tar.gz</code>를 기본 경로로 사용합니다.</p>
  </dd>




