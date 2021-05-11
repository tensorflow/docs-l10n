# TFX コマンドラインインターフェイスの使用

TFX コマンドラインインターフェイス（CLI）は、Apache Airflow、Apache Beam、Kubeflow Pipelines などのパイプラインオーケストレーターを使用してあらゆるパイプラインアクションを実行します。また、ローカルオーケストレーターを使用して、開発やデバッグを高速化することもできます。CLI を使用すると次のことができます。

- パイプラインを作成、更新、削除する。
- パイプラインを実行し、さまざまなオーケストレーターでの実行を監視する。
- パイプラインとパイプラインの実行を一覧表示する。

注: TFX CLI は現在、互換性を保証していません。新しいバージョンがリリースされると、CLI インターフェイスが変更される可能性があります。

## TFX CLI の概要

TFX CLI は、TFX パッケージの一部としてインストールされます。すべての CLI コマンドは、以下の構造に従います。

<pre class="devsite-terminal">tfx &lt;var&gt;command-group&lt;/var&gt; &lt;var&gt;command&lt;/var&gt; &lt;var&gt;flags&lt;/var&gt;
</pre>

現在、次の <var>command-group</var> オプションがサポートされています。

- [tfx pipeline](#tfx-pipeline) - TFX パイプラインを作成および管理します。
- [tfx run](#tfx-run) - さまざまなオーケストレーションプラットフォームで TFX パイプラインの実行を作成および管理します。
- [tfx template](#tfx-template-experimental) - TFX パイプラインテンプレートを一覧表示およびコピーするための実験的なコマンド。

各コマンドグループは、一連の<var>コマンド</var>を提供します。これらのコマンドの使用方法の詳細については、[パイプラインコマンド](#tfx-pipeline)、[実行コマンド](#tfx-run)、および[テンプレートコマンド](#tfx-template-experimental)のセクションを参照してください。

警告：現在、すべてのコマンドはすべてのオーケストレーターでサポートされていません。すべてのオーケストレーターでサポートされていないコマンドでは、サポートされているエンジンについて記載されています。

フラグを使用すると、CLI コマンドに引数を渡すことができます。フラグ内の単語は、ハイフン（`-`）またはアンダースコア（` _ `）で区切られます。例えば、パイプライン名フラグは、`--pipeline-name`または`-pipeline_name`のいずれかとして指定できます。このドキュメントでは、簡潔にするためにアンダースコア付きのフラグを指定しています。[TFX CLI で使用される<var>フラグ</var>の詳細はこちらを参照してください](#understanding-tfx-cli-flags)。

## tfx pipeline

`tfx pipeline`コマンドグループのコマンドの構造は次のとおりです。

<pre class="devsite-terminal">tfx pipeline &lt;var&gt;command&lt;/var&gt; &lt;var&gt;required-flags&lt;/var&gt; [&lt;var&gt;optional-flags&lt;/var&gt;]
</pre>

`tfx pipeline`コマンドグループのコマンドの詳細については次のセクションを参照してください。

### create

指定したオーケストレーターに新しいパイプラインを作成します。

Usage:

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline create --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --package_path=&lt;var&gt;package-path&lt;/var&gt; \
--build_target_image=&lt;var&gt;build-target-image&lt;/var&gt; --build_base_image=&lt;var&gt;build-base-image&lt;/var&gt; \
--skaffold_cmd=&lt;var&gt;skaffold-command&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>パイプライン構成ファイルへのパス。</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>（オプション）Kubeflow Pipelines API サービスのエンドポイント。Kubeflow Pipelines API サービスのエンドポイントは、Kubeflow Pipelines ダッシュボードの URL と同じです。エンドポイント値は次のようになります。</p>
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
    <p>（オプション）パイプラインに使用されるオーケストレーター。エンジンの値は、次のいずれかの値と一致する必要があります。</p>
    <ul>
      <li>
<strong>airflow</strong>: エンジンを Apache Airflow に設定します</li>
      <li>
<strong>beam</strong>: エンジンを Apache Beam に設定します</li>
      <li>
<strong>kubeflow</strong>: エンジンを Kubeflow に設定します</li>
      <li>
<strong>local</strong>: エンジンをローカルオーケストレーターに設定します</li>
    </ul>
    <p>エンジンが設定されていない場合、エンジンは環境に基づいて自動検出されます。</p>
    <p>      **要注意：パイプライン構成ファイルの DagRunner に必要とされるオーケストレーターは、選択されたエンジンまたは自動検出されたエンジンと一致する必要があります。エンジンの自動検出は、ユーザー環境に基づいています。Apache Airflow と Kubeflow Pipelines がインストールされていない場合、デフォルトでローカルオーケストレーターが使用されます。</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>    （オプション）IAP で保護されたエンドポイントのクライアント ID。</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>（オプション）Kubeflow Pipelines API に接続するための Kubernetes 名前空間。名前空間が指定されていない場合、値はデフォルトで<code>kubeflow</code>になります。</dd>


  <dt>--package_path=<var>package-path</var>
</dt>
  <dd>
    <p>（オプション）コンパイルされたパイプラインへのファイルとしてのパス。コンパイルされたパイプラインは、圧縮ファイル(<code>.tar.gz</code>、<code>.tgz</code>、または、<code>.zip</code>) または YAML ファイル(<code>.yaml</code> または<code>.yml</code>)である必要があります。</p>
    <p><var>package-path</var> が指定されていない場合、TFX はデフォルトパスとして次を使用します。<code>&lt;var&gt;current_directory&lt;/var&gt;/&lt;var&gt;pipeline_name&lt;/var&gt;.tar.gz</code></p>
  </dd>
  <dt>--build_target_image=<var>build-target-image</var>
</dt>
  <dd>
    <p>      （オプション）<var>エンジン</var> が <strong>kubeflow</strong> の場合、TFX はパイプラインのコンテナイメージを作成します。ビルドターゲットイメージは、パイプラインコンテナイメージを作成するときに使用する名前、コンテナイメージリポジトリ、およびタグを指定します。タグを指定しない場合、コンテナイメージは<code>latest</code>としてタグ付けされます。</p>
    <p>Kubeflow Pipelines クラスタでパイプラインを実行するには、クラスタが指定されたコンテナイメージリポジトリにアクセスできる必要があります。</p>
  </dd>
  <dt>--build_base_image=<var>build-base-image</var>
</dt>
  <dd>
    <p>（オプション）<var>エンジン</var> が <strong>kubeflow</strong> の場合、TFX はパイプラインのコンテナイメージを作成します。ビルドベースイメージは、パイプラインコンテナイメージをビルドするときに使用するベースコンテナイメージを指定します。</p>
  </dd>
  <dt>--skaffold_cmd=<var>skaffold-cmd</var>
</dt>
  <dd>
    <p>（オプション）コンピュータ上の <a href="https://skaffold.dev/" class="external">Skaffold</a> へのパス。</p>
  </dd>



#### 例:

Apache Airflow:

<pre class="devsite-terminal">tfx pipeline create --engine=airflow --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

Kubeflow:

<pre class="devsite-terminal">tfx pipeline create --engine=kubeflow --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; --package_path=&lt;var&gt;package-path&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--skaffold_cmd=&lt;var&gt;skaffold-cmd&lt;/var&gt;
</pre>

ローカル:

<pre class="devsite-terminal">tfx pipeline create --engine=local --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

ユーザー環境からエンジンを自動検出するには、以下の例のようなエンジンフラグの使用を避けてください。詳細については、フラグのセクションを参照してください。

<pre class="devsite-terminal">tfx pipeline create --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; --endpoint --iap_client_id --namespace \
--package_path --skaffold_cmd
</pre>

### update

指定されたオーケストレーターの既存のパイプラインを更新します。

Usage:

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline update --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --package_path=&lt;var&gt;package-path&lt;/var&gt; \
--skaffold_cmd=&lt;var&gt;skaffold-command&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>パイプライン構成ファイルへのパス。</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>（オプション）Kubeflow Pipelines API サービスのエンドポイント。Kubeflow Pipelines API サービスのエンドポイントは、Kubeflow Pipelines ダッシュボードの URL と同じです。エンドポイント値は次のようになります。</p>
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
    <p>（オプション）パイプラインに使用されるオーケストレーター。エンジンの値は、次のいずれかの値と一致する必要があります。</p>
    <ul>
      <li>
<strong>airflow</strong>: エンジンを Apache Airflow に設定します</li>
      <li>
<strong>beam</strong>: エンジンを Apache Beam に設定します</li>
      <li>
<strong>kubeflow</strong>: エンジンを Kubeflow に設定します</li>
      <li>
<strong>local</strong>: エンジンをローカルオーケストレーターに設定します</li>
    </ul>
    <p>エンジンが設定されていない場合、エンジンは環境に基づいて自動検出されます。</p>
    <p>      **要注意：パイプライン構成ファイルの DagRunner に必要とされるオーケストレーターは、選択されたエンジンまたは自動検出されたエンジンと一致する必要があります。エンジンの自動検出は、ユーザー環境に基づいています。Apache Airflow と Kubeflow Pipelines がインストールされていない場合、デフォルトでローカルオーケストレーターが使用されます。</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>    （オプション）IAP で保護されたエンドポイントのクライアント ID。</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>（オプション）Kubeflow Pipelines API に接続するための Kubernetes 名前空間。名前空間が指定されていない場合、値はデフォルトで<code>kubeflow</code>になります。</dd>


  <dt>--package_path=<var>package-path</var>
</dt>
  <dd>
    <p>（オプション）コンパイルされたパイプラインへのファイルとしてのパス。コンパイルされたパイプラインは、圧縮ファイル(<code>.tar.gz</code>、<code>.tgz</code>、または、<code>.zip</code>) または YAML ファイル(<code>.yaml</code> または<code>.yml</code>)である必要があります。</p>
    <p><var>package-path</var> が指定されていない場合、TFX はデフォルトパスとして次を使用します。<code>&lt;var&gt;current_directory&lt;/var&gt;/&lt;var&gt;pipeline_name&lt;/var&gt;.tar.gz</code></p>
  </dd>
  <dt>--skaffold_cmd=<var>skaffold-cmd</var>
</dt>
  <dd>
    <p>（オプション）コンピュータ上の <a href="https://skaffold.dev/" class="external">Skaffold</a> へのパス。</p>
  </dd>



#### 例:

Apache Airflow:

<pre class="devsite-terminal">tfx pipeline update --engine=airflow --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

Kubeflow:

<pre class="devsite-terminal">tfx pipeline update --engine=kubeflow --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; --package_path=&lt;var&gt;package-path&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--skaffold_cmd=&lt;var&gt;skaffold-cmd&lt;/var&gt;
</pre>

ローカル:

<pre class="devsite-terminal">tfx pipeline update --engine=local --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

### compile

パイプライン構成ファイルをコンパイルして Kubeflow にワークフローファイルを作成し、コンパイル時に次のチェックを実行します。

1. パイプラインパスが有効かどうかを確認します。
2. パイプラインの詳細がパイプライン構成ファイルから正常に抽出されているかどうかを確認します。
3. パイプライン構成の DagRunner がエンジンと一致するかどうかを確認します。
4. 提供されたパッケージパスにワークフローファイルが正常に作成されているかどうかを確認します（Kubeflow の場合のみ）。

パイプラインを作成または更新する前に使用することが推薦されます。

Usage:

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline compile --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--engine=&lt;var&gt;engine&lt;/var&gt; \
--package_path=&lt;var&gt;package-path&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>パイプライン構成ファイルへのパス。</dd>
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>（オプション）パイプラインに使用されるオーケストレーター。エンジンの値は、次のいずれかの値と一致する必要があります。</p>
    <ul>
      <li>
<strong>airflow</strong>: エンジンを Apache Airflow に設定します</li>
      <li>
<strong>beam</strong>: エンジンを Apache Beam に設定します</li>
      <li>
<strong>kubeflow</strong>: エンジンを Kubeflow に設定します</li>
      <li>
<strong>local</strong>: エンジンをローカルオーケストレーターに設定します</li>
    </ul>
    <p>エンジンが設定されていない場合、エンジンは環境に基づいて自動検出されます。</p>
    <p>      **要注意：パイプライン構成ファイルの DagRunner に必要とされるオーケストレーターは、選択されたエンジンまたは自動検出されたエンジンと一致する必要があります。エンジンの自動検出は、ユーザー環境に基づいています。Apache Airflow と Kubeflow Pipelines がインストールされていない場合、デフォルトでローカルオーケストレーターが使用されます。</p>
  </dd>
  <dt>--package_path=<var>package-path</var>
</dt>
  <dd>
    <p>（オプション）コンパイルされたパイプラインへのファイルとしてのパス。コンパイルされたパイプラインは、圧縮ファイル(<code>.tar.gz</code>、<code>.tgz</code>、または、<code>.zip</code>) または YAML ファイル(<code>.yaml</code>または<code>.yml</code>)である必要があります。</p>
    <p><var>package-path</var> が指定されていない場合、TFX はデフォルトパスとして次を使用します。<code>&lt;var&gt;current_directory&lt;/var&gt;/&lt;var&gt;pipeline_name&lt;/var&gt;.tar.gz</code></p>
  </dd>
</dl>

#### 例:

Apache Airflow

<pre class="devsite-terminal">tfx pipeline compile --engine=airflow --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

Kubeflow:

<pre class="devsite-terminal">tfx pipeline compile --engine=kubeflow --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; --package_path=&lt;var&gt;package-path&lt;/var&gt;
</pre>

ローカル:

<pre class="devsite-terminal">tfx pipeline compile --engine=local --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

### delete

指定されたオーケストレーターからパイプラインを削除します。

Usage:

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline delete --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>パイプライン構成ファイルへのパス。</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>（オプション）Kubeflow Pipelines API サービスのエンドポイント。Kubeflow Pipelines API サービスのエンドポイントは、Kubeflow Pipelines ダッシュボードの URL と同じです。エンドポイント値は次のようになります。</p>
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
    <p>（オプション）パイプラインに使用されるオーケストレーター。エンジンの値は、次のいずれかの値と一致する必要があります。</p>
    <ul>
      <li>
<strong>airflow</strong>: エンジンを Apache Airflow に設定します</li>
      <li>
<strong>beam</strong>: エンジンを Apache Beam に設定します</li>
      <li>
<strong>kubeflow</strong>: エンジンを Kubeflow に設定します</li>
      <li>
<strong>local</strong>: エンジンをローカルオーケストレーターに設定します</li>
    </ul>
    <p>エンジンが設定されていない場合、エンジンは環境に基づいて自動検出されます。</p>
    <p>      **要注意：パイプライン構成ファイルの DagRunner に必要とされるオーケストレーターは、選択されたエンジンまたは自動検出されたエンジンと一致する必要があります。エンジンの自動検出は、ユーザー環境に基づいています。Apache Airflow と Kubeflow Pipelines がインストールされていない場合、デフォルトでローカルオーケストレーターが使用されます。</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>    （オプション）IAP で保護されたエンドポイントのクライアント ID。</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>（オプション）Kubeflow Pipelines API に接続するための Kubernetes 名前空間。名前空間が指定されていない場合、値はデフォルトで<code>kubeflow</code>になります。</dd>



#### 例:

Apache Airflow:

<pre class="devsite-terminal">tfx pipeline delete --engine=airflow --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;
</pre>

Kubeflow:

<pre class="devsite-terminal">tfx pipeline delete --engine=kubeflow --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

ローカル:

<pre class="devsite-terminal">tfx pipeline delete --engine=local --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;
</pre>

### list

指定されたオーケストレーター内のすべてのパイプラインを一覧表示します。

Usage:

<pre class="devsite-click-to-copy devsite-terminal">tfx pipeline list [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>（オプション）Kubeflow Pipelines API サービスのエンドポイント。Kubeflow Pipelines API サービスのエンドポイントは、Kubeflow Pipelines ダッシュボードの URL と同じです。エンドポイント値は次のようになります。</p>
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
    <p>（オプション）パイプラインに使用されるオーケストレーター。エンジンの値は、次のいずれかの値と一致する必要があります。</p>
    <ul>
      <li>
<strong>airflow</strong>: エンジンを Apache Airflow に設定します</li>
      <li>
<strong>beam</strong>: エンジンを Apache Beam に設定します</li>
      <li>
<strong>kubeflow</strong>: エンジンを Kubeflow に設定します</li>
      <li>
<strong>local</strong>: エンジンをローカルオーケストレーターに設定します</li>
    </ul>
    <p>エンジンが設定されていない場合、エンジンは環境に基づいて自動検出されます。</p>
    <p>      **要注意：パイプライン構成ファイルの DagRunner に必要とされるオーケストレーターは、選択されたエンジンまたは自動検出されたエンジンと一致する必要があります。エンジンの自動検出は、ユーザー環境に基づいています。Apache Airflow と Kubeflow Pipelines がインストールされていない場合、デフォルトでローカルオーケストレーターが使用されます。</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>    （オプション）IAP で保護されたエンドポイントのクライアント ID。</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>（オプション）Kubeflow Pipelines API に接続するための Kubernetes 名前空間。名前空間が指定されていない場合、値はデフォルトで<code>kubeflow</code>になります。</dd>



#### 例:

Apache Airflow:

<pre class="devsite-terminal">tfx pipeline list --engine=airflow
</pre>

Kubeflow:

<pre class="devsite-terminal">tfx pipeline list --engine=kubeflow --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

ローカル:

<pre class="devsite-terminal">tfx pipeline list --engine=local
</pre>

## tfx run

`tfx run`コマンドグループのコマンドの構造は次のとおりです。

<pre class="devsite-terminal">tfx run &lt;var&gt;command&lt;/var&gt; &lt;var&gt;required-flags&lt;/var&gt; [&lt;var&gt;optional-flags&lt;/var&gt;]
</pre>

`tfx run`コマンドグループのコマンドの詳細については次のセクションを参照してください。

### create

オーケストレーターでパイプラインの新しい実行インスタンスを作成します。Kubeflow の場合、クラスター内のパイプラインの最新のパイプラインバージョンが使用されます。

Usage:

<pre class="devsite-click-to-copy devsite-terminal">tfx run create --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>パイプラインの名前。</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>（オプション）Kubeflow Pipelines API サービスのエンドポイント。Kubeflow Pipelines API サービスのエンドポイントは、Kubeflow Pipelines ダッシュボードの URL と同じです。エンドポイント値は次のようになります。</p>
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
    <p>（オプション）パイプラインに使用されるオーケストレーター。エンジンの値は、次のいずれかの値と一致する必要があります。</p>
    <ul>
      <li>
<strong>airflow</strong>: エンジンを Apache Airflow に設定します</li>
      <li>
<strong>beam</strong>: エンジンを Apache Beam に設定します</li>
      <li>
<strong>kubeflow</strong>: エンジンを Kubeflow に設定します</li>
      <li>
<strong>local</strong>: エンジンをローカルオーケストレーターに設定します</li>
    </ul>
    <p>エンジンが設定されていない場合、エンジンは環境に基づいて自動検出されます。</p>
    <p>      **要注意：パイプライン構成ファイルの DagRunner に必要とされるオーケストレーターは、選択されたエンジンまたは自動検出されたエンジンと一致する必要があります。エンジンの自動検出は、ユーザー環境に基づいています。Apache Airflow と Kubeflow Pipelines がインストールされていない場合、デフォルトでローカルオーケストレーターが使用されます。</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>    （オプション）IAP で保護されたエンドポイントのクライアント ID。</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>（オプション）Kubeflow Pipelines API に接続するための Kubernetes 名前空間。名前空間が指定されていない場合、値はデフォルトで<code>kubeflow</code>になります。</dd>



#### 例:

Apache Airflow:

<pre class="devsite-terminal">tfx run create --engine=airflow --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;
</pre>

Kubeflow:

<pre class="devsite-terminal">tfx run create --engine=kubeflow --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

ローカル:

<pre class="devsite-terminal">tfx run create --engine=local --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;
</pre>

### terminate

指定されたパイプラインの実行を停止します。

** 要注意：現在、Kubeflow でのみサポートされています。

Usage:

<pre class="devsite-click-to-copy devsite-terminal">tfx run terminate --run_id=&lt;var&gt;run-id&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--run_id=<var>run-id</var>
</dt>
  <dd>パイプライン実行の一意の識別子。</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>（オプション）Kubeflow Pipelines API サービスのエンドポイント。Kubeflow Pipelines API サービスのエンドポイントは、Kubeflow Pipelines ダッシュボードの URL と同じです。エンドポイント値は次のようになります。</p>
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
    <p>（オプション）パイプラインに使用されるオーケストレーター。エンジンの値は、次のいずれかの値と一致する必要があります。</p>
    <ul>
      <li>
<strong>kubeflow</strong>: エンジンを Kubeflow に設定します</li>
    </ul>
    <p>エンジンが設定されていない場合、エンジンは環境に基づいて自動検出されます。</p>
    <p>      **要注意：パイプライン構成ファイルの DagRunner に必要とされるオーケストレーターは、選択されたエンジンまたは自動検出されたエンジンと一致する必要があります。エンジンの自動検出は、ユーザー環境に基づいています。Apache Airflow と Kubeflow Pipelines がインストールされていない場合、デフォルトでローカルオーケストレーターが使用されます。</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>    （オプション）IAP で保護されたエンドポイントのクライアント ID。</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>（オプション）Kubeflow Pipelines API に接続するための Kubernetes 名前空間。名前空間が指定されていない場合、値はデフォルトで<code>kubeflow</code>になります。</dd>



#### 例:

Kubeflow:

<pre class="devsite-terminal">tfx run delete --engine=kubeflow --run_id=&lt;var&gt;run-id&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

### list

パイプラインのすべての実行を一覧表示します。

** 要注意：現在、ローカルおよび ApacheBeam ではサポートされていません。

Usage:

<pre class="devsite-click-to-copy devsite-terminal">tfx run list --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>パイプラインの名前。</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>（オプション）Kubeflow Pipelines API サービスのエンドポイント。Kubeflow Pipelines API サービスのエンドポイントは、Kubeflow Pipelines ダッシュボードの URL と同じです。エンドポイント値は次のようになります。</p>
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
    <p>（オプション）パイプラインに使用されるオーケストレーター。エンジンの値は、次のいずれかの値と一致する必要があります。</p>
    <ul>
      <li>
<strong>airflow</strong>: エンジンを Apache Airflow に設定します</li>
      <li>
<strong>kubeflow</strong>: エンジンを Kubeflow に設定します</li>
    </ul>
    <p>エンジンが設定されていない場合、エンジンは環境に基づいて自動検出されます。</p>
    <p>      **要注意：パイプライン構成ファイルの DagRunner に必要とされるオーケストレーターは、選択されたエンジンまたは自動検出されたエンジンと一致する必要があります。エンジンの自動検出は、ユーザー環境に基づいています。Apache Airflow と Kubeflow Pipelines がインストールされていない場合、デフォルトでローカルオーケストレーターが使用されます。</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>    （オプション）IAP で保護されたエンドポイントのクライアント ID。</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>（オプション）Kubeflow Pipelines API に接続するための Kubernetes 名前空間。名前空間が指定されていない場合、値はデフォルトで<code>kubeflow</code>になります。</dd>



#### 例:

Apache Airflow:

<pre class="devsite-terminal">tfx run list --engine=airflow --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;
</pre>

Kubeflow:

<pre class="devsite-terminal">tfx run list --engine=kubeflow --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

### status

現在の実行のステータスを返します。

** 要注意：現在、ローカルおよび ApacheBeam ではサポートされていません。

Usage:

<pre class="devsite-click-to-copy devsite-terminal">tfx run status --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; --run_id=&lt;var&gt;run-id&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>パイプラインの名前。</dd>
  <dt>--run_id=<var>run-id</var>
</dt>
  <dd>パイプライン実行の一意の識別子。</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>（オプション）Kubeflow Pipelines API サービスのエンドポイント。Kubeflow Pipelines API サービスのエンドポイントは、Kubeflow Pipelines ダッシュボードの URL と同じです。エンドポイント値は次のようになります。</p>
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
    <p>（オプション）パイプラインに使用されるオーケストレーター。エンジンの値は、次のいずれかの値と一致する必要があります。</p>
    <ul>
      <li>
<strong>airflow</strong>: エンジンを Apache Airflow に設定します</li>
      <li>
<strong>kubeflow</strong>: エンジンを Kubeflow に設定します</li>
    </ul>
    <p>エンジンが設定されていない場合、エンジンは環境に基づいて自動検出されます。</p>
    <p>      **要注意：パイプライン構成ファイルの DagRunner に必要とされるオーケストレーターは、選択されたエンジンまたは自動検出されたエンジンと一致する必要があります。エンジンの自動検出は、ユーザー環境に基づいています。Apache Airflow と Kubeflow Pipelines がインストールされていない場合、デフォルトでローカルオーケストレーターが使用されます。</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>    （オプション）IAP で保護されたエンドポイントのクライアント ID。</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>（オプション）Kubeflow Pipelines API に接続するための Kubernetes 名前空間。名前空間が指定されていない場合、値はデフォルトで<code>kubeflow</code>になります。</dd>



#### 例:

Apache Airflow:

<pre class="devsite-terminal">tfx run status --engine=airflow --run_id=&lt;var&gt;run-id&lt;/var&gt; --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;
</pre>

Kubeflow:

<pre class="devsite-terminal">tfx run status --engine=kubeflow --run_id=&lt;var&gt;run-id&lt;/var&gt; --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

### delete

指定されたパイプラインの実行を削除します。

** 要注意：現在、Kubeflow でのみサポートされています。

Usage:

<pre class="devsite-click-to-copy devsite-terminal">tfx run delete --run_id=&lt;var&gt;run-id&lt;/var&gt; [--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;]
</pre>

<dl>
  <dt>--run_id=<var>run-id</var>
</dt>
  <dd>パイプライン実行の一意の識別子。</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>（オプション）Kubeflow Pipelines API サービスのエンドポイント。Kubeflow Pipelines API サービスのエンドポイントは、Kubeflow Pipelines ダッシュボードの URL と同じです。エンドポイント値は次のようになります。</p>
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
    <p>（オプション）パイプラインに使用されるオーケストレーター。エンジンの値は、次のいずれかの値と一致する必要があります。</p>
    <ul>
      <li>
<strong>kubeflow</strong>: エンジンを Kubeflow に設定します</li>
    </ul>
    <p>エンジンが設定されていない場合、エンジンは環境に基づいて自動検出されます。</p>
    <p>      **要注意：パイプライン構成ファイルの DagRunner に必要とされるオーケストレーターは、選択されたエンジンまたは自動検出されたエンジンと一致する必要があります。エンジンの自動検出は、ユーザー環境に基づいています。Apache Airflow と Kubeflow Pipelines がインストールされていない場合、デフォルトでローカルオーケストレーターが使用されます。</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>    （オプション）IAP で保護されたエンドポイントのクライアント ID。</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>（オプション）Kubeflow Pipelines API に接続するための Kubernetes 名前空間。名前空間が指定されていない場合、値はデフォルトで<code>kubeflow</code>になります。</dd>



#### 例:

Kubeflow:

<pre class="devsite-terminal">tfx run delete --engine=kubeflow --run_id=&lt;var&gt;run-id&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

## tfx template[実験的]

`tfx template`コマンドグループのコマンドの構造は次のとおりです。

<pre class="devsite-terminal">tfx template &lt;var&gt;command&lt;/var&gt; &lt;var&gt;required-flags&lt;/var&gt; [&lt;var&gt;optional-flags&lt;/var&gt;]
</pre>

`tfx template`コマンドグループのコマンドの詳細は次のセクションを参照してください。テンプレートは実験的な機能であり、随時変更される可能性があります。

### list

利用可能な TFX パイプラインテンプレートを一覧表示します。

Usage:

<pre class="devsite-click-to-copy devsite-terminal">tfx template list
</pre>

### copy

テンプレートを宛先ディレクトリにコピーします。

Usage:

<pre class="devsite-click-to-copy devsite-terminal">tfx template copy --model=&lt;var&gt;model&lt;/var&gt; --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
--destination_path=&lt;var&gt;destination-path&lt;/var&gt;
</pre>

<dl>
  <dt>--model=<var>model</var>
</dt>
  <dd>パイプラインテンプレートにより構築されたモデルの名前。</dd>
  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>パイプラインの名前。</dd>
  <dt>--destination_path=<var>destination-path</var>
</dt>
  <dd>テンプレートのコピー先のパス。</dd>
</dl>

## TFX CLI フラグを理解する

### 一般的なフラグ

<dl>
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>パイプラインに使用されるオーケストレーター。エンジンの値は、次のいずれかの値と一致する必要があります。</p>
    <ul>
      <li>
<strong>airflow</strong>: エンジンを Apache Airflow に設定します</li>
      <li>
<strong>beam</strong>: エンジンを Apache Beam に設定します</li>
      <li>
<strong>kubeflow</strong>: エンジンを Kubeflow に設定します</li>
      <li>
<strong>local</strong>: エンジンをローカルオーケストレーターに設定します</li>
    </ul>
    <p>エンジンが設定されていない場合、エンジンは環境に基づいて自動検出されます。</p>
    <p>      **要注意：パイプライン構成ファイルの DagRunner に必要とされるオーケストレーターは、選択されたエンジンまたは自動検出されたエンジンと一致する必要があります。エンジンの自動検出は、ユーザー環境に基づいています。Apache Airflow と Kubeflow Pipelines がインストールされていない場合、デフォルトでローカルオーケストレーターが使用されます。</p>
  </dd>
</dl>

  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>パイプラインの名前。</dd>


  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>パイプライン構成ファイルへのパス。</dd>


  <dt>--run_id=<var>run-id</var>
</dt>
  <dd>パイプライン実行の一意の識別子。</dd>





### Kubeflow 特定のフラグ

<dl>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>Kubeflow Pipelines API サービスのエンドポイント。Kubeflow Pipelines API サービスのエンドポイントは、Kubeflow Pipelines ダッシュボードの URL と同じです。エンドポイント値は次のようになります。</p>
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
  <dd>IAP で保護されたエンドポイントのクライアント ID。</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>Kubeflow Pipelines API に接続するための Kubernetes 名前空間。名前空間が指定されていない場合、値はデフォルトで<code>kubeflow</code>になります。</dd>


  <dt>--package_path=<var>package-path</var>
</dt>
  <dd>
    <p>コンパイルされたパイプラインへのファイルとしてのパス。コンパイルされたパイプラインは、圧縮ファイル(<code>.tar.gz</code>、<code>.tgz</code>、または、<code>.zip</code>) または YAML ファイル(<code>.yaml</code>または<code>.yml</code>)である必要があります。</p>
    <p><var>package-path</var> が指定されていない場合、TFX はデフォルトパスとして次を使用します。<code>&lt;var&gt;current_directory&lt;/var&gt;/&lt;var&gt;pipeline_name&lt;/var&gt;.tar.gz</code></p>
  </dd>





## TFX CLI により生成されたファイル

パイプラインが作成されて実行されると、パイプライン管理用に複数のファイルが生成されます。

- ${HOME}/tfx/local, beam, airflow
    - 構成から読み取られたパイプラインメタデータは、`${HOME}/tfx/${ORCHESTRATION_ENGINE}/${PIPELINE_NAME}`に保存されます。`AIRFLOW_HOME`や`KUBEFLOW_HOME`などの環境変数を設定することでこの場所をカスタマイズできます。この動作は、将来のリリースで変更される可能性があります。このディレクトリは、実行の作成またはパイプラインの更新に必要なパイプライン情報（パイプライン ID を含む）を Kubeflow Pipelines クラスタに格納するために使用されます。
    - TFX 0.25 より前では、これらのファイルは`${HOME}/${ORCHESTRATION_ENGINE}`の下にありました。スムーズに移行するために、TFX 0.25 では、古い場所にあるファイルは自動的に新しい場所に移動されます。
    - TFX 0.27 以降、kubeflow はこれらのメタデータファイルをローカルファイルシステムに作成しません。kubeflow が作成する他のファイルについては、以下を参照してください。
- (Kubeflow のみ) Dockerfile、build.yaml、*pipeline_name*.tar.gz
    - Kubeflow Pipelines では、パイプラインに2種類の入力が必要です。これらのファイルは、現行のディレクトリの TFX により生成されます。
    - 1つ目は、パイプラインでコンポーネントを実行するために使用されるコンテナイメージです。このコンテナイメージは、TFX CLI で Kubeflow パイプラインのパイプラインを作成するときに作成されます。TFX は、[skaffold](https://skaffold.dev/) を使用してコンテナイメージを構築します。`Dockerfile`と`build.yaml`は TFX により生成され、skaffold に渡されます（これらのファイル名は指定されており、現時点では変更できません）。
    - TFX CLI は、指定されたパイプライン定義を Kubeflow Pipelines が理解できる形式に*コンパイル*します。コンパイルの結果は、`_pipeline_name_.tar.gz`として保存されます。このファイル名は、`--package-path`フラグを使用してカスタマイズできます。
