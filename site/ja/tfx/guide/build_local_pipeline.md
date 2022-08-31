# TFX パイプラインをローカルで構築する

TFX を使用すると、次の目的で機械学習（ML）ワークフローをパイプラインとして簡単に調整できます。

- モデルを定期的に再トレーニングし、評価してデプロイできるように、ML プロセスを自動化する。
- モデルパフォーマンスのディープ解析や新たにトレーニングされたモデルの検証を含む ML パイプラインを作成し、パフォーマンスと信頼性を確保する。
- 異常に関するトレーニングデータを監視し、トレーニングとサービング間の歪みを排除する
- パイプラインをさまざまなハイパーパラメータの組み合わせで実行することで、実験を加速させる。

一般的なパイプライン開発プロセスは、ローカルマシンで開始され、データ分析とコンポーネントのセットアップが行われてから本番環境にデプロイされます。このガイドでは、パイプラインをローカルで構築する 2 つの方法について説明します。

- ML ワークフローのニーズに適合するように TFX パイプラインテンプレートをカスタマイズします。TFX パイプラインテンプレートは事前構築済みのワークフローであり、TFX 標準コンポーネントを使ったベストプラクティスを実演します。
- TFX を使ってパイプラインを構築します。このユースケースでは、テンプレートを使用せずにパイプラインを定義します。

パイプラインを開発しているときに、 `LocalDagRunner` を使用してパイプラインを実行できます。パイプラインコンポーネントが適切に定義およびテストされたら、Kubeflow や Airflow などの本番環境グレードのオーケストレーターを使用します。

## 始める前に

TFX は Python パッケージであるため、仮想環境や Docker コンテナなどの Python 開発環境をセットアップする必要があります。

```bash
pip install tfx
```

TFX パイプラインが初めての方は、読み進める前に、[TFX パイプラインの中心的概念を学習](understanding_tfx_pipelines)してください。

## テンプレートを使用してパイプラインを構築する

TFX パイプラインテンプレートを使用すると、ユースケースに合わせてカスタマイズできるパイプライン定義のビルド済みセットを提供して、パイプライン開発を簡単に開始できます。

以降のセクションでは、テンプレートのコピーを作成し、ニーズに合わせてカスタマイズする方法を説明します。

### パイプラインテンプレートのコピーを作成する

1. 利用可能な TFX パイプラインテンプレートを一覧表示します。

    <pre class="devsite-click-to-copy devsite-terminal">    tfx template list
        </pre>

2. 一覧からテンプレートを選択します。

    <pre class="devsite-click-to-copy devsite-terminal">    tfx template copy --model=&lt;var&gt;template&lt;/var&gt; --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
        --destination_path=&lt;var&gt;destination-path&lt;/var&gt;
        </pre>

    上記のコードから、次の項目を置き換えてください。

    - <var>template</var>: コピーするテンプレートの名前です。
    - <var>pipeline-name</var>: 作成するテンプレートの名前です。
    - <var>destination-path</var>: テンプレートのコピー先のパスです。

    詳細は、[`tfx template copy` コマンド](cli#copy)をご覧ください。

3. パイプラインテンプレートのコピーが、指定したパスに作成されました。

注意: このガイドの残りの部分は、`penguin` テンプレートを選択したことを前提としています。

### パイプラインテンプレートを調べる

このセクションでは、テンプレートによって作成される基盤の概要を説明します。

1. パイプラインのルートディレクトリにコピーされたディレクトリとファイルを調べます

    - **pipeline** ディレクトリ

        - `pipeline.py` - パイプラインを定義し、使用されているコンポーネントを一覧表示します
        - `configs.py` - データの送信元や使用されているオーケストレーターなどの構成の詳細を保持します

    - **data** ディレクトリ

        - これには通常、`ExampleGen` のデフォルトのソースである `data.csv` ファイルが含まれています。`configs.py` でデータソースを変更できます。

    - 前処理コードとモデル実装を含む **models** ディレクトリ

    - テンプレートは、ローカル環境と Kubeflow の DAG ランナーをコピーします。

    - 一部のテンプレートには Python ノートブックも含まれているため、Machine Learning MetaData を使用してデータと成果物を探索できます。

2. パイプラインディレクトリで次のコマンドを実行します。

    <pre class="devsite-click-to-copy devsite-terminal">    tfx pipeline create --pipeline_path local_runner.py
        </pre>

    <pre class="devsite-click-to-copy devsite-terminal">    tfx run create --pipeline_name &lt;var&gt;pipeline_name&lt;/var&gt;
        </pre>

    このコマンドは、`LocalDagRunner` を使用してパイプライン実行を作成します。これにより、パイプラインに次のディレクトリが追加されます。

    - ローカルで使用される ML Metadata ストアが含まれる **tfx_metadata** ディレクトリ。
    - パイプラインのファイル出力が含まれる **tfx_pipeline_output** ディレクトリ。

    注意: `LocalDagRunner` は、TFX でサポートされているいくつかのオーケストレーターの 1 つです。これは、パイプラインをローカルで実行して反復を高速化する場合に特に適しています。データセットが小さい場合もあります。`LocalDagRunner` は単一のマシンで実行されるため、実稼働での使用には適さない場合があり、システムが使用できなくなった場合に作業が失われる可能性が高くなります。TFX は、Apache Beam、Apache Airflow、KubeflowPipeline などのオーケストレーターもサポートしています。別のオーケストレーターで TFX を使用している場合は、そのオーケストレーターに適切な DAG ランナーを使用してください。

    注意: この記事の執筆時点では、`LocalDagRunner` は `penguin` テンプレートで使用されていますが、`taxi` テンプレートは Apache Beam を使用しています。 `taxi` テンプレートの構成ファイルは Beam を使用するように設定されており、CLI コマンドも同じです。

3. パイプラインの `pipeline/configs.py` ファイルを開き、内容を確認します。このスクリプトは、パイプラインとコンポーネント関数で使用される構成オプションを定義します。ここで、データソースの場所や実行のトレーニングステップ数などを指定します。

4. パイプラインの `pipeline/pipeline.py` ファイルを開き、内容を確認します。このスクリプトは、TFX パイプラインを作成します。最初、パイプラインには `ExampleGen` コンポーネントのみが含まれています。

    - パイプラインにさらにステップを追加するには、 <code>pipeline.py</code> の <strong>TODO</strong> コメントの指示に従ってください。

5. `local_runner.py` ファイルを開き、内容を確認します。このスクリプトは、パイプライン実行を作成し、<code>data_path</code> や `preprocessing_fn` などの実行の<em>パラメータ</em>を指定します。

6. テンプレートが作成した基盤を確認し、`LocalDagRunner` を使ったパイプライン実行を作成しました。次のステップでは、要件に合わせてテンプレートをカスタマイズします。

### パイプラインをカスタマイズする

このセクションでは、テンプレートのカスタマイズを開始する方法の概要を説明します。

1. パイプラインを設計します。テンプレートが提供する基盤を元に TFX 標準コンポーネントを使ってタブ区切りデータのパイプラインを実装することができます。既存の ML ワークフローをパイプラインに移動する場合、[TFX 標準コンポーネント](index#tfx_standard_components)を活用できるようにコードを校正する必要がある場合があります。また、ワークフロー独自の特徴量を実装する、またはTFX 標準コンポーネントが未対応の[カスタムコンポーネント](understanding_custom_components)を作成する必要もあります。

2. パイプラインを設計したら、次のプロセスに従ってパイプラインのカスタマイズを繰り返し行います。パイプラインへのデータを飲み込むコンポーネントから始めます。通常は`ExampleGen` コンポーネントです。

    1. ユースケースに合うように、パイプラインまたはコンポーネントをカスタマイズします。このカスタマイズには次のような変更が含まれることがあります。

        - パイプラインパラメータの変更
        - パイプラインへのコンポーネントの追加、またはパイプラインからの削除
        - データ入力ソースの変更。このデータソースはファイルまたは BigQuery などのサービスのクエリです。
        - パイプライン内のコンポーネントの構成の変更
        - コンポーネントのカスタマイズ関数の変更

    2. `local_runner.py` スクリプト、または異なるオーケストレーターを使用している場合は、別の該当する DAG ランナーを使用してローカルでコンポーネントを実行します。スクリプトが失敗した場合、失敗をデバッグし、スクリプトの実行を再試行します。

    3. このカスタマイズがうまくいったら、次のカスタマイズに進みます。

3. この作業を進めると、テンプレートの各ステップをニーズに合わせてカスタマイズできます。

## カスタムパイプラインを構築する

次の手順に従って、テンプレートを使用せずにカスタムパイプラインを構築する方法を学習します。

1. パイプラインを設計します。TFX 標準コンポーネントには、完全な ML ワークフローの実装に役立つ実績のある機能が提供されています。既存の ML ワークフローをパイプラインに移動する場合、TFX 標準コンポーネントを活用できるようにコードを校正する必要がある場合があります。また、データ拡張などの特徴量を実装する[カスタムコンポーネント](understanding_custom_components)を作成する必要もある場合があります。

    - [標準 TFX コンポーネント](index#tfx_standard_components)についてお読みください。
    - [カスタムコンポーネント](understanding_custom_components)についてお読みください。

2. スクリプトファイルを作成して、次の例を参考にパイプラインを定義します。このガイドではこのファイルを `my_pipeline.py` を呼びます。

    <pre class="devsite-click-to-copy prettyprint">    import os
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

    以降のステップでは、`create_pipeline` でパイプラインを定義し、ローカルランナーを使ってパイプラインをローカル実行します。

    次のプロセスに従って、対話的にパイプラインを構築します。

    1. ユースケースに合うように、パイプラインまたはコンポーネントをカスタマイズします。このカスタマイズには次のような変更が含まれることがあります。

        - パイプラインパラメータの変更
        - パイプラインへのコンポーネントの追加、またはパイプラインからの削除
        - データ入力ファイルの変更
        - パイプライン内のコンポーネントの構成の変更
        - コンポーネントのカスタマイズ関数の変更

    2. ローカルランナーを使用するか、スクリプトを直接実行して、ローカルでコンポーネントを実行します。スクリプトが失敗した場合、失敗をデバッグし、スクリプトの実行を再試行します。

    3. このカスタマイズがうまくいったら、次のカスタマイズに進みます。

    パイプラインのワークフローの最初のノードから開始します。通常、最初のノードはパイプラインにデータを取り込みます。

3. ワークフローの最初のノードをパイプラインに追加します。この例のパイプラインは、`ExampleGen` 標準コンポーネントを使用して、`./data` にあるディレクトリから CSV を読み込みます。

    <pre class="devsite-click-to-copy prettyprint">    from tfx.components import CsvExampleGen

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

    `CsvExampleGen` は、指定されたデータパスの CSV 内のデータを使用して、シリアル化されたサンプルレコードを作成します。`CsvExampleGen` コンポーネントの `input_base` パラメータをデータルートに設定します。

4. `my_pipeline.py` と同じディレクトリに `data` ディレクトリを作成します。小さな CSV ファイルを `data` ディレクトリに追加します。

5. 次のコマンドを使用して、`my_pipeline.py` スクリプトを実行します。

    <pre class="devsite-click-to-copy devsite-terminal">    python my_pipeline.py
        </pre>

    結果は次のようになります。

    <pre>    INFO:absl:Component CsvExampleGen depends on [].
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

6. 続けて、繰り返しながらパイプラインにコンポーネントを追加します。
