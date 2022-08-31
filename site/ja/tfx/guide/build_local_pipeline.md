# TFX パイプラインをローカルで構築する

TFX を使用すると、次の目的で機械学習（ML）ワークフローをパイプラインとして簡単に調整できます。

- Automate your ML process, which lets you regularly retrain, evaluate, and deploy your model.
- Create ML pipelines which include deep analysis of model performance and validation of newly trained models to ensure performance and reliability.
- Monitor training data for anomalies and eliminate training-serving skew
- Increase the velocity of experimentation by running a pipeline with different sets of hyperparameters.

一般的なパイプライン開発プロセスは、ローカルマシンで開始され、データ分析とコンポーネントのセットアップが行われてから本番環境にデプロイされます。このガイドでは、パイプラインをローカルで構築する 2 つの方法について説明します。

- Customize a TFX pipeline template to fit the needs of your ML workflow. TFX pipeline templates are prebuilt workflows that demonstrate best practices using the TFX standard components.
- Build a pipeline using TFX. In this use case, you define a pipeline without starting from a template.

パイプラインを開発しているときに、 `LocalDagRunner` を使用してパイプラインを実行できます。パイプラインコンポーネントが適切に定義およびテストされたら、Kubeflow や Airflow などの本番環境グレードのオーケストレーターを使用します。

## 始める前に

TFX は Python パッケージであるため、仮想環境や Docker コンテナなどの Python 開発環境をセットアップする必要があります。

```bash
pip install tfx
```

If you are new to TFX pipelines, [learn more about the core concepts for TFX pipelines](understanding_tfx_pipelines) before continuing.

## Build a pipeline using a template

TFX パイプラインテンプレートを使用すると、ユースケースに合わせてカスタマイズできるパイプライン定義のビルド済みセットを提供して、パイプライン開発を簡単に開始できます。

The following sections describe how to create a copy of a template and customize it to meet your needs.

### Create a copy of the pipeline template

1. 利用可能な TFX パイプラインテンプレートを一覧表示します。

    <pre class="devsite-click-to-copy devsite-terminal">    tfx template list
        </pre>

2. 一覧からテンプレートを選択します。

    <pre class="devsite-click-to-copy devsite-terminal">    tfx template copy --model=&lt;var&gt;template&lt;/var&gt; --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
        --destination_path=&lt;var&gt;destination-path&lt;/var&gt;
        </pre>

    Replace the following:

    - <var>template</var>: The name of the template you want to copy.
    - <var>pipeline-name</var>: The name of the pipeline to create.
    - <var>destination-path</var>: The path to copy the template into.

    Learn more about the [`tfx template copy` command](cli#copy).

3. A copy of the pipeline template has been created at the path you specified.

注意: このガイドの残りの部分は、`penguin` テンプレートを選択したことを前提としています。

### Explore the pipeline template

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

### Customize your pipeline

このセクションでは、テンプレートのカスタマイズを開始する方法の概要を説明します。

1. Design your pipeline. The scaffolding that a template provides helps you implement a pipeline for tabular data using the TFX standard components. If you are moving an existing ML workflow into a pipeline, you may need to revise your code to make full use of [TFX standard components](index#tfx_standard_components). You may also need to create [custom components](understanding_custom_components) that implement features which are unique to your workflow or that are not yet supported by TFX standard components.

2. Once you have designed your pipeline, iteratively customize the pipeline using the following process. Start from the component that ingests data into your pipeline, which is usually the `ExampleGen` component.

    1. Customize the pipeline or a component to fit your use case. These customizations may include changes like:

        - Changing pipeline parameters.
        - Adding components to the pipeline or removing them.
        - Replacing the data input source. This data source can either be a file or queries into services such as BigQuery.
        - Changing a component's configuration in the pipeline.
        - Changing a component's customization function.

    2. `local_runner.py` スクリプト、または異なるオーケストレーターを使用している場合は、別の該当する DAG ランナーを使用してローカルでコンポーネントを実行します。スクリプトが失敗した場合、失敗をデバッグし、スクリプトの実行を再試行します。

    3. Once this customization is working, move on to the next customization.

3. Working iteratively, you can customize each step in the template workflow to meet your needs.

## Build a custom pipeline

Use the following instructions to learn more about building a custom pipeline without using a template.

1. Design your pipeline. The TFX standard components provide proven functionality to help you implement a complete ML workflow. If you are moving an existing ML workflow into a pipeline, you may need to revise your code to make full use of TFX standard components. You may also need to create [custom components](understanding_custom_components) that implement features such as data augmentation.

    - Learn more about [standard TFX components](index#tfx_standard_components).
    - Learn more about [custom components](understanding_custom_components).

2. Create a script file to define your pipeline using the following example. This guide refers to this file as `my_pipeline.py`.

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

    Iteratively build your pipeline using the following process.

    1. Customize the pipeline or a component to fit your use case. These customizations may include changes like:

        - Changing pipeline parameters.
        - Adding components to the pipeline or removing them.
        - Replacing a data input file.
        - Changing a component's configuration in the pipeline.
        - Changing a component's customization function.

    2. ローカルランナーを使用するか、スクリプトを直接実行して、ローカルでコンポーネントを実行します。スクリプトが失敗した場合、失敗をデバッグし、スクリプトの実行を再試行します。

    3. Once this customization is working, move on to the next customization.

    Start from the first node in your pipeline's workflow, typically the first node ingests data into your pipeline.

3. Add the first node in your workflow to your pipeline. In this example, the pipeline uses the `ExampleGen` standard component to load a CSV from a directory at `./data`.

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

4. Create a `data` directory in the same directory as `my_pipeline.py`. Add a small CSV file to the `data` directory.

5. 次のコマンドを使用して、`my_pipeline.py` スクリプトを実行します。

    <pre class="devsite-click-to-copy devsite-terminal">    python my_pipeline.py
        </pre>

    The result should be something like the following:

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

6. Continue to iteratively add components to your pipeline.
