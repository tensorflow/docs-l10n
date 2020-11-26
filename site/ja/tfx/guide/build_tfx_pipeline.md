# TFX パイプラインを構築する

TFX を使用すると、機械学習（ML）ワークフローのパイプラインとしてのオーケストレーションをより簡単に行え、次の項目を実現することができます。

- モデルを定期的に再トレーニングし、評価してデプロイできるように、ML プロセスを自動化する。
- モデルパフォーマンスのディープ解析や新たにトレーニングされたモデルの検証を含む ML パイプラインを作成し、パフォーマンスと信頼性を確保する。
- 異常に関するトレーニングデータを監視し、トレーニングとサービング間の歪みを排除する
- パイプラインを様々なハイパーパラメータの組み合わせで実行することで、実験を加速する

このガイドでは、次の 2 つの方法によるパイプライン作成を説明します。

- ML ワークフローのニーズに適合するように TFX パイプラインテンプレートをカスタマイズします。TFX パイプラインテンプレートは事前構築済みのワークフローであり、TFX 標準コンポーネントを使ったベストプラクティスを実演します。
- TFX を使ってパイプラインを構築します。このユースケースでは、テンプレートを使用せずにパイプラインを定義します。

TFX パイプラインが初めての方は、読み進める前に、[TFX パイプラインの中心的概念を学習](understanding_tfx_pipelines)してください。

## TFX パイプラインの概要

注意: 詳しい内容に掘り下げる前に、まずはパイプラインを構築してみようと思う方は、[テンプレートを使ってパイプラインを構築する](#build_a_pipeline_using_a_template)ことから始めてください。

TFX パイプラインは、[`Pipeline` クラス](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/pipeline.py){: .external } を使って定義されます。次の例では、`Pipeline` クラスの使用方法を実演しています。

<pre class="devsite-click-to-copy prettyprint">pipeline.Pipeline(
    pipeline_name=<var>pipeline-name</var>,
    pipeline_root=<var>pipeline-root</var>,
    components=<var>components</var>,
    enable_cache=<var>enable-cache</var>,
    metadata_connection_config=<var>metadata-connection-config</var>,
    beam_pipeline_args=<var>beam_pipeline_args</var>
)
</pre>

上記のコードから、次の項目を置き換えてください。

- <var>pipeline-name</var>: このパイプラインの名前です。パイプライン名は一意である必要があります。

    TFX は、ML メタデータでパイプライン面を使用してコンポーネント入力アーティファクトをクエリします。パイプライン名を再利用した場合、どのような動作になるか予測できません。

- <var>pipeline-root</var>: このパイプラインの出力のルートパスです。ルートパスは、オーケストレータが読み取りと書き込みのアクセス権を与えられたディレクトリへのフルパスです。TFX はランタイム時にこのパイプラインルートを使用して、コンポーネントアーティファクトの出力パスを生成します。ローカル上のディレクトリ、または Google Cloud Storage や HDFS などのサポートされた分散ファイルシステム上のディレクトリを使用できます。

- <var>components</var>: このパイプラインのワークフローを構成するコンポーネントインスタンスのリストです。

- <var>enable-cache</var>:（オプション）パイプライン実行を加速するために、このパイプラインでキャッシュを使用するかどうかを指定するブール型の値です。

- <var>metadata-connection-config</var>:（オプション）ML メタデータの接続構成です。

- <var>beam_pipeline_args</var>:（オプション）Beam を使って計算を実行するすべてのコンポーネントに関し、Apache Beam ランナーに渡される一連の引数です。

### コンポーネント実行グラフを定義する

コンポーネントインスタンスは、アーティファクトを出力として生成し、通常は上流のコンポーネントインスタンスで入力として生成されたアーティファクトに依存します。コンポーネントインスタンスの実行シーケンスは、アーティファクト依存関係の有向非巡回グラフ（DAG）を作成することで決定されます。

たとえば、`ExampleGen` 標準コンポーネントはCSV ファイルからデータを飲み込んで、シリアル化されたサンプルレコードを出力します。`StatisticsGen` 標準コンポーネントはそのサンプルレコードを入力として受け取り、データセットの統計を生成します。この例では、`SchemaGen` が `ExampleGen` の出力に依存しているため、`StatisticsGen` のインスタンスは `ExampleGen` の後に続く必要があります。

また、コンポーネントの [`add_upstream_node` と `add_downstream_node`](https://github.com/tensorflow/tfx/blob/master/tfx/components/base/base_node.py){: .external } メソッドを使って、タスクベースの依存関係を定義することもできます。現在のコンポーネントを指定されたコンポーネントの後に実行する場合は `add_upstream_node`、現在のコンポーネントを指定されたコンポーネントの前に実行する場合は `add_upstream_node` を使用します。

注意: 通常、タスクベースの依存関係の使用は推奨されません。アーティファクト依存関係で実行グラフを定義すると、TFX の持つ、アーティファクト系統の自動トラッキングとキャシング機能を利用できます。

### キャッシング

TFX パイプラインをキャッシングすると、前回のパイプライン実行で同一の入力セットを使って実行されたコンポーネントをパイプラインから省略することができます。キャッシングが有効になっている場合、パイプラインは各コンポーネントのシグネチャ、コンポーネント、および入力セットをこのパイプラインの前回のコンポーネント実行に一致させようとします。一致する場合、パイプラインは前回の実行で得たコンポーネント出力を使用し、一致しない場合は、コンポーネントを実行します。

パイプラインに非確定的コンポーネントを使用している場合は、キャッシングを使用しないでください。たとえば、パイプラインの乱数を作成するコンポーネントを作成した場合、キャッシングが有効であれば、このコンポーネントは一度しか実行されません。この例の後続の実行では、乱数を生成する代わりに、初回の実行で得た乱数が使用されてしまうことになります。

## テンプレートを使用してパイプラインを構築する

TFX パイプラインテンプレートを使うと、事前構築済みのパイプラインをユースケースに合わせてカスタマイズできるため、パイプラインの作成をより簡単に行えます。

以降のセクションでは、テンプレートのコピーを作成し、ニーズに合わせてカスタマイズする方法を説明します。

### パイプラインテンプレートのコピーを作成する

1. 次のコマンドを実行して、TFX パイプラインテンプレートのリストを取得します。

    <pre class="devsite-click-to-copy devsite-terminal">tfx template list
    </pre>

2. リストからテンプレートを選択します。現在利用できるテンプレートは **taxi** のみです。選択したら、次のコマンドを実行します。

    <pre class="devsite-click-to-copy devsite-terminal">tfx template copy --model=<var>template</var> --pipeline_name=<var>pipeline-name</var> \
    --destination_path=<var>destination-path</var>
    </pre>

    上記のコードから、次の項目を置き換えてください。

    - <var>template</var>: コピーするテンプレートの名前です。
    - <var>pipeline-name</var>: 作成するテンプレートの名前です。
    - <var>destination-path</var>: テンプレートのコピー先のパスです。

    詳細は、[`tfx template copy` コマンド](cli#copy)をご覧ください。

3. パイプラインテンプレートのコピーが、指定したパスに作成されました。

### パイプラインテンプレートを調べる

このセクションでは、**taxi** テンプレートによって作成される基盤の概要を説明します。

1. テンプレートからパイプラインにコピーされたファイルを調べます。**taxi** テンプレートが作成する項目を次に示します。

    - **data.csv** ファイルを含む **data** ディレクトリ

    - `tf.estimators` と Keras を使った前処理コードとモデル実装を含む **models** ディレクトリ

    - パイプライン実装と構成スクリプトを含む **pipeline** ディレクトリ

    - テンプレートは、次の項目をコピー先のパスに作成します。

        - Apache Beam と Kuberflow Pipelines 用の DAG ランナーコード
        - [ML メタデータ](mlmd)ストアのアーティファクトを調べるためのノートブック

2. pipeline ディレクトリで次のコマンドを実行します。

    <pre class="devsite-click-to-copy devsite-terminal">python beam_dag_runner.py
    </pre>

    このコマンドは、Apache Beam を使ったパイプライン実行を作成し、パイプラインに次のディレクトリを追加します。

    - Apache Beam がローカルで使用する ML メタデータストアを含む **tfx_metadata** ディレクトリ
    - パイプラインのファイル出力を含む **tfx_pipeline_output** ディレクトリ

    注意: Apache Beam は、TFX でサポートされているオーケストレータです。できれば小さめのデータセットを使ってパイプラインをローカルで実行し、イテレーションを高速化する場合に特に適していますが、単一のマシンで実行するため、システムが停止した場合に作業内容が失われる可能性があるため、本番環境には不向きです。TFX は、Apache Airflow や Kubeflow Pipeline などのオーケストレータもサポートしています。異なるオーケストレータで TFX を使用している場合は、そのオーケストレータに適した DAG ランナーを使用してください。

3. パイプラインの `pipeline/configs.py` ファイルを開き、コンテンツを確認します。このスクリプトはパイプラインとコンポーネント関数が使用する構成オプションを定義します。

4. パイプラインの `pipeline/pipeline.py` ファイルを開き、コンテンツを確認します。最初は、ExampleGen コンポーネントしかパイプラインに含まれていません。パイプラインにステップを追加するには、パイプラインの **TODO** コメントに記載されている指示に従ってください。

5. パイプラインの `beam_dag_runner.py` ファイルを開き、コンテンツを確認します。このスクリプトはパイプライン実行を作成し、`data_path` や `preprocessing_fn` といった実行の*パラメータ*を指定します。

6. テンプレートが作成した基盤を確認し、Apache Beam を使ったパイプライン実行を作成しました。次のステップでは、要件に合わせてテンプレートをカスタマイズします。

### パイプラインをカスタマイズする

このセクションでは、**taxi** テンプレートのカスタマイズ方法の概要を説明します。

1. パイプラインを設計します。テンプレートが提供する基盤を元に TFX 標準コンポーネントを使ってタブ区切りデータのパイプラインを実装することができます。既存の ML ワークフローをパイプラインに移動する場合、[TFX 標準コンポーネント](index#tfx_standard_components)を活用できるようにコードを校正する必要がある場合があります。また、ワークフロー独自の特徴量を実装する、またはTFX 標準コンポーネントが未対応の[カスタムコンポーネント](understanding_custom_components)を作成する必要もあります。

2. パイプラインを設計したら、次のプロセスに従ってパイプラインのカスタマイズを繰り返し行います。パイプラインへのデータを飲み込むコンポーネントから始めます。通常は`ExampleGen` コンポーネントです。

    1. ユースケースに合うように、パイプラインまたはコンポーネントをカスタマイズします。このカスタマイズには次のような変更が含まれることがあります。

        - パイプラインパラメータの変更
        - パイプラインへのコンポーネントの追加、またはパイプラインからの削除
        - データ入力ソースの変更。このデータソースはファイルまたは BigQuery などのサービスのクエリです。
        - パイプライン内のコンポーネントの構成の変更
        - コンポーネントのカスタマイズ関数の変更

    2. `beam_dag_runner.py` スクリプト、または異なるオーケストレータを使用している場合は別の該当する DAG ランナーを使用して、ローカルでコンポーネントを実行します。スクリプトが失敗すると、失敗をデバッグし、スクリプトの実行を再試行します。

    3. このカスタマイズがうまくいったら、次のカスタマイズに進みます。

3. この作業を進めると、テンプレートの各ステップをニーズに合わせてカスタマイズできます。

## カスタムパイプラインを構築する

次の手順に従って、テンプレートを使用せずにカスタムパイプラインを構築する方法を学習します。

1. パイプラインを設計します。TFX 標準コンポーネントには、完全な ML ワークフローの実装に役立つ実績のある機能が提供されています。既存の ML ワークフローをパイプラインに移動する場合、TFX 標準コンポーネントを活用できるようにコードを校正する必要がある場合があります。また、データ拡張などの特徴量を実装する[カスタムコンポーネント](understanding_custom_components)を作成する必要もある場合があります。

    - [標準 TFX コンポーネント](index#tfx_standard_components)についてお読みください。
    - [カスタムコンポーネント](understanding_custom_components)についてお読みください。

2. スクリプトファイルを作成して、次の例を参考にパイプラインを定義します。このガイドではこのファイルを `my_pipeline.py` を呼びます。

    <pre class="devsite-click-to-copy prettyprint">import os
    from typing import Optional, Text, List
    from absl import logging
    from ml_metadata.proto import metadata_store_pb2
    from tfx.orchestration import metadata
    from tfx.orchestration import pipeline
    from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

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

      return pipeline.Pipeline(
            pipeline_name=pipeline_name,
            pipeline_root=pipeline_root,
            components=components,
            enable_cache=enable_cache,
            metadata_connection_config=metadata_connection_config,
            beam_pipeline_args=beam_pipeline_args,
        )

    def run_pipeline():
      my_pipeline = create_pipeline(
          pipeline_name=PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT,
          enable_cache=ENABLE_CACHE,
          metadata_connection_config=metadata.sqlite_metadata_connection_config(METADATA_PATH)
          )

      BeamDagRunner().run(my_pipeline)

    if __name__ == '__main__':
      logging.set_verbosity(logging.INFO)
      run_pipeline()
    </pre>

    以降のステップでは、`create_pipeline` でパイプラインを定義し、`run_pipeline` で Apache Beam を使ってパイプラインをローカル実行します。

    次のプロセスに従って、対話的にパイプラインを構築します。

    1. ユースケースに合うように、パイプラインまたはコンポーネントをカスタマイズします。このカスタマイズには次のような変更が含まれることがあります。

        - パイプラインパラメータの変更
        - パイプラインへのコンポーネントの追加、またはパイプラインからの削除
        - データ入力ファイルの変更
        - パイプライン内のコンポーネントの構成の変更
        - コンポーネントのカスタマイズ関数の変更

    2. スクリプトファイルを実行し、Apache Beam または異なるオーケストレータを使用して、ローカルでコンポーネントを実行します。スクリプトが失敗すると、失敗をデバッグし、スクリプトの実行を再試行します。

    3. このカスタマイズがうまくいったら、次のカスタマイズに進みます。

    パイプラインのワークフローの最初のノードから開始します。通常、最初のノードはパイプラインにデータを取り込みます。

3. ワークフローの最初のノードをパイプラインに追加します。この例のパイプラインは、`ExampleGen` 標準コンポーネントを使用して、`./data` にあるディレクトリから CSV を読み込みます。

    <pre class="devsite-click-to-copy prettyprint">from tfx.components import CsvExampleGen
    from tfx.utils.dsl_utils import external_input

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

      example_gen = CsvExampleGen(input=external_input(data_path))
      components.append(example_gen)

      return pipeline.Pipeline(
            pipeline_name=pipeline_name,
            pipeline_root=pipeline_root,
            components=components,
            enable_cache=enable_cache,
            metadata_connection_config=metadata_connection_config,
            beam_pipeline_args=beam_pipeline_args,
        )

    def run_pipeline():
      my_pipeline = create_pipeline(
          pipeline_name=PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT,
          data_path=DATA_PATH,
          enable_cache=ENABLE_CACHE,
          metadata_connection_config=metadata.sqlite_metadata_connection_config(METADATA_PATH)
          )

      BeamDagRunner().run(my_pipeline)
    </pre>

    `CsvExampleGen` は、指定されたデータパスにある CSV のデータを使用してシリアル化されたサンプルレコードを作成します。`CsvExampleGen` コンポーネントの `input` パラメータを [`external_input`](https://github.com/tensorflow/tfx/blob/master/tfx/utils/dsl_utils.py){: .external } で設定することで、データパスがパイプラインに渡され、パスをアーティファクトとして保存することを指定します。

4. `my_pipeline.py` と同じディレクトリに `data` ディレクトリを作成します。小さな CSV ファイルを `data` ディレクトリに追加します。

5. 次のコマンドを使用して、`my_pipeline.py` スクリプトを実行し、Apache Beam または別のオーケストレータを使ってパイプラインをテストします。

    <pre class="devsite-click-to-copy devsite-terminal">python my_pipeline.py</pre>

    結果は次のようになります。

    <pre>INFO:absl:Component CsvExampleGen depends on [].
    INFO:absl:Component CsvExampleGen is scheduled.
    INFO:absl:Component CsvExampleGen is running.
    INFO:absl:Running driver for CsvExampleGen
    INFO:absl:MetadataStore with DB connection initialized
    INFO:absl:Running executor for CsvExampleGen
    INFO:absl:Generating examples.
    INFO:absl:Using 1 process(es) for Beam pipeline execution.
    INFO:absl:Processing input csv data ./data/* to TFExample.
    WARNING:root:Couldn't find python-snappy so the implementation of _TFRecordUtil._masked_crc32c is not as fast as it could be.
    INFO:absl:Examples generated.
    INFO:absl:Running publisher for CsvExampleGen
    INFO:absl:MetadataStore with DB connection initialized
    INFO:absl:Component CsvExampleGen is finished.
    </pre>

6. 続けて、繰り返しながらパイプラインにコンポーネントを追加します。
