# ML メタデータ

[ML メタデータ（MLMD）](https://github.com/google/ml-metadata)は、ML 開発者とデータサイエンティストのワークフローに関連付けられたメタデータの記録と取得を行うライブラリです。MLMD は [TensorFlow Extended（TFX）](https://www.tensorflow.org/tfx)の基本構成要素ですが、独立して使用できるように設計されています。

本番 ML パイプラインを実行するたびに、さまざまなパイプラインコンポーネント、その実行（トレーニングランなど）、および結果のアーティファクト（トレーニング済みのモデルなど）に関する情報が含まれたメタデータが生成されます。予期しないパイプラインの動作やエラーが発生した場合には、このメタデータを使用して、パイプラインコンポーネントの系統を分析し、問題をデバッグすることができます。このメタデータは、ソフトウェア開発におけるログ記録に相当すると考えても良いでしょう。

MLMD によって、相互に接続された ML パイプラインの各部位を個別にではなく全体として理解し、分析することができます。また、次のような ML パイプラインに関する疑問に答える上でも役立ちます。

- どのデータセットでモデルはトレーニングされたのか
- モデルのトレーニングにはどのハイパーパラメータが使用されたのか
- どのパイプラインランでモデルが作成されたのか
- どのトレーニングランでこのモデルが生じたのか
- このモデルはどのバージョンの TensorFlow で作成されたのか
- 失敗したモデルがプッシュされたのはいつか

## メタデータストア

MLMD は次の種類のメタデータを**メタデータストア**と呼ばれるデータベースに登録します。

1. ML パイプラインのコンポーネント/ステップで生成されたアーティファクトに関するメタデータ
2. これらのコンポーネント/ステップの実行に関するメタデータ
3. パイプラインと関連する系統の情報に関するメタデータ

メタデータストアには、ストレージバックエンドへのメタデータの記録とそこからの取得を行うための API があります。ストレージバックエンドは接続可能であり、拡張が可能です。MLMD はすぐに利用できる SQLite（インメモリとディスクをサポート）と MySQL のリファレンス実装を提供しています。

以下の図は、MLMD の一部を構成するさまざまなコンポーネントを大まかに示しています。

![ML メタデータの概要](images/mlmd_overview.png)

### メタデータのストレージバックエンドとストア接続構成

`MetadataStore` オブジェクトは、使用しているストレージバックエンドに対応する接続構成を受信します。

- **Fake Database** は、高速実験とローカルランを行うためのインメモリ DB（SQLite を使用）を提供します。このデータベースはストアオブジェクトが破棄されると削除されます。

```python
import ml_metadata as mlmd
from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2

connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.fake_database.SetInParent() # Sets an empty fake database proto.
store = metadata_store.MetadataStore(connection_config)
```

- **SQLite** はディスクからファイルを読み取り、ディスクに書き込みます。

```python
connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.sqlite.filename_uri = '...'
connection_config.sqlite.connection_mode = 3 # READWRITE_OPENCREATE
store = metadata_store.MetadataStore(connection_config)
```

- **MySQL** は MySQL サーバーに接続します。

```python
connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.mysql.host = '...'
connection_config.mysql.port = '...'
connection_config.mysql.database = '...'
connection_config.mysql.user = '...'
connection_config.mysql.password = '...'
store = metadata_store.MetadataStore(connection_config)
```

同様に、MySQL インスタンスを Google CloudSQL（[クイックスタート](https://cloud.google.com/sql/docs/mysql/quickstart)、[接続の概要](https://cloud.google.com/sql/docs/mysql/connect-overview)）で使用する場合、該当する場合は SSL オプションを使用することも可能です。

```python
connection_config.mysql.ssl_options.key = '...'
connection_config.mysql.ssl_options.cert = '...'
connection_config.mysql.ssl_options.ca = '...'
connection_config.mysql.ssl_options.capath = '...'
connection_config.mysql.ssl_options.cipher = '...'
connection_config.mysql.ssl_options.verify_server_cert = '...'
store = metadata_store.MetadataStore(connection_config)
```

## データモデル

メタデータストアは次のデータモデルを使用して、ストレージバックエンドにメタデータを記録し、そこから取得します。

- `ArtifactType` は、メタデータストアに保存されているアーティファクトの種類とそのプロパティを説明します。これらの種類はオンザフライ方式でコードのメタデータストアに登録するか、シリアル化形式からストアに読み込むことができます。種類を登録したら、その定義がストアの寿命期間、利用できるようになります。
- `Artifact` は `ArtifactType` の特定のインスタンスと、メタデータストアに書き込まれたプロパティを説明します。
- `ExecutionType` は、コンポーネントまたはワークフローのステップの種類とランタイムパラメーターを説明します。
- `Execution` は ML ワークフローのコンポーネントランまたはステップとランタイムパラメーターのレコードです。実行は、`ExecutionType` のインスタントとして考えられます。実行は、ML パイプラインまたはステップを実行すると記録されます。
- `Event` はアーティファクトと実行の関係のレコードです。実行が発生すると、イベントは実行で使用されたすべてのアーティファクトと生成されたすべてのアーティファクトを記録します。これらのレコードによって、ワークフロー全体で系統を追跡することができます。すべてのイベントを見ることで、MLMD はどの実行が発生し、どのアーティファクトが結果的に作成されたのかを把握します。その上で、アーティファクトから上流のすべての入力にさかのぼることができます。
- `ContextType` は、ワークフロー内のアーティファクトと実行の概念的なグループの種類と、その構造上のプロパティを説明します。たとえば、プロジェクト、パイプラインラン、実験、所有者などです。
- `Context` は `ContextType` のインスタンスです。グループ内の共有情報をキャプチャします。たとえば、プロジェクト名、変更リストのコミット ID、実験の注釈などです。`ContextType` にはユーザー定義の一意の名前があります。
- `Attribution` は、アーティファクトとコンテキストの関係のレコードです。
- `Association` は、実行とコンテキストの関係のレコードです。

## MLMD の機能

ML ワークフローとその系統のすべてのコンポーネント/ステップの入力と出力を追跡すると、ML プラットフォームでさまざまな重要な機能が有効になります。主なメリットの一部を以下のリストに示します。

- **特定の種類のアーティファクトをすべてリストする。** 例: トレーニング済みのすべてのモデル
- **同じ種類のアーティファクトを 2 つ読み込んで比較する。** 例: 2 つの実験の結果の比較
- **コンテキストの関連するすべての実行とその入力と出力アーティファクトの DAG を示す。** 例: 実験のワークフローを視覚化して、デバッグや検出を行う
- **すべてのイベントをさかのぼってアーティファクトがどのように作成されたのかを確認する。** 例: どのデータがモデルに取り込まれたのか、データ保持計画を適用する
- **ある特定のアーティファクトを使って作成されたすべてのアーティファクトを識別する。** 例: 特定のデータセットからトレーニングされたすべてのモデルを確認する、不良データに基づくモデルをマークする
- **実行が前と同じ入力で実行されたかどうかを判断する。** 例: コンポーネント/ステップが同じ作業をすでに完了しており、前の出力を再利用できるかどうかを判断する
- **ワークフローランのコンテキストを記録してクエリする。** 例: ワークフローランにしよされた所有者と変更リストを追跡する、実験別に系統をグループ化する、プロジェクト別にアーティファクトを管理する
- **プロパティと 1 ホップ隣接ノードでの宣言的ノードフィルタ機能。** 例: 型のアーティファクトをパイプラインコンテキスト下で探します。特定のプロパティ値が範囲内にある型付きのアーティファクトを返します。同じ入力でコンテキスト内の前の実行を検索します。

MLMD API とメタデータストアを使用して系統情報を取得する方法を示す例については、[MLMD チュートリアル](https://www.tensorflow.org/tfx/tutorials/mlmd/mlmd_tutorial)をご覧ください。

### ML メタデータを ML ワークフローに組み込む

MLMD をシステムに組み込みたいと考えているプラットフォーム開発者であれば、以下の低レベルの MLMD API を使ってトレーニングタスクの実行を追跡するワークフローの例を利用できます。また、ノートブック環境でより高いレベルの Python API を使って実験的なメタデータを記録することもできます。

![ML メタデータのサンプルフロー](images/mlmd_flow.png)

1. アーティファクトの種類を登録する

```python
# Create ArtifactTypes, e.g., Data and Model
data_type = metadata_store_pb2.ArtifactType()
data_type.name = "DataSet"
data_type.properties["day"] = metadata_store_pb2.INT
data_type.properties["split"] = metadata_store_pb2.STRING
data_type_id = store.put_artifact_type(data_type)

model_type = metadata_store_pb2.ArtifactType()
model_type.name = "SavedModel"
model_type.properties["version"] = metadata_store_pb2.INT
model_type.properties["name"] = metadata_store_pb2.STRING
model_type_id = store.put_artifact_type(model_type)

# Query all registered Artifact types.
artifact_types = store.get_artifact_types()
```

1. ML ワークフローのすべてのステップの実行タイプを登録する

```python
# Create an ExecutionType, e.g., Trainer
trainer_type = metadata_store_pb2.ExecutionType()
trainer_type.name = "Trainer"
trainer_type.properties["state"] = metadata_store_pb2.STRING
trainer_type_id = store.put_execution_type(trainer_type)

# Query a registered Execution type with the returned id
[registered_type] = store.get_execution_types_by_id([trainer_type_id])
```

1. DataSet ArtifactType のアーティファクトを作成する

```python
# Create an input artifact of type DataSet
data_artifact = metadata_store_pb2.Artifact()
data_artifact.uri = 'path/to/data'
data_artifact.properties["day"].int_value = 1
data_artifact.properties["split"].string_value = 'train'
data_artifact.type_id = data_type_id
[data_artifact_id] = store.put_artifacts([data_artifact])

# Query all registered Artifacts
artifacts = store.get_artifacts()

# Plus, there are many ways to query the same Artifact
[stored_data_artifact] = store.get_artifacts_by_id([data_artifact_id])
artifacts_with_uri = store.get_artifacts_by_uri(data_artifact.uri)
artifacts_with_conditions = store.get_artifacts(
      list_options=mlmd.ListOptions(
          filter_query='uri LIKE "%/data" AND properties.day.int_value > 0'))
```

1. Trainer ランの実行を作成する

```python
# Register the Execution of a Trainer run
trainer_run = metadata_store_pb2.Execution()
trainer_run.type_id = trainer_type_id
trainer_run.properties["state"].string_value = "RUNNING"
[run_id] = store.put_executions([trainer_run])

# Query all registered Execution
executions = store.get_executions_by_id([run_id])
# Similarly, the same execution can be queried with conditions.
executions_with_conditions = store.get_executions(
    list_options = mlmd.ListOptions(
        filter_query='type = "Trainer" AND properties.state.string_value IS NOT NULL'))
```

1. 入力イベントを定義し、データを読み取る

```python
# Define the input event
input_event = metadata_store_pb2.Event()
input_event.artifact_id = data_artifact_id
input_event.execution_id = run_id
input_event.type = metadata_store_pb2.Event.DECLARED_INPUT

# Record the input event in the metadata store
store.put_events([input_event])
```

1. 出力アーティファクトを宣言する

```python
# Declare the output artifact of type SavedModel
model_artifact = metadata_store_pb2.Artifact()
model_artifact.uri = 'path/to/model/file'
model_artifact.properties["version"].int_value = 1
model_artifact.properties["name"].string_value = 'MNIST-v1'
model_artifact.type_id = model_type_id
[model_artifact_id] = store.put_artifacts([model_artifact])
```

1. 出力イベントを記録する

```python
# Declare the output event
output_event = metadata_store_pb2.Event()
output_event.artifact_id = model_artifact_id
output_event.execution_id = run_id
output_event.type = metadata_store_pb2.Event.DECLARED_OUTPUT

# Submit output event to the Metadata Store
store.put_events([output_event])
```

1. 実行を完了としてマークする

```python
trainer_run.id = run_id
trainer_run.properties["state"].string_value = "COMPLETED"
store.put_executions([trainer_run])
```

1. attribution と assertion アーティファクトを使って、アーティファクトと実行をコンテキストの下にグループ化する

```python
# Create a ContextType, e.g., Experiment with a note property
experiment_type = metadata_store_pb2.ContextType()
experiment_type.name = "Experiment"
experiment_type.properties["note"] = metadata_store_pb2.STRING
experiment_type_id = store.put_context_type(experiment_type)

# Group the model and the trainer run to an experiment.
my_experiment = metadata_store_pb2.Context()
my_experiment.type_id = experiment_type_id
# Give the experiment a name
my_experiment.name = "exp1"
my_experiment.properties["note"].string_value = "My first experiment."
[experiment_id] = store.put_contexts([my_experiment])

attribution = metadata_store_pb2.Attribution()
attribution.artifact_id = model_artifact_id
attribution.context_id = experiment_id

association = metadata_store_pb2.Association()
association.execution_id = run_id
association.context_id = experiment_id

store.put_attributions_and_associations([attribution], [association])

# Query the Artifacts and Executions that are linked to the Context.
experiment_artifacts = store.get_artifacts_by_context(experiment_id)
experiment_executions = store.get_executions_by_context(experiment_id)

# You can also use neighborhood queries to fetch these artifacts and executions
# with conditions.
experiment_artifacts_with_conditions = store.get_artifacts(
    list_options = mlmd.ListOptions(
        filter_query=('contexts_a.type = "Experiment" AND contexts_a.name = "exp1"')))
experiment_executions_with_conditions = store.get_executions(
    list_options = mlmd.ListOptions(
        filter_query=('contexts_a.id = {}'.format(experiment_id))))
```

## MLMD をリモート gRPC サーバーと使用する

以下に示すように、MLMD をリモート gRPC サーバーと使用することができます。

- サーバーを起動します。

```bash
bazel run -c opt --define grpc_no_ares=true  //ml_metadata/metadata_store:metadata_store_server
```

デフォルトでは、サーバーはリクエストごとにフェイクのインメモリ db を使用し、呼び出し間でメタデータを永続化しません。また、MySQL インスタンスまたは Sqlite ファイルを使用するように MLMD の `MetadataStoreServerConfig` で構成することもできます。この構成はテキスト形式の protobuf ファイルに保存し、`--metadata_store_server_config_file=path_to_the_config_file` を使ってバイナリーに渡すことができます。

テキスト protobuf 形式の `MetadataStoreServerConfig` ファイルの例：

```textpb
connection_config {
  sqlite {
    filename_uri: '/tmp/test_db'
    connection_mode: READWRITE_OPENCREATE
  }
}
```

- クライアントスタブを作成して Python で使用します。

```python
from grpc import insecure_channel
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2
from ml_metadata.proto import metadata_store_service_pb2_grpc

channel = insecure_channel('localhost:8080')
stub = metadata_store_service_pb2_grpc.MetadataStoreServiceStub(channel)
```

- MLMD を RPC 呼び出しで使用します。

```python
# Create ArtifactTypes, e.g., Data and Model
data_type = metadata_store_pb2.ArtifactType()
data_type.name = "DataSet"
data_type.properties["day"] = metadata_store_pb2.INT
data_type.properties["split"] = metadata_store_pb2.STRING

request = metadata_store_service_pb2.PutArtifactTypeRequest()
request.all_fields_match = True
request.artifact_type.CopyFrom(data_type)
stub.PutArtifactType(request)

model_type = metadata_store_pb2.ArtifactType()
model_type.name = "SavedModel"
model_type.properties["version"] = metadata_store_pb2.INT
model_type.properties["name"] = metadata_store_pb2.STRING

request.artifact_type.CopyFrom(model_type)
stub.PutArtifactType(request)
```

## リソース

MLMD ライブラリには、ML パイプラインですぐに使用できる高位 API があります。詳細は、[MLMD API ドキュメント](https://www.tensorflow.org/tfx/ml_metadata/api_docs/python/mlmd)をご覧ください。

プロパティと 1 ホップ隣接ノードでの MLMD 宣言的ノードフィルタ機能の使用方法については、[MLMD Declarative Nodes Filtering](https://github.com/google/ml-metadata/blob/v1.2.0/ml_metadata/proto/metadata_store.proto#L708-L786) をご覧ください。

また、MLMD を使ってパイプラインコンポーネントの系統を追跡する方法については、[MLMD チュートリアル](https://www.tensorflow.org/tfx/tutorials/mlmd/mlmd_tutorial)をご覧ください。

MLMD には、リリース間でスキーマとデータ移行を処理するためのユーティリティが用意されています。詳細は、MLMD の[ガイド](https://github.com/google/ml-metadata/blob/master/g3doc/get_started.md#upgrade-the-mlmd-library)をご覧ください。
