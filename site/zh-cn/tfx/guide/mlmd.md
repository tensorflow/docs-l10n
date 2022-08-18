# ML Metadata

[ML Metadata (MLMD)](https://github.com/google/ml-metadata) 是一个用于记录和检索与 ML 开发者和数据科学家工作流相关联的元数据的库。MLMD 是 [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx) 的组成部分，但其设计使其可以独立使用。

正式环境 ML 流水线的每一次运行都会生成元数据，其中包含有关各种流水线组件、它们的执行（例如，训练运行），以及结果工件（例如，经过训练的模型）。如果发生意外的流水线行为或错误，则可以利用此元数据分析流水线组件的沿袭和调试问题。可以将此元数据视为软件开发中的日志记录。

MLMD 可以帮助您理解和分析 ML 流水线的所有互连部分，而不是孤立地对其进行分析，并且可以帮助您回答有关 ML 流水线的问题，例如：

- 模型是在哪个数据集上训练的？
- 用来训练模型的超参数是什么？
- 哪个流水线运行创建了模型？
- 哪个训练运行产生了该模型？
- 哪个版本的 TensorFlow 创建了该模型？
- 何时推送失败的模型？

## Metadata Store

MLMD 在名为 **Metadata Store** 的数据库中注册以下类型的元数据。

1. 有关通过 ML 流水线的组件/步骤生成的工件的元数据
2. 有关这些组件/步骤执行的元数据
3. 有关流水线和相关沿袭信息的元数据

Metadata Store 提供了用于向存储后端记录和从存储后端检索元数据的 API。存储后端可插拔且可扩展。MLMD 为 SQLite（支持内存和磁盘）和 MySQL 提供了开箱即用的参考实现。

此图显示了 MLMD 各个组件的高级概览。

![ML Metadata Overview](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/guide/images/mlmd_overview.png?raw=true)

### 元数据存储后端和库连接配置

MetadataStore 对象接收与使用的存储后端相对应的连接配置。

- **假数据库**为快速实验和本地运行提供了一个内存中数据库（使用 SQLite）。销毁存储对象时，将删除此数据库。

```python
import ml_metadata as mlmd
from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2

connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.fake_database.SetInParent() # Sets an empty fake database proto.
store = metadata_store.MetadataStore(connection_config)
```

- **SQLite** 从磁盘读取和写入文件。

```python
connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.sqlite.filename_uri = '...'
connection_config.sqlite.connection_mode = 3 # READWRITE_OPENCREATE
store = metadata_store.MetadataStore(connection_config)
```

- **MySQL** 连接到 MySQL 服务器。

```python
connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.mysql.host = '...'
connection_config.mysql.port = '...'
connection_config.mysql.database = '...'
connection_config.mysql.user = '...'
connection_config.mysql.password = '...'
store = metadata_store.MetadataStore(connection_config)
```

同样，当将 MySQL 实例与 Google CloudSQL（[快速入门](https://cloud.google.com/sql/docs/mysql/quickstart)、[连接概览](https://cloud.google.com/sql/docs/mysql/connect-overview)）一起使用时，如果适用，也可以使用 SSL 选项。

```python
connection_config.mysql.ssl_options.key = '...'
connection_config.mysql.ssl_options.cert = '...'
connection_config.mysql.ssl_options.ca = '...'
connection_config.mysql.ssl_options.capath = '...'
connection_config.mysql.ssl_options.cipher = '...'
connection_config.mysql.ssl_options.verify_server_cert = '...'
store = metadata_store.MetadataStore(connection_config)
```

## 数据模型

Metadata Store 使用以下数据模型从存储后端记录和检索元数据。

- `ArtifactType` 描述了工件的类型及其存储在 Metadata Store 中的属性。您可以在 Metadata Store 中用代码即时注册这些类型，也可通过序列化格式将它们加载到存储中。注册某种类型后，此类型的定义在存储的整个生命周期内都可用。
- `Artifact` 描述了 `ArtifactType` 的特定实例及其写入 Metadata Store 的属性。
- `ExecutionType` 描述了组件的类型或工作流中的步骤，及其运行时参数。
- `Execution` 是 ML 工作流中的组件运行或步骤，以及运行时参数的记录。可将 Execution 视为 `ExecutionType` 的实例。每次运行 ML 流水线或步骤时，都会记录执行。
- `Event` 是工件与执行之间关系的记录。当执行发生时，事件会记录执行使用的每个工件，以及产生的每个工件。这些记录允许在整个工作流中进行沿袭跟踪。通过查看所有事件，MLMD 可以了解发生了哪些执行，以及由此产生了哪些工件。然后 MLMD 可以从任意工件递归回其所有上游输入。
- `ContextType` 描述了工作流中工件和执行的概念组类型及其结构属性。例如：项目、流水线运行、实验、所有者等。
- `Context` 是 `ContextType` 的实例。它会捕获组内的共享信息。例如：项目名称、变更列表提交 ID、实验注解等。它在其 `ContextType` 中具有用户定义的唯一名称。
- `Attribution` 是工件与上下文之间关系的记录。
- `Association` 是执行与上下文之间关系的记录。

## MLMD 功能

跟踪 ML 工作流中所有组件/步骤的输入和输出及其沿袭信息，可以使 ML 平台实现多个重要功能。以下列表提供了一些主要优点的简单概述。

- **列出特定类型的所有工件**。示例：所有经过训练的模型。
- **加载两个相同类型的工件进行比较**。示例：比较两个实验的结果。
- **显示所有相关执行及其上下文的输入和输出工件的 DAG**。示例：可视化实验的工作流，用于调试和发现。
- **遍历所有事件以查看工件是如何创建的**。示例：查看哪些数据进入模型；强制执行数据保留计划。
- **识别使用给定工件创建的所有工件**。示例：查看根据特定数据集训练的所有模型；基于不良数据标记模型。
- **确定以前是否在相同的输入上运行过某个执行**。示例：确定某个组件/步骤是否已经完成相同的工作，并且可以重用之前的输出。
- **记录和查询工作流运行的上下文**。示例：跟踪用于工作流运行的所有者和变更列表；按实验对沿袭信息进行分组；按项目管理工件。
- **属性和 1 跳近邻节点上的声明性节点过滤功能。**示例：在某个流水线上下文中查找某个类型的工件；返回给定属性的值在一个范围内的类型化工件；在具有相同输入的上下文中查找先前的执行。

请参阅 [MLMD 教程](https://www.tensorflow.org/tfx/tutorials/mlmd/mlmd_tutorial)中的示例，该示例展示了如何使用 MLMD API 和 Metadata Store 检索沿袭信息。

### 将 ML Metadata 集成到 ML 工作流中

如果您是平台开发者，且有兴趣将 MLMD 集成到您的系统，请使用下面的示例工作流来使用低级 MLMD API 跟踪训练任务的执行。您还可以在笔记本环境中使用更高级别的 Python API 记录实验元数据。

![ML Metadata Example Flow](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/guide/images/mlmd_flow.png?raw=true)

1. 注册工件类型

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

1. 为 ML 工作流中的所有步骤注册执行类型

```python
# Create an ExecutionType, e.g., Trainer
trainer_type = metadata_store_pb2.ExecutionType()
trainer_type.name = "Trainer"
trainer_type.properties["state"] = metadata_store_pb2.STRING
trainer_type_id = store.put_execution_type(trainer_type)

# Query a registered Execution type with the returned id
[registered_type] = store.get_execution_types_by_id([trainer_type_id])
```

1. 创建 DataSet ArtifactType 工件

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

1. 创建 Trainer 运行的执行

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

1. 定义输入事件并读取数据

```python
# Define the input event
input_event = metadata_store_pb2.Event()
input_event.artifact_id = data_artifact_id
input_event.execution_id = run_id
input_event.type = metadata_store_pb2.Event.DECLARED_INPUT

# Record the input event in the metadata store
store.put_events([input_event])
```

1. 声明输出工件

```python
# Declare the output artifact of type SavedModel
model_artifact = metadata_store_pb2.Artifact()
model_artifact.uri = 'path/to/model/file'
model_artifact.properties["version"].int_value = 1
model_artifact.properties["name"].string_value = 'MNIST-v1'
model_artifact.type_id = model_type_id
[model_artifact_id] = store.put_artifacts([model_artifact])
```

1. 记录输出事件

```python
# Declare the output event
output_event = metadata_store_pb2.Event()
output_event.artifact_id = model_artifact_id
output_event.execution_id = run_id
output_event.type = metadata_store_pb2.Event.DECLARED_OUTPUT

# Submit output event to the Metadata Store
store.put_events([output_event])
```

1. 将执行标记为已完成

```python
trainer_run.id = run_id
trainer_run.properties["state"].string_value = "COMPLETED"
store.put_executions([trainer_run])
```

1. 使用归因和断言工件在上下文中对工件和执行进行分组

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

## 在远程 gRPC 服务器上使用 MLMD

您可以在远程 gRPC 服务器上使用 MLMD，如下所示：

- 启动服务器

```bash
bazel run -c opt --define grpc_no_ares=true  //ml_metadata/metadata_store:metadata_store_server
```

默认情况下，服务器会对每个请求使用伪内存数据库，并且不会在调用之间保留元数据。也可以使用 MLMD `MetadataStoreServerConfig` 对其进行配置，以使用 Sqlite 文件或 MySQL 实例。可以将配置存储在文本 protobuf 文件中，并通过 `--metadata_store_server_config_file=path_to_the_config_file` 将其传递给二进制文件。

文本 protobuf 格式的示例 `MetadataStoreServerConfig` 文件如下：

```textpb
connection_config {
  sqlite {
    filename_uri: '/tmp/test_db'
    connection_mode: READWRITE_OPENCREATE
  }
}
```

- 创建客户端存根并在 Python 中使用它

```python
from grpc import insecure_channel
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2
from ml_metadata.proto import metadata_store_service_pb2_grpc

channel = insecure_channel('localhost:8080')
stub = metadata_store_service_pb2_grpc.MetadataStoreServiceStub(channel)
```

- 将 MLMD 与 RPC 调用一起使用

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

## 资源

MLMD 库有一个高级 API，您可以很容易地将其与 ML 流水线一起使用。请参阅 [MLMD API 文档](https://www.tensorflow.org/tfx/ml_metadata/api_docs/python/mlmd)，了解更多详细信息。

查看 [MLMD 声明性节点过滤](https://github.com/google/ml-metadata/blob/v1.2.0/ml_metadata/proto/metadata_store.proto#L708-L786)，了解如何在属性和 1 近邻节点上使用 MLMD 声明性节点过滤功能。

另外，请查看 [MLMD 教程](https://www.tensorflow.org/tfx/tutorials/mlmd/mlmd_tutorial)，了解如何使用 MLMD 跟踪流水线组件的沿袭。

MLMD 提供了实用程序来处理跨版本的架构和数据迁移。有关详细信息，请参阅 MLMD [指南](https://github.com/google/ml-metadata/blob/master/g3doc/get_started.md#upgrade-the-mlmd-library)。
