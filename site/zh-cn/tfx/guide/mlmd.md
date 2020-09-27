# ML Metadata

[ML Metadata (MLMD)](https://github.com/google/ml-metadata) 库可用于记录和检索与 ML 开发者和数据科学家工作流相关联的元数据。MLMD 是 [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx) 不可或缺的一部分，但它采用可以独立使用的设计。作为更广泛的 TFX 平台的一部分，大多数用户仅在检查流水线组件的结果时才会与 MLMD 交互，例如在笔记本或 TensorBoard 中。

下图给出了构成 MLMD 的各个组件。存储后端是可插拔的并且支持扩展。MLMD 为 SQLite（可以驻留在内存中和磁盘上的数据库）和 MySQL 提供了现成的参考实现。MetadataStore 提供了 API，可用于向/从存储后端记录/检索元数据。MLMD 可以注册：

- 有关通过流水线的组件/步骤生成的工件的元数据
- 有关这些组件/步骤执行的元数据
- 有关流水线和相关沿袭信息的元数据

下面将详细解释这些概念。

![ML Metadata Overview](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/guide/images/mlmd_overview.png?raw=true)

## MLMD 实现的功能

跟踪 ML 工作流中所有组件/步骤的输入和输出及其沿袭信息，可以使 ML 平台实现多个重要功能。以下列表提供了一些主要优点的简单概述。

- **列出特定类型的所有工件**。示例：所有经过训练的模型。
- **加载两个相同类型的工件进行比较**。示例：比较两个实验的结果。
- **显示所有相关执行及其上下文的输入和输出工件的 DAG**。示例：可视化实验的工作流，用于调试和发现。
- **遍历所有事件以查看工件是如何创建的**。示例：查看哪些数据进入模型；强制执行数据保留计划。
- **识别使用给定工件创建的所有工件**。示例：查看根据特定数据集训练的所有模型；基于不良数据标记模型。
- **确定以前是否在相同的输入上运行过某个执行**。示例：确定某个组件/步骤是否已经完成相同的工作，并且可以重用之前的输出。
- **记录和查询工作流运行的上下文**。示例：跟踪用于工作流运行的所有者和变更列表；按实验对沿袭信息进行分组；按项目管理工件。

## 元数据存储后端和存储连接配置

在设置数据存储之前，您需要设置导入。

```python
from ml_metadata import metadata_store
from ml_metadata.proto import metadata_store_pb2
```

MetadataStore 对象接收与使用的存储后端相对应的连接配置。

- **假数据库**为快速实验和本地运行提供了一个内存中数据库（使用 SQLite）。销毁存储对象时，将删除此数据库。

```python
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

## Metadata Store

### 概念

Metadata Store 使用以下数据模型从存储后端记录和检索元数据。

- `ArtifactType` 描述了工件的类型及其存储在 Metadata Store 中的属性。这些类型可使用代码在 Metadata Store 中即时注册，也可通过序列化格式加载到存储中。注册某种类型后，此类型的定义在存储的整个生命周期内都可用。
- `Artifact` 描述了 `ArtifactType` 的特定实例及其写入 Metadata Store 的属性。
- `ExecutionType` 描述了工作流中组件或步骤的类型及其运行时参数。
- `Execution` 是 ML 工作流中的组件运行或步骤以及运行时参数的记录。可将 Execution 视为 `ExecutionType` 的一个实例。每当开发者运行 ML 流水线或步骤时，都会记录每个步骤的执行情况。
- `Event` 是 `Artifact` 与 `Executions` 之间关系的记录。当 `Execution` 发生时，`Event` 会记录 `Execution` 使用的每个 Artifact，以及产生的每个 `Artifact`。这些记录允许在整个工作流中跟踪起源。通过查看所有 Event，MLMD 可以了解发生了哪些 Execution、创建了哪些 Artifact 作为结果，并且可以从任意 `Artifact` 递归返回到它所有的上游输入。
- `ContextType` 描述了工作流中 `Artifacts` 和 `Executions` 的概念组类型及其结构属性。例如：项目、流水线运行、实验、所有者。
- `Context` 是 `ContextType` 的实例。它会捕获组内的共享信息。例如：项目名称、变更列表提交 ID、实验注解。它在 `ContextType` 中具有用户定义的唯一名称。
- `Attribution` 是 Artifact 与 Context 之间关系的记录。
- `Association` 是 Execution 与 Context 之间关系的记录。

### 使用 ML Metadata 跟踪 ML 工作流

下图描述了如何使用低级 ML 元数据 API 来跟踪训练任务的执行，随后是代码示例。请注意，本部分中的代码显示了将由 ML 平台开发者（而不是直接由开发者）使用 ML Metadata API 将其平台与 ML Metadata 集成。此外，我们将提供更高级别的 Python API，供数据科学家在笔记本环境下记录他们的实验元数据。

![ML Metadata Example Flow](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/guide/images/mlmd_flow.png?raw=true)

1. 在可以记录执行前，必须先注册 ArtifactType。

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
```

1. 在可以记录执行前，必须为 ML 工作流中的所有步骤注册 ExecutionType。

```python
# Create ExecutionType, e.g., Trainer
trainer_type = metadata_store_pb2.ExecutionType()
trainer_type.name = "Trainer"
trainer_type.properties["state"] = metadata_store_pb2.STRING
trainer_type_id = store.put_execution_type(trainer_type)
```

1. 注册类型后，我们随即创建一个数据集工件。

```python
# Declare input artifact of type DataSet
data_artifact = metadata_store_pb2.Artifact()
data_artifact.uri = 'path/to/data'
data_artifact.properties["day"].int_value = 1
data_artifact.properties["split"].string_value = 'train'
data_artifact.type_id = data_type_id
data_artifact_id = store.put_artifacts([data_artifact])[0]
```

1. 创建数据集工件后，我们可以为 Trainer 运行创建 Execution。

```python
# Register the Execution of a Trainer run
trainer_run = metadata_store_pb2.Execution()
trainer_run.type_id = trainer_type_id
trainer_run.properties["state"].string_value = "RUNNING"
run_id = store.put_executions([trainer_run])[0]
```

1. 声明输入事件并读取数据。

```python
# Declare the input event
input_event = metadata_store_pb2.Event()
input_event.artifact_id = data_artifact_id
input_event.execution_id = run_id
input_event.type = metadata_store_pb2.Event.DECLARED_INPUT

# Submit input event to the Metadata Store
store.put_events([input_event])
```

1. 现在已经读取了输入，我们声明输出工件。

```python
# Declare output artifact of type SavedModel
model_artifact = metadata_store_pb2.Artifact()
model_artifact.uri = 'path/to/model/file'
model_artifact.properties["version"].int_value = 1
model_artifact.properties["name"].string_value = 'MNIST-v1'
model_artifact.type_id = model_type_id
model_artifact_id = store.put_artifacts([model_artifact])[0]
```

1. 创建模型工件后，我们可以记录输出事件。

```python
# Declare the output event
output_event = metadata_store_pb2.Event()
output_event.artifact_id = model_artifact_id
output_event.execution_id = run_id
output_event.type = metadata_store_pb2.Event.DECLARED_OUTPUT

# Submit output event to the Metadata Store
store.put_events([output_event])
```

1. 现在已记录所有内容，可以将 Execution 标记为已完成。

```python
trainer_run.id = run_id
trainer_run.properties["state"].string_value = "COMPLETED"
store.put_executions([trainer_run])
```

1. 随后，可将工件和执行分组到一个 Context（例如，实验）中。

```python
# Similarly, create a ContextType, e.g., Experiment with a `note` property
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
experiment_id = store.put_contexts([my_experiment])[0]

attribution = metadata_store_pb2.Attribution()
attribution.artifact_id = model_artifact_id
attribution.context_id = experiment_id

association = metadata_store_pb2.Association()
association.execution_id = run_id
association.context_id = experiment_id

store.put_attributions_and_associations([attribution], [association])
```

### 对于远程 grpc 服务器

1. 使用以下代码启动服务器

```bash
bazel run -c opt --define grpc_no_ares=true  //ml_metadata/metadata_store:metadata_store_server
```

1. 创建客户端存根并在 Python 中使用它

```python
from grpc import insecure_channel
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2
from ml_metadata.proto import metadata_store_service_pb2_grpc
channel = insecure_channel('localhost:8080')
stub = metadata_store_service_pb2_grpc.MetadataStoreServiceStub(channel)
```

1. 将 MLMD 与 RPC 调用一起使用

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
