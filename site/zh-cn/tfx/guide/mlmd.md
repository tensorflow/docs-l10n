# ML Metadata

[ML Metadata (MLMD)](https://github.com/google/ml-metadata) is a library for recording and retrieving metadata associated with ML developer and data scientist workflows. MLMD is an integral part of [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx), but is designed so that it can be used independently.

Every run of a production ML pipeline generates metadata containing information about the various pipeline components, their executions (e.g. training runs), and resulting artifacts (e.g. trained models). In the event of unexpected pipeline behavior or errors, this metadata can be leveraged to analyze the lineage of pipeline components and debug issues. Think of this metadata as the equivalent of logging in software development.

下面将详细解释这些概念。

- Which dataset did the model train on?
- What were the hyperparameters used to train the model?
- Which pipeline run created the model?
- Which training run led to this model?
- Which version of TensorFlow created this model?
- When was the failed model pushed?

## Metadata store

MLMD registers the following types of metadata in a database called the **Metadata Store**.

1. Metadata about the artifacts generated through the components/steps of your ML pipelines
2. Metadata about the executions of these components/steps
3. Metadata about pipelines and associated lineage information

The Metadata Store provides APIs to record and retrieve metadata to and from the storage backend. The storage backend is pluggable and can be extended. MLMD provides reference implementations for SQLite (which supports in-memory and disk) and MySQL out of the box.

在设置数据存储之前，您需要设置导入。

![ML Metadata Overview](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/guide/images/mlmd_overview.png?raw=true)

### 概念

MetadataStore 对象接收与使用的存储后端相对应的连接配置。

- **Fake Database** provides an in-memory DB (using SQLite) for fast experimentation and local runs. The database is deleted when the store object is destroyed.

```python
from ml_metadata import metadata_store
from ml_metadata.proto import metadata_store_pb2
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

下图描述了如何使用低级 ML 元数据 API 来跟踪训练任务的执行，随后是代码示例。请注意，本部分中的代码显示了将由 ML 平台开发者（而不是直接由开发者）使用 ML Metadata API 将其平台与 ML Metadata 集成。此外，我们将提供更高级别的 Python API，供数据科学家在笔记本环境下记录他们的实验元数据。

```python
connection_config.mysql.ssl_options.key = '...'
connection_config.mysql.ssl_options.cert = '...'
connection_config.mysql.ssl_options.ca = '...'
connection_config.mysql.ssl_options.capath = '...'
connection_config.mysql.ssl_options.cipher = '...'
connection_config.mysql.ssl_options.verify_server_cert = '...'
store = metadata_store.MetadataStore(connection_config)
```

## 元数据存储后端和存储连接配置

Metadata Store 使用以下数据模型从存储后端记录和检索元数据。

- `ArtifactType` describes an artifact's type and its properties that are stored in the metadata store. You can register these types on-the-fly with the metadata store in code, or you can load them in the store from a serialized format. Once you register a type, its definition is available throughout the lifetime of the store.
- An `Artifact` describes a specific instance of an `ArtifactType`, and its properties that are written to the metadata store.
- An `ExecutionType` describes a type of component or step in a workflow, and its runtime parameters.
- An `Execution` is a record of a component run or a step in an ML workflow and the runtime parameters. An execution can be thought of as an instance of an `ExecutionType`. Executions are recorded when you run an ML pipeline or step.
- An `Event` is a record of the relationship between artifacts and executions. When an execution happens, events record every artifact that was used by the execution, and every artifact that was produced. These records allow for lineage tracking throughout a workflow. By looking at all events, MLMD knows what executions happened and what artifacts were created as a result. MLMD can then recurse back from any artifact to all of its upstream inputs.
- A `ContextType` describes a type of conceptual group of artifacts and executions in a workflow, and its structural properties. For example: projects, pipeline runs, experiments, owners etc.
- A `Context` is an instance of a `ContextType`. It captures the shared information within the group. For example: project name, changelist commit id, experiment annotations etc. It has a user-defined unique name within its `ContextType`.
- An `Attribution` is a record of the relationship between artifacts and contexts.
- An `Association` is a record of the relationship between executions and contexts.

## Metadata Store

跟踪 ML 工作流中所有组件/步骤的输入和输出及其沿袭信息，可以使 ML 平台实现多个重要功能。以下列表提供了一些主要优点的简单概述。

- **列出特定类型的所有工件**。示例：所有经过训练的模型。
- **加载两个相同类型的工件进行比较**。示例：比较两个实验的结果。
- **显示所有相关执行及其上下文的输入和输出工件的 DAG**。示例：可视化实验的工作流，用于调试和发现。
- **遍历所有事件以查看工件是如何创建的**。示例：查看哪些数据进入模型；强制执行数据保留计划。
- **识别使用给定工件创建的所有工件**。示例：查看根据特定数据集训练的所有模型；基于不良数据标记模型。
- **确定以前是否在相同的输入上运行过某个执行**。示例：确定某个组件/步骤是否已经完成相同的工作，并且可以重用之前的输出。
- **记录和查询工作流运行的上下文**。示例：跟踪用于工作流运行的所有者和变更列表；按实验对沿袭信息进行分组；按项目管理工件。
- **Declarative nodes filtering capabilities on properties and 1-hop neighborhood nodes.** Examples: look for artifacts of a type and under some pipeline context; return typed artifacts where a given property’s value is within a range; find previous executions in a context with the same inputs.

See the [MLMD tutorial](https://www.tensorflow.org/tfx/tutorials/mlmd/mlmd_tutorial) for an example that shows you how to use the MLMD API and the metadata store to retrieve lineage information.

### 使用 ML Metadata 跟踪 ML 工作流

If you are a platform developer interested in integrating MLMD into your system, use the example workflow below to use the low-level MLMD APIs to track the execution of a training task. You can also use higher-level Python APIs in notebook environments to record experiment metadata.

![ML Metadata Example Flow](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/guide/images/mlmd_flow.png?raw=true)

1. 在可以记录执行前，必须为 ML 工作流中的所有步骤注册 ExecutionType。

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

1. 注册类型后，我们随即创建一个数据集工件。

```python
# Create ExecutionType, e.g., Trainer
trainer_type = metadata_store_pb2.ExecutionType()
trainer_type.name = "Trainer"
trainer_type.properties["state"] = metadata_store_pb2.STRING
trainer_type_id = store.put_execution_type(trainer_type)
```

1. 创建数据集工件后，我们可以为 Trainer 运行创建 Execution。

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

1. 声明输入事件并读取数据。

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

1. 现在已经读取了输入，我们声明输出工件。

```python
# Declare the input event
input_event = metadata_store_pb2.Event()
input_event.artifact_id = data_artifact_id
input_event.execution_id = run_id
input_event.type = metadata_store_pb2.Event.DECLARED_INPUT

# Submit input event to the Metadata Store
store.put_events([input_event])
```

1. 创建模型工件后，我们可以记录输出事件。

```python
# Declare output artifact of type SavedModel
model_artifact = metadata_store_pb2.Artifact()
model_artifact.uri = 'path/to/model/file'
model_artifact.properties["version"].int_value = 1
model_artifact.properties["name"].string_value = 'MNIST-v1'
model_artifact.type_id = model_type_id
model_artifact_id = store.put_artifacts([model_artifact])[0]
```

1. 现在已记录所有内容，可以将 Execution 标记为已完成。

```python
# Declare the output event
output_event = metadata_store_pb2.Event()
output_event.artifact_id = model_artifact_id
output_event.execution_id = run_id
output_event.type = metadata_store_pb2.Event.DECLARED_OUTPUT

# Submit output event to the Metadata Store
store.put_events([output_event])
```

1. 随后，可将工件和执行分组到一个 Context（例如，实验）中。

```python
trainer_run.id = run_id
trainer_run.properties["state"].string_value = "COMPLETED"
store.put_executions([trainer_run])
```

1. 使用以下代码启动服务器

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

## Use MLMD with a remote gRPC server

You can use MLMD with remote gRPC servers as shown below:

- Start a server

```bash
bazel run -c opt --define grpc_no_ares=true  //ml_metadata/metadata_store:metadata_store_server
```

By default, the server uses a fake in-memory db per request and does not persist the metadata across calls. It can also be configured with a MLMD `MetadataStoreServerConfig` to use SQLite files or MySQL instances. The config can be stored in a text protobuf file and passed to the binary with `--metadata_store_server_config_file=path_to_the_config_file`.

An example `MetadataStoreServerConfig` file in text protobuf format:

```textpb
connection_config {
  sqlite {
    filename_uri: '/tmp/test_db'
    connection_mode: READWRITE_OPENCREATE
  }
}
```

- Create the client stub and use it in Python

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

## Resources

The MLMD library has a high-level API that you can readily use with your ML pipelines. See the [MLMD API documentation](https://www.tensorflow.org/tfx/ml_metadata/api_docs/python/mlmd) for more details.

Check out [MLMD Declarative Nodes Filtering](https://github.com/google/ml-metadata/blob/v1.2.0/ml_metadata/proto/metadata_store.proto#L708-L786) to learn how to use MLMD declarative nodes filtering capabilities on properties and 1-hop neighborhood nodes.

Also check out the [MLMD tutorial](https://www.tensorflow.org/tfx/tutorials/mlmd/mlmd_tutorial) to learn how to use MLMD to trace the lineage of your pipeline components.

MLMD provides utilities to handle schema and data migrations across releases. See the MLMD [Guide](https://github.com/google/ml-metadata/blob/master/g3doc/get_started.md#upgrade-the-mlmd-library) for more details.
