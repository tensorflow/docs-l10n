# ML 메타데이터

[ML Metadata(MLMD)](https://github.com/google/ml-metadata)는 ML 개발자 및 데이터 과학자 워크플로와 관련된 메타데이터를 기록하고 검색하기 위한 라이브러리입니다. MLMD는 [TensorFlow Extended(TFX)](https://www.tensorflow.org/tfx)의 필수 부분이지만 독립적으로 사용할 수 있도록 설계되었습니다.

프로덕션 ML 파이프라인의 모든 실행의 결과로, 다양한 파이프라인 구성 요소, 실행(예: 학습 실행) 및 결과 아티팩트(예: 학습된 모델)에 대한 정보를 포함하는 메타데이터가 생성됩니다. 예기치 않은 파이프라인 동작이나 오류가 발생하는 경우 이 메타데이터를 활용하여 파이프라인 구성 요소의 계보를 분석하고 문제를 디버그할 수 있습니다. 이 메타데이터를 소프트웨어 개발을 기록하는 것으로 생각하면 이해가 쉽습니다.

MLMD는 ML 파이프라인의 상호 연결된 모든 부분을 개별적으로 분석하지 않고도 쉽게 이해하고 분석할 수 있게 해주며 ML 파이프라인에 관한 다음과 같은 물음에 답하는 데 도움을 줄 수 있습니다.

- 모델이 학습한 데이터세트는 무엇입니까?
- 모델 훈련에 사용된 하이퍼 파라미터는 무엇입니까?
- 모델을 생성한 파이프라인 실행은 무엇입니까?
- 이 모델로 이어진 훈련 실행은 무엇입니까?
- 이 모델을 만든 TensorFlow 버전은 무엇입니까?
- 실패한 모델은 언제 푸시되었습니까?

## 메타데이터 저장소

MLMD는 **메타데이터 저장소**라는 데이터베이스에 다음 유형의 메타데이터를 등록합니다.

1. ML 파이프라인의 구성 요소/단계를 통해 생성된 아티팩트에 대한 메타데이터
2. 이러한 구성 요소/단계의 실행에 대한 메타데이터
3. 파이프라인 및 관련 계보 정보에 대한 메타데이터

메타데이터 저장소는 저장소 백엔드에서 메타데이터를 기록하고 검색하는 API를 제공합니다. 스토리지 백엔드는 플러그 가능하고 확장할 수 있습니다. MLMD는 SQLite(인메모리 및 디스크 지원) 및 MySQL에 대한 참조 구현을 즉시 제공합니다.

이 그래픽은 MLMD의 일부인 다양한 구성 요소에 대한 높은 수준의 개요를 보여줍니다.

![ML Metadata Overview](https://github.com/tensorflow/docs-l10n/blob/master/site/ko/tfx/guide/images/mlmd_overview.png?raw=true)

### 메타데이터 스토리지 백엔드 및 저장소 연결 구성

`MetadataStore` 객체는 사용된 스토리지 백엔드에 해당하는 연결 구성을 수신합니다.

- **Fake Database**는 빠른 실험과 로컬 실행을 위해 인메모리 DB(SQLite 사용)를 제공합니다. 저장소 객체가 삭제되면 데이터베이스가 삭제됩니다.

```python
import ml_metadata as mlmd
from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2

connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.fake_database.SetInParent() # Sets an empty fake database proto.
store = metadata_store.MetadataStore(connection_config)
```

- **SQLite**는 디스크에서 파일을 읽고 작성합니다.

```python
connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.sqlite.filename_uri = '...'
connection_config.sqlite.connection_mode = 3 # READWRITE_OPENCREATE
store = metadata_store.MetadataStore(connection_config)
```

- **MySQL**은 MySQL 서버에 연결됩니다.

```python
connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.mysql.host = '...'
connection_config.mysql.port = '...'
connection_config.mysql.database = '...'
connection_config.mysql.user = '...'
connection_config.mysql.password = '...'
store = metadata_store.MetadataStore(connection_config)
```

마찬가지로, Google CloudSQL([quickstart](https://cloud.google.com/sql/docs/mysql/quickstart), [connect-overview](https://cloud.google.com/sql/docs/mysql/connect-overview))과 함께 MySQL 인스턴스를 사용하면, 해당하는 경우 SSL 옵션을 사용할 수도 있습니다.

```python
connection_config.mysql.ssl_options.key = '...'
connection_config.mysql.ssl_options.cert = '...'
connection_config.mysql.ssl_options.ca = '...'
connection_config.mysql.ssl_options.capath = '...'
connection_config.mysql.ssl_options.cipher = '...'
connection_config.mysql.ssl_options.verify_server_cert = '...'
store = metadata_store.MetadataStore(connection_config)
```

## 데이터 모델

메타데이터 저장소는 다음 데이터 모델을 사용하여 스토리지 백엔드에서 메타데이터를 기록하고 검색합니다.

- `ArtifactType`은 아티팩트 유형 및 메타데이터 저장소에 저장되는 속성을 설명합니다. 이러한 유형을 코드의 메타데이터 저장소에 즉시 등록하거나, 직렬화된 형식에서 저장소에 로드할 수 있습니다. 유형을 등록하면 저장소의 수명 기간 동안 해당 정의를 사용할 수 있습니다.
- `Artifact`는 `ArtifactType`의 특정 인스턴스와 메타데이터 저장소에 작성된 해당 속성을 설명합니다.
- `ExecutionType`은 워크플로의 구성 요소 유형 또는 단계와 해당 런타임 매개변수를 설명합니다.
- `Execution`은 ML 워크플로의 구성 요소 실행 또는 단계, 및 런타임 매개변수에 대한 기록입니다. 실행은 `ExecutionType`의 인스턴스로 생각할 수 있습니다. ML 파이프라인 또는 단계를 실행할 때 실행이 기록됩니다.
- `Event`는 아티팩트와 실행 간의 관계에 대한 기록입니다. 실행이 수행되면, 이벤트가 실행에 사용된 모든 아티팩트와 생성된 모든 아티팩트를 기록합니다. 이러한 기록을 사용하면 워크플로 전체에서 계보를 추적할 수 있습니다. 모든 이벤트를 살펴봄으로써 MLMD는 어떤 실행이 이루어졌는지, 그리고 그 결과로 만들어진 아티팩트는 무엇인지 파악합니다. 그러면 MLMD가 모든 아티팩트에서 모든 업스트림 입력으로 되돌릴 수 있습니다.
- `ContextType`은 워크플로에서 아티팩트 및 실행의 개념적 그룹 유형과 구조적 속성을 설명합니다. 예: 프로젝트, 파이프라인 실행, 실험, 소유자 등
- `Context`는 `ContextType`의 인스턴스로, 그룹 내에서 공유된 정보를 캡처합니다(예: 프로젝트 이름, 변경 목록 커밋 ID, 실험 주석 등). `ContextType` 내에 사용자가 정의한 고유 이름이 있습니다.
- `Attribution`은 아티팩트와 컨텍스트 간의 관계에 대한 기록입니다.
- `Association`은 실행과 컨텍스트 간의 관계에 대한 기록입니다.

## MLMD 기능

ML 워크플로 및 해당 계보에서 모든 구성 요소/단계의 입력 및 출력을 추적하면 ML 플랫폼에서 몇 가지 중요한 기능을 사용할 수 있습니다. 다음 목록은 몇 가지 주요 이점에 대한 포괄적인 개요를 제공합니다.

- **특정 유형의 모든 아티팩트를 나열합니다.** 예: 훈련된 모든 모델
- **비교를 위해 같은 유형의 두 아티팩트를 로드합니다.** 예: 두 실험의 결과를 비교합니다.
- **모든 관련 실행의 DAG와 컨텍스트의 입력 및 출력 아티팩트를 표시합니다.** 예: 디버깅 및 발견을 위한 실험의 워크플로를 시각화합니다.
- **모든 이벤트를 다시 반복하여 아티팩트가 어떻게 만들어졌는지 확인합니다.** 예: 모델에 들어간 데이터를 확인하고, 데이터 보존 계획을 시행합니다.
- **주어진 아티팩트를 사용하여 만들어진 모든 아티팩트를 식별합니다.** 예: 특정 데이터세트에서 훈련된 모든 모델을 확인하고, 잘못된 데이터를 기반으로 한 모델을 표시합니다.
- **이전에 같은 입력에서 실행이 수행되었는지 확인합니다.** 예: 구성 요소/단계가 이미 같은 작업을 완료했고 이전 출력을 다시 사용할 수 있는지 확인합니다.
- **워크플로 실행의 컨텍스트를 기록하고 쿼리합니다.** 예: 워크플로 실행에 사용되는 소유자 및 변경 목록을 추적하고, 실험별로 계보를 그룹화하며, 프로젝트별로 아티팩트를 관리합니다.
- **속성 및 1홉 인접 노드에 대한 선언적 노드 필터링 기능.** 예: 일부 파이프라인 컨텍스트에서 형식 아티팩트 찾기; 지정된 속성의 값이 범위 내에 있는 형식이 지정된 아티팩트 반환; 동일한 입력을 가진 컨텍스트에서 이전 실행 찾기.

MLMD API 및 메타데이터 저장소를 사용하여 계보 정보를 검색하는 방법을 보여주는 예제는 [MLMD 튜토리얼](https://www.tensorflow.org/tfx/tutorials/mlmd/mlmd_tutorial)을 참조하세요.

### ML 워크플로에 ML 메타데이터 통합하기

MLMD를 시스템에 통합하는 데 관심이 있는 플랫폼 개발자인 경우, 아래 예제 워크플로를 사용하여 저수준 MLMD API로 학습 작업의 실행을 추적해 보세요. 노트북 환경에서 더 높은 수준의 Python API를 사용하여 실험 메타데이터를 기록할 수도 있습니다.

![ML Metadata Example Flow](https://github.com/tensorflow/docs-l10n/blob/master/site/ko/tfx/guide/images/mlmd_flow.png?raw=true)

1. 아티팩트 유형을 등록합니다.

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

1. ML 워크플로의 모든 단계에 대한 실행 유형을 등록합니다.

```python
# Create an ExecutionType, e.g., Trainer
trainer_type = metadata_store_pb2.ExecutionType()
trainer_type.name = "Trainer"
trainer_type.properties["state"] = metadata_store_pb2.STRING
trainer_type_id = store.put_execution_type(trainer_type)

# Query a registered Execution type with the returned id
[registered_type] = store.get_execution_types_by_id([trainer_type_id])
```

1. DataSet ArtifactType의 아티팩트를 만듭니다.

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

1. Trainer 실행의 실행을 만듭니다.

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

1. 입력 이벤트를 정의하고 데이터를 읽습니다.

```python
# Define the input event
input_event = metadata_store_pb2.Event()
input_event.artifact_id = data_artifact_id
input_event.execution_id = run_id
input_event.type = metadata_store_pb2.Event.DECLARED_INPUT

# Record the input event in the metadata store
store.put_events([input_event])
```

1. 출력 아티팩트를 선언합니다.

```python
# Declare the output artifact of type SavedModel
model_artifact = metadata_store_pb2.Artifact()
model_artifact.uri = 'path/to/model/file'
model_artifact.properties["version"].int_value = 1
model_artifact.properties["name"].string_value = 'MNIST-v1'
model_artifact.type_id = model_type_id
[model_artifact_id] = store.put_artifacts([model_artifact])
```

1. 출력 이벤트를 기록합니다.

```python
# Declare the output event
output_event = metadata_store_pb2.Event()
output_event.artifact_id = model_artifact_id
output_event.execution_id = run_id
output_event.type = metadata_store_pb2.Event.DECLARED_OUTPUT

# Submit output event to the Metadata Store
store.put_events([output_event])
```

1. 실행을 완료된 것으로 표시합니다.

```python
trainer_run.id = run_id
trainer_run.properties["state"].string_value = "COMPLETED"
store.put_executions([trainer_run])
```

1. 특성 및 어설션 아티팩트를 사용하여 하나의 컨텍스트 아래에 아티팩트와 실행을 그룹화합니다.

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

## 원격 gRPC 서버와 함께 MLMD 사용하기

아래와 같이 원격 gRPC 서버에서 MLMD를 사용할 수 있습니다.

- 서버 시작

```bash
bazel run -c opt --define grpc_no_ares=true  //ml_metadata/metadata_store:metadata_store_server
```

기본적으로, 서버는 매 요청시 가짜 인메모리 db를 사용하며 호출 간에 메타데이터를 유지하지 않습니다. SQLite 파일 또는 MySQL 인스턴스를 사용하도록 MLMD `MetadataStoreServerConfig`로 구성할 수도 있습니다. 구성은 텍스트 protobuf 파일에 저장하고 `-metadata_store_server_config_file`을 이용해 바이너리로 전달할 수 있습니다.

텍스트 protobuf 형식의 `MetadataStoreServerConfig` 파일 예:

```textpb
connection_config {
  sqlite {
    filename_uri: '/tmp/test_db'
    connection_mode: READWRITE_OPENCREATE
  }
}
```

- 클라이언트 스텁을 만들고 Python에서 사용합니다.

```python
from grpc import insecure_channel
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2
from ml_metadata.proto import metadata_store_service_pb2_grpc
channel = insecure_channel('localhost:8080')
stub = metadata_store_service_pb2_grpc.MetadataStoreServiceStub(channel)
```

- RPC 호출에 MLMD를 사용합니다.

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

## 리소스

MLMD 라이브러리에는 ML 파이프라인과 함께 쉽게 사용할 수 있는 고급 API가 있습니다. 자세한 내용은 [MLMD API 문서](https://www.tensorflow.org/tfx/ml_metadata/api_docs/python/mlmd)를 참조하세요.

속성 및 1홉 이웃 노드에서 MLMD 선언적 노드 필터링 기능을 사용하는 방법을 알아보려면 [MLMD 선언적 노드 필터링](https://github.com/google/ml-metadata/blob/v1.2.0/ml_metadata/proto/metadata_store.proto#L708-L786)을 확인하세요.

또한 [MLMD 튜토리얼](https://www.tensorflow.org/tfx/tutorials/mlmd/mlmd_tutorial)에서 MLMD를 사용하여 파이프라인 구성 요소의 계보를 추적하는 방법을 알아보세요.

MLMD는 릴리스 간에 스키마와 데이터 마이그레이션을 처리하는 유틸리티를 제공합니다. 자세한 내용은 MLMD [가이드](https://github.com/google/ml-metadata/blob/master/g3doc/get_started.md#upgrade-the-mlmd-library)를 참조하세요.
