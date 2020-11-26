# TFX 파이프라인 빌드하기

TFX를 사용하면 머신러닝(ML) 워크플로를 파이프라인으로 쉽게 오케스트레이션하여 다음과 같은 결과를 얻을 수 있습니다.

- ML 프로세스를 자동화하여 모델을 정기적으로 재훈련, 평가 및 배포할 수 있습니다.
- 모델 성능에 대한 심층적인 분석과 새로 훈련된 모델의 검증을 포함한 ML 파이프라인을 생성하여 성능과 안정성을 보장합니다.
- 훈련 데이터에 이상이 있는지 모니터링하고 이상 징후를 발견하고 훈련이 편향적으로 적용되는 것을 방지합니다.
- 다양한 하이퍼 매개변수 세트로 파이프라인을 실행하여 실험 속도를 높입니다.

이 가이드에서는 파이프라인을 빌드하는 두 가지 방법을 설명합니다.

- ML 워크플로의 요구 사항에 맞게 TFX 파이프라인 템플릿을 사용자 정의합니다. TFX 파이프라인 템플릿은 TFX 표준 구성 요소를 사용한 모범 사례를 보여주는 사전 빌드된 워크플로입니다.
- TFX를 사용하여 파이프라인을 빌드합니다. 이 사용 사례에서는 템플릿으로 시작하지 않고 파이프라인을 정의합니다.

TFX 파이프라인을 처음 접하는 경우, 우선 [TFX 파이프라인의 핵심 개념에 대해 자세히 알아보세요](understanding_tfx_pipelines).

## TFX 파이프라인 개요

참고: 깊게 들어가기 전에 첫 파이프라인을 빌드해보고 싶습니까? [템플릿을 사용하여 파이프라인 빌드하기](#build_a_pipeline_using_a_template)로 시작하세요.

TFX 파이프라인은 [`Pipeline` 클래스](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/pipeline.py){: .external}를 사용하여 정의됩니다. 다음 예는 `Pipeline` 클래스를 사용하는 방법입니다.

<pre class="devsite-click-to-copy prettyprint">pipeline.Pipeline(
    pipeline_name=<var>pipeline-name</var>,
    pipeline_root=<var>pipeline-root</var>,
    components=<var>components</var>,
    enable_cache=<var>enable-cache</var>,
    metadata_connection_config=<var>metadata-connection-config</var>,
    beam_pipeline_args=<var>beam_pipeline_args</var>
)
</pre>

다음을 대체합니다.

- <var>pipeline-name</var>: 이 파이프라인의 이름입니다. 파이프라인 이름은 고유해야 합니다.

    TFX는 구성 요소 입력 아티팩트에 대한 ML 메타데이터를 쿼리할 때 파이프라인 이름을 사용합니다. 파이프라인 이름을 재사용하면 예기치 않은 동작이 발생할 수 있습니다.

- <var>pipeline-root</var>: 이 파이프라인 출력의 루트 경로입니다. 루트 경로는 오케스트레이터가 읽기 및 쓰기 액세스 권한을 가진 디렉토리의 전체 경로여야 합니다. 런타임에 TFX는 파이프라인 루트를 사용하여 구성 요소 아티팩트에 대한 출력 경로를 생성합니다. 이 디렉토리는 로컬이거나 Google Cloud Storage 또는 HDFS와 같은 지원되는 분산 파일 시스템에 있을 수 있습니다.

- <var>components</var>: 이 파이프라인의 워크플로를 구성하는 구성 요소 인스턴스의 목록입니다.

- <var>enable-cache</var>: (선택 사항) 이 파이프라인이 파이프라인 실행 속도를 높이기 위해 캐싱을 사용하는지 여부를 나타내는 부울 값입니다.

- <var>metadata-connection-config</var>: (선택 사항) ML 메타데이터에 대한 연결 구성입니다.

- <var>beam_pipeline_args</var>: (선택 사항) 계산을 실행하기 위해 Beam을 사용하는 모든 구성 요소에 대해 Apache Beam 실행기에 전달되는 인수 세트입니다.

### 구성 요소 실행 그래프 정의하기

구성 요소 인스턴스는 아티팩트를 출력으로 생성하고 일반적으로 업스트림 구성 요소 인스턴스가 입력으로 생성한 아티팩트에 의존합니다. 구성 요소 인스턴스의 실행 순서는 아티팩트 종속성의 DAG(방향성 비순환 그래프)를 생성하여 결정됩니다.

예를 들어, `ExampleGen` 표준 구성 요소는 CSV 파일에서 데이터를 수집하고 직렬화된 예제 레코드를 출력할 수 있습니다. `StatisticsGen` 표준 구성 요소는 이러한 예제 레코드를 입력으로 받고 데이터세트 통계를 생성합니다. 이 예에서 `SchemaGen`은 `ExampleGen`의 출력에 의존하기 때문에 `StatisticsGen`의 인스턴스는 `ExampleGen`을 따라야 합니다.

구성 요소의 [`add_upstream_node` 및 `add_downstream_node`](https://github.com/tensorflow/tfx/blob/master/tfx/components/base/base_node.py){: .external} 메서드를 사용하여 작업 기반 종속성을 정의할 수도 있습니다. `add_upstream_node`를 사용하여 현재 구성 요소가 지정된 구성 요소 다음에 실행되어야 함을 지정할 수 있습니다. `add_downstream_node`를 사용하면 현재 구성 요소가 지정된 구성 요소보다 먼저 실행되어야 함을 지정할 수 있습니다.

참고: 작업 기반 종속성을 사용하는 것은 일반적으로 권장되지 않습니다. 아티팩트 종속성으로 실행 그래프를 정의하면 TFX의 자동 아티팩트 계보 추적 및 캐싱 기능을 활용할 수 있습니다.

### 캐싱

TFX 파이프라인 캐싱을 사용하면 파이프라인이 이전 파이프라인 실행에서 동일한 입력 세트로 실행된 구성 요소를 건너뛸 수 있습니다. 캐싱이 활성화된 경우, 파이프라인은 각 구성 요소의 서명, 구성 요소 및 입력 세트를 이 파이프라인의 이전 구성 요소 실행 중 하나와 일치시키려고 합니다. 일치하는 항목이 있으면 파이프라인은 이전 실행의 구성 요소 출력을 사용합니다. 일치하는 항목이 없으면 구성 요소가 실행됩니다.

파이프라인이 비결정적 구성 요소를 사용하는 경우에는 캐싱을 사용하지 마세요. 예를 들어, 파이프라인에 대한 난수를 생성하는 구성 요소를 생성하는 경우, 캐시를 사용하면 이 구성 요소가 한 번 실행됩니다. 이 예에서 후속 실행에 난수를 생성하는 대신 첫 번째 실행의 난수가 사용됩니다.

## 템플릿을 사용하여 파이프라인 빌드하기

TFX 파이프라인 템플릿을 사용하면 사용 사례에 맞게 사용자 정의할 수 있는 사전 빌드된 파이프라인을 제공하여 파이프라인 개발을 더 쉽게 시작할 수 있습니다.

다음 섹션에서는 템플릿 사본을 만들고 필요에 맞게 사용자 정의하는 방법에 대해 설명합니다.

### 파이프라인 템플릿의 복사본 만들기

1. 다음 명령을 실행하여 TFX 파이프라인 템플릿을 나열합니다.

    <pre class="devsite-click-to-copy devsite-terminal">tfx template list
    </pre>

2. 목록에서 템플릿을 선택합니다. 현재 **taxi**가 유일한 템플릿입니다. 그리고 다음 명령을 실행합니다.

    <pre class="devsite-click-to-copy devsite-terminal">tfx template copy --model=<var>template</var> --pipeline_name=<var>pipeline-name</var> \
    --destination_path=<var>destination-path</var>
    </pre>

    다음을 대체합니다.

    - <var>template</var>: 복사할 템플릿의 이름입니다.
    - <var>pipeline-name</var>: 생성할 파이프라인의 이름입니다.
    - <var>destination-path</var>: 템플릿을 복사해 넣을 경로입니다.

    [`tfx template copy` 명령](cli#copy)에 대해 자세히 알아보세요.

3. 지정한 경로에 파이프라인 템플릿의 복사본이 생성됩니다.

### 파이프라인 템플릿 살펴보기

이 섹션에서는 **택시** 템플릿으로 만든 스캐폴딩을 개괄적으로 설명합니다.

1. 템플릿에서 파이프라인으로 복사된 파일을 탐색합니다. **택시** 템플릿은 다음을 생성합니다.

    - **data.csv** 파일이 있는 **data** 디렉토리

    - 전처리 코드와 `tf.estimators` 및 Keras를 사용한 모델 구현을 포함한 **models** 디렉토리

    - 파이프라인 구현과 구성 스크립트가 있는 **pipeline** 디렉토리

    - 템플릿은 다음을 대상 경로에 복사합니다.

        - Apache Beam 및 Kubeflow 파이프라인용 DAG 실행기 코드
        - [ML 메타데이터](mlmd) 저장소에서 아티팩트를 탐색하기 위한 노트북

2. 파이프라인 디렉토리에서 다음 명령을 실행합니다.

    <pre class="devsite-click-to-copy devsite-terminal">python beam_dag_runner.py
    </pre>

    이 명령은 Apache Beam으로 파이프라인 실행을 생성하여 파이프라인에 다음 디렉토리를 추가합니다.

    - Apache Beam에서 로컬로 사용하는 ML 메타데이터 저장소가 포함된 **tfx_metadata** 디렉토리
    - 파이프라인의 파일 출력을 포함하는 **tfx_pipeline_output** 디렉토리

    참고: Apache Beam은 TFX에서 지원되는 여러 오케스트레이터 중 하나입니다. Apache Beam은 더 작은 데이터세트로 더 빠른 반복을 위해 로컬에서 파이프라인을 실행하는 데 특히 적합합니다. Apache Beam은 단일 머신에서 실행되기 때문에 프로덕션 용도로 적합하지 않을 수 있으며, 시스템을 사용할 수 없게 되었을 때 작업을 손실할 위험이 더 큽니다. TFX는 Apache Airflow 및 Kubeflow Pipeline과 같은 오케스트레이터도 지원합니다. 다른 오케스트레이터와 함께 TFX를 사용하는 경우, 해당 오케스트레이터에 적합한 DAG 실행기를 사용하세요.

3. 파이프라인의 `pipeline/configs.py` 파일을 열고 내용을 검토합니다. 이 스크립트는 파이프라인 및 구성 요소 함수에서 사용하는 구성 옵션을 정의합니다.

4. 파이프라인의 `pipeline/pipeline.py` 파일을 열고 내용을 검토합니다. 이 스크립트는 TFX 파이프라인을 생성합니다. 처음에는 파이프라인에 ExampleGen 구성 요소만 포함됩니다. 파이프라인 파일의 **TODO** 주석에 나온 지침에 따라 파이프라인에 스텝을 추가합니다.

5. 파이프라인의 `beam_dag_runner.py` 파일을 열고 내용을 검토합니다. 이 스크립트는 파이프라인 실행을 생성하고 `data_path` 및 `preprocessing_fn`과 같은 실행 *매개변수를* 지정합니다.

6. 템플릿으로 생성된 스캐폴딩을 검토하고 Apache Beam을 사용하여 파이프라인 실행을 생성했습니다. 이제, 요구 사항에 맞게 템플릿을 사용자 정의합니다.

### 파이프라인 사용자 정의하기

이 섹션에서는 **택시** 템플릿의 사용자 정의를 시작하는 방법에 대해 개괄적으로 설명합니다.

1. 파이프라인을 설계합니다. 템플릿이 제공하는 스캐폴딩은 TFX 표준 구성 요소를 사용하여 테이블 형식 데이터에 적합하게 파이프라인을 구현하는 데 도움을 줍니다. 기존 ML 워크플로를 파이프라인으로 이동하는 경우, [TFX 표준 구성 요소](index#tfx_standard_components)를 최대한 활용하도록 코드를 수정해야 할 수 있습니다. 워크플로에 고유하거나 아직 TFX 표준 구성 요소에서 지원하지 않는 기능을 구현하는 [사용자 정의 구성 요소](understanding_custom_components)를 만들어야 할 수도 있습니다.

2. 파이프라인을 설계했으면 다음 프로세스를 사용하여 파이프라인을 반복적으로 사용자 정의합니다. 일반적으로, `ExampleGen` 구성 요소인 파이프라인으로 데이터를 수집하는 구성 요소부터 시작합니다.

    1. 사용 사례에 맞게 파이프라인 또는 구성 요소를 사용자 정의합니다. 이러한 사용자 정의에는 다음과 같은 변경이 포함될 수 있습니다.

        - 파이프라인 매개변수 변경
        - 파이프라인에 구성 요소를 추가하거나 제거
        - Replacing the data input source. This data source can either be a file or queries into services such as BigQuery.
        - 파이프라인에서 구성 요소의 구성 변경
        - 구성요소의 사용자 정의 함수 변경

    2. `beam_dag_runner.py` 스크립트를 사용하거나, 다른 오케스트레이터를 사용하는 경우 다른 적절한 DAG 실행기를 사용하여 구성 요소를 로컬에서 실행합니다. 스크립트가 실패하면 오류를 디버그하고 스크립트를 다시 실행합니다.

    3. 이 사용자 정의가 문제 없이 동작하면 다음 사용자 정의로 진행합니다.

3. 반복적으로 작업하면서 필요에 맞게 템플릿 워크플로의 각 스텝을 사용자 정의할 수 있습니다.

## 사용자 정의 파이프라인 빌드하기

템플릿을 사용하지 않고 사용자 정의 파이프라인을 빌드하는 방법에 대해 자세히 알아보려면 다음 지침을 따르세요.

1. 파이프라인을 설계합니다. TFX 표준 구성 요소는 완전한 ML 워크플로를 구현하는 데 도움이 되는 입증된 기능을 제공합니다. 기존 ML 워크플로를 파이프라인으로 이동하는 경우, TFX 표준 구성 요소를 최대한 활용하기 위해 코드를 수정해야 할 수 있습니다. 또한 데이터 증대와 같은 기능을 구현하는 [사용자 정의 구성 요소](understanding_custom_components)를 만들어야 할 수도 있습니다.

    - [표준 TFX 구성 요소](index#tfx_standard_components)에 대해 자세히 알아보세요.
    - [사용자 정의 구성 요소](understanding_custom_components)에 대해 자세히 알아보세요.

2. 다음 예를 사용하여 파이프라인을 정의하는 스크립트 파일을 만듭니다. 이 가이드에서는 이 파일을 `my_pipeline.py`라고 합니다.

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

    다음 단계에서는 `create_pipeline`에서 파이프라인을 정의하고 `run_pipeline`에서 Apache Beam을 사용하여 파이프라인을 로컬로 실행합니다.

    다음 프로세스를 사용하여 파이프라인을 반복적으로 빌드합니다.

    1. 사용 사례에 맞게 파이프라인 또는 구성 요소를 사용자 정의합니다. 이러한 사용자 정의에는 다음과 같은 변경이 포함될 수 있습니다.

        - 파이프라인 매개변수 변경
        - 파이프라인에 구성 요소를 추가하거나 제거
        - 데이터 입력 파일 교체
        - 파이프라인에서 구성 요소의 구성 변경
        - 구성요소의 사용자 정의 함수 변경

    2. 스크립트 파일을 실행하여 Apache Beam 또는 다른 오케스트레이터를 사용하여 구성 요소를 로컬에서 실행합니다. 스크립트가 실패하면 오류를 디버그하고 스크립트를 다시 실행합니다.

    3. 이 사용자 정의가 문제 없이 동작하면 다음 사용자 정의로 진행합니다.

    파이프라인 워크플로의 첫 번째 노드부터 시작합니다. 일반적으로, 첫 번째 노드는 파이프라인으로 데이터를 수집합니다.

3. 워크플로의 첫 번째 노드를 파이프라인에 추가합니다. 이 예에서 파이프라인은 `ExampleGen` 표준 구성 요소를 사용하여 `./data`의 디렉토리에서 CSV를 로드합니다.

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

    `CsvExampleGen`은 지정된 데이터 경로에서 CSV의 데이터를 사용하여 직렬화된 예제 레코드를 만듭니다. `CsvExampleGen` 구성 요소의 `input` 매개변수를 [`external_input`](https://github.com/tensorflow/tfx/blob/master/tfx/utils/dsl_utils.py){: .external}로 설정하여 데이터 경로가 파이프라인으로 전달되고 경로가 아티팩트로 저장되도록 지정합니다.

4. `my_pipeline.py`와 동일한 디렉토리에 `data` 디렉토리를 만듭니다. `data` 디렉토리에 작은 CSV 파일을 추가합니다.

5. 다음 명령을 사용하여 `my_pipeline.py` 스크립트를 실행하고 Apache Beam 또는 다른 오케스트레이터로 파이프라인을 테스트합니다.

    <pre class="devsite-click-to-copy devsite-terminal">python my_pipeline.py
    </pre>

    결과는 다음과 같아야 합니다.

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

6. 계속해서 파이프라인에 구성 요소를 반복적으로 추가합니다.
