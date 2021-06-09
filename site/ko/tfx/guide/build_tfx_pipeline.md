# TFX 파이프라인 빌드하기

참고: TFX 파이프라인의 개념 보기는 [TFX 파이프라인 이해](understanding_tfx_pipelines)를 참조하세요.

참고: 깊게 들어가기 전에 첫 파이프라인을 빌드 해보고 싶습니까? [템플릿을 사용하여 파이프라인 빌드](#build_a_pipeline_using_a_template)하기로 시작하세요.

## `Pipeline` 클래스 사용

TFX 파이프라인은 [`Pipeline` 클래스](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/pipeline.py){: .external}를 사용하여 정의됩니다. 다음 예는 `Pipeline` 클래스를 사용하는 방법입니다.

<pre class="devsite-click-to-copy prettyprint">pipeline.Pipeline(
    pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;,
    pipeline_root=&lt;var&gt;pipeline-root&lt;/var&gt;,
    components=&lt;var&gt;components&lt;/var&gt;,
    enable_cache=&lt;var&gt;enable-cache&lt;/var&gt;,
    metadata_connection_config=&lt;var&gt;metadata-connection-config&lt;/var&gt;,
)
</pre>

다음을 대체합니다.

- <var>pipeline-name</var>: 이 파이프라인의 이름입니다. 파이프라인 이름은 고유해야 합니다.

    TFX는 구성 요소 입력 아티팩트에 대한 ML 메타데이터를 쿼리할 때 파이프라인 이름을 사용합니다. 파이프라인 이름을 재사용하면 예기치 않은 동작이 발생할 수 있습니다.

- <var>pipeline-root</var>: 이 파이프라인 출력의 루트 경로입니다. 루트 경로는 오케스트레이터가 읽기 및 쓰기 액세스 권한을 가진 디렉토리의 전체 경로여야 합니다. 런타임에 TFX는 파이프라인 루트를 사용하여 구성 요소 아티팩트에 대한 출력 경로를 생성합니다. 이 디렉토리는 로컬이거나 Google Cloud Storage 또는 HDFS와 같은 지원되는 분산 파일 시스템에 있을 수 있습니다.

- <var>components</var>: 이 파이프라인의 워크플로를 구성하는 구성 요소 인스턴스의 목록입니다.

- <var>enable-cache</var>: (선택 사항) 이 파이프라인이 파이프라인 실행 속도를 높이기 위해 캐싱을 사용하는지 여부를 나타내는 부울 값입니다.

- <var>metadata-connection-config</var>: (선택 사항) ML 메타데이터에 대한 연결 구성입니다.

## 구성 요소 실행 그래프 정의하기

구성 요소 인스턴스는 아티팩트를 출력으로 생성하고 일반적으로 업스트림 구성 요소 인스턴스가 입력으로 생성한 아티팩트에 의존합니다. 구성 요소 인스턴스의 실행 순서는 아티팩트 종속성의 DAG(방향성 비순환 그래프)를 생성하여 결정됩니다.

예를 들어, `ExampleGen` 표준 구성 요소는 CSV 파일에서 데이터를 수집하고 직렬화된 예제 레코드를 출력할 수 있습니다. `StatisticsGen` 표준 구성 요소는 이러한 예제 레코드를 입력으로 받고 데이터세트 통계를 생성합니다. 이 예에서 `SchemaGen`은 `ExampleGen`의 출력에 의존하기 때문에 `StatisticsGen`의 인스턴스는 `ExampleGen`을 따라야 합니다.

### 작업 기반 종속성

참고: 작업 기반 종속성을 사용하는 것은 일반적으로 권장되지 않습니다. 아티팩트 종속성으로 실행 그래프를 정의하면 TFX의 자동 아티팩트 계보 추적 및 캐싱 기능을 활용할 수 있습니다.

구성 요소의 [`add_upstream_node` 및 `add_downstream_node`](https://github.com/tensorflow/tfx/blob/master/tfx/components/base/base_node.py){: .external} 메서드를 사용하여 작업 기반 종속성을 정의할 수도 있습니다. `add_upstream_node`를 사용하여 현재 구성 요소가 지정된 구성 요소 다음에 실행되어야 함을 지정할 수 있습니다. `add_downstream_node`를 사용하면 현재 구성 요소가 지정된 구성 요소보다 먼저 실행되어야 함을 지정할 수 있습니다.

## 파이프라인 템플릿

파이프라인을 빠르게 설정하고 모든 부분이 어떻게 결합되는지 확인하는 가장 쉬운 방법은 템플릿을 사용하는 것입니다. 템플릿 사용은 [로컬로 TFX 파이프라인 빌드하기](build_local_pipeline)에서 다룹니다.

## 캐싱

TFX 파이프라인 캐싱을 사용하면 파이프라인이 이전 파이프라인 실행에서 동일한 입력 세트로 실행된 구성 요소를 건너뛸 수 있습니다. 캐싱이 활성화된 경우, 파이프라인은 각 구성 요소의 서명, 구성 요소 및 입력 세트를 이 파이프라인의 이전 구성 요소 실행 중 하나와 일치시키려고 합니다. 일치하는 항목이 있으면 파이프라인은 이전 실행의 구성 요소 출력을 사용합니다. 일치하는 항목이 없으면 구성 요소가 실행됩니다.

파이프라인이 비결정적 구성 요소를 사용하는 경우에는 캐싱을 사용하지 마세요. 예를 들어, 파이프라인에 대한 난수를 생성하는 구성 요소를 생성하는 경우, 캐시를 사용하면 이 구성 요소가 한 번 실행됩니다. 이 예에서 후속 실행에 난수를 생성하는 대신 첫 번째 실행의 난수가 사용됩니다.
