# TFX 파이프라인 이해하기

MLOps는 머신러닝(ML) 워크플로를 자동화, 관리 및 감사하는 데 도움이 되는 DevOps 사례를 적용하는 방법입니다. ML 워크플로에는 다음 단계가 포함됩니다.

- 데이터를 준비, 분석 및 변환합니다.
- 모델을 훈련하고 평가합니다.
- 훈련된 모델을 운영에 배포합니다.
- ML 아티팩트를 추적하고 종속성을 이해합니다.

이러한 단계를 임시적 방식으로 관리하는 것은 어렵고 시간이 많이 소요될 수 있습니다.

TFX는 Apache Airflow, Apache Beam 및 Kubeflow Pipelines와 같은 다양한 오케스트레이터에서 ML 프로세스를 오케스트레이션하는 데 도움이 되는 도구 키트를 제공하여 MLOps를 쉽게 구현할 수 있도록 지원합니다. 워크플로를 TFX 파이프라인으로 구현하여 다음을 수행할 수 있습니다.

- ML 프로세스를 자동화하여 모델을 정기적으로 재훈련, 평가 및 배포할 수 있습니다.
- 대규모 데이터세트 및 워크로드를 처리하기 위해 분산 컴퓨팅 리소스를 활용합니다.
- 다양한 하이퍼 매개변수 세트로 파이프라인을 실행하여 실험 속도를 높입니다.

이 가이드에서는 TFX 파이프라인을 이해하는 데 필요한 핵심 개념을 설명합니다.

## 아티팩트

TFX 파이프라인에서 단계의 출력을 **아티팩트**라고 합니다. 워크플로의 후속 단계에서는 아티팩트를 입력으로 사용할 수 있습니다. 이러한 방식으로 TFX를 사용하면 워크플로 단계 간에 데이터를 전송할 수 있습니다.

예를 들어, `ExampleGen` 표준 구성 요소는 `StatisticsGen` 표준 구성 요소와 같은 구성 요소가 입력으로 사용하는 직렬화된 예를 내보냅니다.

아티팩트는 <a>ML 메타데이터</a> 저장소에 등록된 <strong>아티팩트 유형</strong>을 사용하는 강력한 유형이어야 합니다. [ML 메타데이터에서 사용되는 개념](mlmd#concepts)에 대해 자세히 알아보세요.

아티팩트 유형에는 이름이 있으며 해당 속성의 스키마를 정의합니다. 아티팩트 유형 이름은 ML 메타데이터 저장소에서 고유해야 합니다. TFX는 복잡한 데이터 유형 및 값 유형(예: 문자열, 정수, 부동 소수점)을 설명하는 여러 [표준 아티팩트 유형](https://github.com/tensorflow/tfx/blob/master/tfx/types/standard_artifacts.py){: .external}을 제공합니다. 이 [아티팩트 유형을 재사용](https://github.com/tensorflow/tfx/blob/master/tfx/types/standard_artifacts.py){: .external}하거나 [`Artifact`](https://github.com/tensorflow/tfx/blob/master/tfx/types/artifact.py){: .external}에서 파생되는 사용자 정의 아티팩트 유형을 정의할 수 있습니다.

## 매개변수

매개변수는 파이프라인이 실행되기 전에 알려진 파이프라인에 대한 입력입니다. 매개변수를 사용하면 코드 대신 구성을 통해 파이프라인의 동작 또는 파이프라인의 일부를 변경할 수 있습니다.

예를 들어, 매개변수를 사용하여 파이프라인의 코드를 변경하지 않고 다른 하이퍼 매개변수 세트로 파이프라인을 실행할 수 있습니다.

매개변수를 사용하면 다양한 매개변수 세트로 파이프라인을 더 쉽게 실행할 수 있으므로 실험 속도를 높일 수 있습니다.

[RuntimeParameter 클래스](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/data_types.py){: .external}에 대해 자세히 알아보세요.

## 구성 요소

**구성 요소**는 TFX 파이프라인에서 단계로 사용할 수 있는 ML 작업의 구현입니다. 구성 요소는 다음으로 구성됩니다.

- 구성 요소의 입력 및 출력 아티팩트과 구성 요소의 필수 매개변수를 정의하는 구성 요소 사양
- 데이터 수집 및 변환 또는 모델 훈련 및 평가와 같은 ML 워크플로의 단계를 수행하는 코드를 구현하는 실행자
- 파이프라인에서 사용하기 위해 구성 요소 사양 및 실행자를 패키지화하는 구성 요소 인터페이스

TFX는 파이프라인에서 사용할 수 있는 몇 가지 [표준 구성 요소](index#tfx_standard_components)를 제공합니다. 이러한 구성 요소가 요구 사항을 충족하지 않으면 사용자 정의 구성 요소를 빌드할 수 있습니다. [사용자 정의 구성 요소에 대해 자세히 알아보세요](understanding_custom_components).

## 파이프라인

TFX 파이프라인은 Apache Airflow, Apache Beam 및 Kubeflow Pipelines와 같은 다양한 오케스트레이터에서 실행할 수 있는 ML 워크플로의 이식 가능한 구현입니다. 파이프라인은 구성 요소 인스턴스와 입력 매개변수로 구성됩니다.

구성 요소 인스턴스는 아티팩트를 출력으로 생성하고, 일반적으로 업스트림 구성 요소 인스턴스에서 입력으로 생성한 아티팩트에 따라 달라집니다. 구성 요소 인스턴스의 실행 시퀀스는 아티팩트 종속성의 방향성 비순환 그래프를 만들어 결정됩니다.

예를 들어, 다음을 수행하는 파이프라인을 고려하세요.

- 사용자 정의 구성 요소를 사용하여 독점 시스템에서 직접 데이터를 수집합니다.
- StatisticsGen 표준 구성 요소를 사용하여 훈련 데이터의 통계를 계산합니다.
- SchemaGen 표준 구성 요소를 사용하여 데이터 스키마를 만듭니다.
- ExampleValidator 표준 구성 요소를 사용하여 훈련 데이터에 이상이 있는지 확인합니다.
- Transform 표준 구성 요소를 사용하여 데이터세트에 대한 특성 엔지니어링(feature engineering)을 수행합니다.
- Trainer 표준 구성 요소를 사용하여 모델을 훈련합니다.
- Evaluator 구성 요소를 사용하여 훈련된 모델을 평가합니다.
- 모델이 평가를 통과하면 파이프라인은 사용자 정의 구성 요소를 사용하여 훈련된 모델을 독점 배포 시스템의 큐에 추가합니다.

![](images/tfx_pipeline_graph.svg)

구성 요소 인스턴스의 실행 시퀀스를 결정하기 위해 TFX는 아티팩트 종속성을 분석합니다.

- 데이터 수집 구성 요소에는 아티팩트 종속성이 없으므로 그래프의 첫 번째 노드가 될 수 있습니다.
- StatisticsGen은 데이터 수집으로 생성된 *예제*에 따라 달라지므로 데이터 수집 후에 실행해야 합니다.
- SchemaGen은 StatisticsGen에서 생성된 *통계*에 따라 달라지므로 StatisticsGen 후에 실행해야 합니다.
- ExampleValidator는 StatisticsGen에서 생성된 *통계*와 SchemaGen으로 생성된 *스키마*에 따라 달라지므로, StatisticsGen과 SchemaGen 후에 실행해야 합니다.
- Transform은 데이터 수집으로 생성된 *예제*와 SchemaGen에서 만들어진 *스키마*에 따라 달라지므로 데이터 수집과 SchemaGen 후에 실행해야 합니다.
- Trainer는 데이터 수집으로 생성된 *예제* , SchemaGen에서 만들어진 *스키마* 및 Transform으로 생성된 *저장된 모델*에 따라 달라집니다. Trainer는 데이터 수집, SchemaGen과 Transform 후에만 실행할 수 있습니다.
- Evaluator는 데이터 수집으로 생성된 *예제*와 Trainer에서 생성된 *저장된 모델*에 따라 달라지므로 데이터 수집과 Trainer 후에 실행해야 합니다.
- 사용자 정의 배포자는 Trainer에서 생성된 *저장된 모델*과 Evaluator에서 생성된 *분석 결과*에 따라 달라지므로 배포자는 Trainer와 Evaluator 후에 실행해야 합니다.

이 분석을 기반으로 오케스트레이터는 다음과 같이 실행합니다.

- 데이터 수집, StatisticsGen, SchemaGen 구성 요소 인스턴스가 순차적으로 실행됩니다.
- ExampleValidator 및 Transform 구성 요소는 입력 아티팩트 종속성을 공유하고 서로의 출력에 의존하지 않으므로 병렬로 실행될 수 있습니다.
- Transform 구성 요소가 완료되면 Trainer, Evaluator 및 사용자 정의 배포자 구성 요소 인스턴스가 순차적으로 실행됩니다.

[TFX 파이프라인 빌드하기](build_tfx_pipeline)에 대해 자세히 알아보세요.

## TFX 파이프라인 템플릿

TFX 파이프라인 템플릿을 사용하면 사용 사례에 맞게 사용자 정의할 수 있는 사전 빌드된 파이프라인을 제공하여 파이프라인 개발을 더 쉽게 시작할 수 있습니다.

[TFX 파이프라인 템플릿 사용자 정의하기](build_tfx_pipeline#build-a-pipeline-using-a-template)에 대해 자세히 알아보세요.

## 파이프라인 실행

실행은 파이프라인의 단일 실행입니다.

## 오케스트레이터

오케스트레이터는 파이프라인 실행을 수행할 수 있는 시스템입니다. TFX는 [Apache Airflow](airflow), [Apache Beam](beam_orchestrator) 및 [Kubeflow Pipelines](kubeflow)와 같은 오케스트레이터를 지원합니다. TFX는 또한 *DagRunner*라는 용어를 사용하여 오케스트레이터를 지원하는 구현을 나타냅니다.
