# Apache Beam 및 TFX

[Apache Beam](https://beam.apache.org/)은 다양한 실행 엔진에서 실행되는 배치 및 스트리밍 데이터 처리 작업을 실행하기 위한 프레임워크를 제공합니다. 일부 TFX 라이브러리는 작업 실행에 Beam을 사용하여 컴퓨팅 클러스터 전체에서 높은 수준의 확장성을 실현합니다. Beam에는 단일 컴퓨팅 노드에서 실행되고 개발, 테스트 또는 소규모 배포에 매우 유용한 직접 러너를 포함하여 다양한 실행 엔진 또는 '러너'에 대한 지원이 포함됩니다. Beam은 코드 수정 없이 지원되는 모든 러너에서 TFX를 실행할 수 있도록 해주는 추상화 레이어를 제공합니다. TFX는 Beam Python API를 사용하므로 Python API에서 지원하는 러너로 제한됩니다.

## 배포 및 확장성

워크로드 요구 사항이 증가함에 따라 Beam을 대규모 컴퓨팅 클러스터를 대상으로 한 대규모 배포로 확장할 수 있으며, 이는 기본 러너의 확장성에 의해서만 제한됩니다. 대규모 배포에서 러너는 일반적으로 애플리케이션 배포, 확장 및 관리를 자동화하기 위해 Kubernetes 또는 Apache Mesos와 같은 컨테이너 오케스트레이션 시스템에 배포됩니다.

Apache Beam에 대한 자세한 내용은 [Apache Beam](https://beam.apache.org/) 설명서를 참조하세요.

Google Cloud 사용자에게 [Dataflow](https://cloud.google.com/dataflow)는 리소스 자동 확장, 동적 작업 재조정, 다른 Google Cloud 서비스와의 긴밀한 통합, 기본 제공 보안, 모니터링을 통해 서버 없이 비용 효율적인 플랫폼을 제공하는 권장 러너입니다.

## 사용자 정의 Python 코드 및 종속성

TFX 파이프라인에서 Beam을 사용할 때의 한 가지 주목할 복잡성 중 하나는 사용자 정의 코드 및/또는 추가 Python 모듈에서 필요한 종속성을 처리한다는 것입니다. 다음은 이것이 문제가 될 수 있는 몇 가지 예입니다.

- preprocessing_fn은 사용자 자신의 Python 모듈을 참조해야 함
- Evaluator 구성 요소에 대한 사용자 정의 추출기
- TFX 구성 요소에서 하위 클래스화된 사용자 정의 모듈

TFX는 Beam의 [Python 파이프라인 종속성 관리](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/) 지원에 의존하여 Python 종속성을 처리합니다. 현재 이를 관리하는 두 가지 방법이 있습니다.

1. Python 코드 및 종속성을 소스 패키지로 제공
2. [Dataflow 전용] 컨테이너 이미지를 작업자로 사용

이어서 이 내용에 대해 알아봅니다.

### Python 코드 및 종속성을 소스 패키지로 제공하기

다음과 같은 사용자에게 권장됩니다.

1. Python 패키징에 익숙합니다.
2. Python 소스 코드만 사용합니다(예: C 모듈 또는 공유 라이브러리 없음).

[Python 파이프라인 종속성 관리](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/)의 경로 중 하나를 따라 다음 beam_pipeline_args 중 하나를 사용하여 이를 제공합니다.

- --setup_file
- --extra_package
- --requirements_file

알림: 위의 경우 동일한 버전의 `tfx`가 종속성으로 나열되어 있는지 확인하세요.

### [Dataflow 전용] 컨테이너 이미지를 작업자로 사용하기

TFX 0.26.0 이상은 Dataflow 작업자를 위한 [사용자 정의 컨테이너 이미지](https://beam.apache.org/documentation/runtime/environments/#customizing-container-images) 사용을 실험적으로 지원합니다.

이를 사용하려면 다음을 수행해야 합니다.

- `tfx`와 사용자의 사용자 정의 코드 및 종속성이 사전 설치된 Docker 이미지를 빌드합니다.
    - (1) `tfx>=0.26`을 사용하고 (2) python 3.7을 사용하여 파이프라인을 개발하는 사용자의 경우, 가장 쉬운 방법은 공식 `tensorflow/tfx` 이미지의 해당 버전을 확장하는 것입니다.

```Dockerfile
# You can use a build-arg to dynamically pass in the
# version of TFX being used to your Dockerfile.

ARG TFX_VERSION
FROM tensorflow/tfx:${TFX_VERSION}
# COPY your code and dependencies in
```

- 빌드된 이미지를 Dataflow에서 사용하는 프로젝트에서 액세스할 수 있는 컨테이너 이미지 레지스트리에 푸시합니다.
    - Google Cloud 사용자는 위의 단계를 멋지게 자동화하는 [Cloud Build](https://cloud.google.com/cloud-build/docs/quickstart-build) 사용을 고려할 수 있습니다.
- 다음 `beam_pipeline_args`를 제공합니다.

```python
beam_pipeline_args.extend([
    '--runner=DataflowRunner',
    '--project={project-id}',
    '--worker_harness_container_image={image-ref}',
    '--experiments=use_runner_v2',
])
```

**TODO(b/171733562): use_runner_v2가 Dataflow의 기본값이 되면 제거합니다.**

**TODO(b/179738639): https://issues.apache.org/jira/browse/BEAM-5440에 따라 로컬에서 사용자 정의 컨테이너를 테스트하는 방법에 대한 문서를 작성합니다.**

## Beam 파이프라인 인수

여러 TFX 구성 요소가 분산 데이터 처리를 위해 Beam을 사용하며, 파이프라인 생성 중에 지정되는 `beam_pipeline_args`로 구성됩니다.

```python
my_pipeline = Pipeline(
    ...,
    beam_pipeline_args=[...])
```

TFX 0.30 이상은 구성 요소별 파이프라인 레벨 Beam 인수를 확장하기 위해 `with_beam_pipeline_args` 인터페이스를 추가합니다.

```python
example_gen = CsvExampleGen(input_base=data_root).with_beam_pipeline_args([...])
```
