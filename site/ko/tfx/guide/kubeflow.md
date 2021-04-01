# TFX 파이프라인 오케스트레이션하기

## Kubeflow Pipelines

[Kubeflow](https://www.kubeflow.org/)는 Kubernetes에 머신러닝(ML) 워크플로를 간단하고 이식 가능하며 확장 가능하게 배포하는 오픈 소스 ML 플랫폼입니다. [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/pipelines-overview/)는 Kubeflow에서 재현 가능한 워크플로를 구성하고 실행할 수 있는 Kubeflow 플랫폼의 일부이며, 실험 및 노트북 기반 환경과 통합됩니다. Kubernetes의 Kubeflow Pipelines 서비스에는 호스팅된 메타데이터 저장소, 컨테이너 기반 오케스트레이션 엔진, 노트북 서버 및 UI가 포함되어 있어 사용자가 복잡한 ML 파이프라인을 대규모로 개발, 실행 및 관리할 수 있습니다. Kubeflow Pipelines SDK를 사용하면 프로그래밍 방식으로 구성 요소와 구성 및 파이프라인을 만들고 공유할 수 있습니다.

Google Cloud에서 대규모로 TFX를 실행하는 방법에 대한 자세한 내용은 [Kubeflow Pipelines의 TFX 예제](https://www.tensorflow.org/tfx/tutorials/tfx/cloud-ai-platform-pipelines)를 참조하세요.
