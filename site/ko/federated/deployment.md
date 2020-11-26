# 배포

In addition to defining computations, TFF provides tools for executing them. Whereas the primary focus is on simulations, the interfaces and tools we provide are more general. This document outlines the options for deployment to various types of platform.

참고: 이 문서는 아직 제작 중입니다.

## Overview

TFF 계산을 위한 두 가지 주요 배포 모드가 있습니다.

- **네이티브 백엔드**: 백엔드를 [`computation.proto`](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/proto/v0/computation.proto)에 정의된 대로 TFF 계산의 구문 구조를 해석할 수 있는 경우, *native*라고 합니다. 네이티브 백엔드가 반드시 모든 언어 구문이나 내장 함수를 지원할 필요는 없습니다. 네이티브 백엔드는 Python 코드에서 소비하기 위한 [`tff.framework.Executor`](https://www.tensorflow.org/federated/api_docs/python/tff/framework/Executor)와 같은 표준 TFF *executor* 인터페이스 중 하나 또는 gRPC 엔드포인트로 노출된 [`executor.proto`](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/proto/v0/executor.proto)에 정의된 언어 독립적 버전을 구현해야 합니다.

    위의 인터페이스를 지원하는 네이티브 백엔드는 기본 참조 런타임 대신 양방향으로 사용하여 노트북 또는 실험 스크립트를 실행할 수 있습니다. 대부분의 네이티브 백엔드는 *해석 모드*에서 동작합니다. 즉, 정의된 대로 계산 정의를 처리하고 점진적으로 실행하지만, 항상 그럴 필요는 없습니다. 네이티브 백엔드는 더 나은 성능을 위해 또는 구조를 단순화하기 위해 계산의 일부를 *변환*(*컴파일* 또는 JIT 컴파일)할 수 있습니다. 이를 사용하는 일반적인 예는 계산에 나타나는 페더레이션 연산자 세트를 줄여 변환의 백엔드 다운스트림 부분이 전체 세트에 노출될 필요가 없도록 하는 것입니다.

- **비 네이티브 백엔드**: 비 네이티브 백엔드는 네이티브 백엔드와 달리 TFF 계산 구조를 직접 해석할 수 없으며 백엔드가 이해하는 다른 *대상 표현*으로 변환되어야 합니다. 이러한 백엔드의 주목할 만한 예는 하둡 클러스터 또는 정적 데이터 파이프라인을 위한 유사한 플랫폼입니다. 이러한 백엔드에 계산을 배치하려면 먼저 *변환*(또는 *컴파일*)되어야 합니다. 설정에 따라 사용자에게 투명하게 수행하거나(즉, 비 네이티브 백엔드는 내부에서 변환을 수행하는 [`tff.framework.Executor`](https://www.tensorflow.org/federated/api_docs/python/tff/framework/Executor)와 같은 표준 실행기 인터페이스로 래핑할 수 있음) 사용자가 계산 또는 계산 세트를 특정 백엔드 클래스가 이해하는 적절한 대상 표현으로 수동으로 변환할 수 있는 도구로 노출할 수 있습니다. 특정 유형의 비 네이티브 백엔드를 지원하는 코드는 [`tff.backends`](https://www.tensorflow.org/federated/api_docs/python/tff/backends) 네임스페이스에서 찾을 수 있습니다. 이 글을 쓰는 시점에서, 비 네이티브 백엔드의 유일한 지원 유형은 단일 라운드 MapReduce를 실행할 수 있는 시스템 클래스입니다.

## 네이티브 백엔드

자세한 내용은 곧 제공될 예정입니다.

## 비 네이티브 백엔드

### MapReduce

자세한 내용은 곧 제공될 예정입니다.
