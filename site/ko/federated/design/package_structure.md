# 패키지 구조

[TOC]

## 개요

### 용어

#### Python 모듈

Python 모듈은 Python 정의 및 문이 포함된 파일입니다. 자세한 정보는 [모듈](https://docs.python.org/3/tutorial/modules.html#modules)을 참조하세요.

#### Python 패키지

Python 패키지는 Python 모듈을 구성하는 방법입니다. 자세한 내용은 [패키지](https://docs.python.org/3/tutorial/modules.html#packages)를 참조하십시오.

#### 공개 TFF API

[TFF API 설명서](https://www.tensorflow.org/federated/api_docs/python/tff)에 노출된 TFF API입니다. 이 설명서는 [explicit_package_contents_filter](https://github.com/tensorflow/docs)에 정의된 로직을 사용하여 [TensorFlow 문서](https://github.com/tensorflow/docs/blob/master/tools/tensorflow_docs/api_generator/public_api.py;l=156)와 함께 생성됩니다.

#### 비공개 TFF API

[TFF API 설명서](https://www.tensorflow.org/federated/api_docs/python/tff)에 노출되지 *않은* TFF API입니다.

#### TFF Python 패키지

[PyPI](https://pypi.org/project/tensorflow-federated/)에 배포된 Python [패키지](https://pypi.org)입니다.

Python 패키지에는 [공개 TFF API](#public-tff-api)와 [비공개 TFF API](#private-tff-api)가 모두 포함되어 있으며, 어떤 API가 공개이고 어떤 API가 비공개인지는 *패키지를 검사*해도 불분명합니다. 예를 들면, 다음과 같습니다.

```python
import tensorflow_federated as tff

tff.Computation  # Public TFF API
tff.proto.v0.computation_pb2.Computation  # Private TFF API
```

따라서, TFF를 사용할 때는 [TFF API 설명서](https://www.tensorflow.org/federated/api_docs/python/tff)를 염두에 두는 것이 유용합니다.

### 다이어그램

```dot
<!--#include file="package_structure.dot"-->
```

**녹색** 노드는 [공개 TFF API](https://github.com/tensorflow/federated)를 사용하는 [GitHub](https://github.com)에서 [TFF 리포지토리](https://github.com/tensorflow/federated)의 일부인 디렉토리를 나타냅니다.

**파란색** 노드는 [공개 TFF API](#public-tff-api)의 일부인 패키지를 나타냅니다.

**회색** 노드는 [공개 TFF API](#public-tff-api)의 일부가 아닌 디렉토리 또는 패키지를 나타냅니다.

## 세부 사항

### TFF 사용하기

#### 연구

`research/` 하위 디렉토리가 TFF를 사용하는 연구 프로젝트가 포함 된 [`federated_research`](https://github.com/google-research/federated) 리포지토리로 이동되었습니다.

#### 예제

[examples](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/examples) 디렉토리에는 TFF 사용 방법의 예가 포함되어 있습니다.

#### 테스트

[tests](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/tests) 디렉토리에는 TFF Python 패키지의 엔드 투 엔드 테스트가 포함되어 있습니다.

### TFF

[tff](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/), TensorFlow Federated 라이브러리입니다.

#### TFF 시뮬레이션

[simulation](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/simulation), Federated Learning시뮬레이션을 실행하기 위한 라이브러리입니다.

[simulation/datasets](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/simulation/datasets), Federated Learning 시뮬레이션을 실행하기 위한 데이터세트입니다.

[simulation/models](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/simulation/models), Federated Learning 시뮬레이션을 실행하기 위한 모델입니다.

#### TFF 학습

[learning](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/learning), Federated Learning 알고리즘을 사용하기 위한 라이브러리입니다.

[learning/framework](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/learning/framework), Federated Learning 알고리즘을 개발하기 위한 라이브러리입니다.

#### TFF 애그리게이터

[aggregators](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/aggregators), 페더레이션 집계를 구성하기 위한 라이브러리입니다.

#### TFF 코어

[core](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core) 패키지, TensorFlow Federated 코어 라이브러리입니다.

[core/backends](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/backends), 계산을 구성, 컴파일 및 실행하기 위한 백엔드입니다.

[core/native](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/backends/native), 네이티브 백엔드와 상호 작용하기 위한 라이브러리입니다.

[core/mapreduce](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/backends/mapreduce), MapReduce와 유사한 백엔드와 상호 작용하기 위한 라이브러리입니다.

[core/iree](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/backends/iree), IREE 백엔드와 상호 작용하기 위한 라이브러리입니다.

[core/templates](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/templates), 일반적으로 사용되는 계산을 위한 템플릿입니다.

[core/utils](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/utils), Federated 알고리즘을 사용하고 개발하기 위한 라이브러리입니다.

[core/test](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/test), TensorFlow Federated를 테스트하기 위한 라이브러리입니다.

[core/api](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/api), [TensorFlow Federated 코어 라이브러리](#tff-core)를 사용하기 위한 라이브러리입니다.

[core/framework](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/framework), [TensorFlow Federated 코어 라이브러리](#tff-core)를 확장하기 위한 라이브러리입니다.

#### TFF 구현

[impl](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl), [TensorFlow Federated 코어 라이브러리](#tff-core)의 구현입니다.

TODO(b/148163833): 일부 모듈은 아직 [impl](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl) 패키지에서 적절한 하위 패키지로 이동되지 않았습니다.

[impl/wrappers](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/wrappers), 계산을 구성하기 위한 데코레이터입니다.

[impl/executors](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/executors), 계산을 실행하기 위한 라이브러리입니다.

[impl/federated_context](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/federated_context), * 페더레이션 컨텍스트를 위한 라이브러리입니다.

[impl/tensorflow_context](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/tensorflow_context), * TensorFlow 컨텍스트를 위한 라이브러리입니다.

[impl/computation](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/computation), * 계산을 위한 라이브러리입니다.

[impl/compiler](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler), 계산을 컴파일하기 위한 라이브러리입니다.

[impl/context_stack](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/context_stack), * 계산의 컨텍스트를 위한 라이브러리입니다.

[impl/utils](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/utils), TensorFlow Federated 코어 라이브러리에서 사용하기 위한 라이브러리입니다.

[impl/types](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/types), 계산의 * 유형을 위한 라이브러리입니다.

#### TFF 프로토콜

[proto](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/proto), TensorFlow Federated 코어 라이브러리에서 사용할 수 있는 Protobuf 라이브러리입니다.

#### TFF 공통 라이브러리

[common_libs](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/common_libs), TensorFlow Federated에서 사용하기 위해 Python을 확장하는 라이브러리입니다.

#### TFF Tensorflow 라이브러리

[tensorflow_libs](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/tensorflow_libs), TensorFlow Federated에서 사용하기 위해 TensorFlow를 확장하는 라이브러리입니다.
