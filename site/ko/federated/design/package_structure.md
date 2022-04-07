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
