# パッケージの構造

[目次]

## 概要

### 用語

#### Python モジュール

Python モジュールは、Python の定義とステートメントを含むファイルです。詳細は、「[モジュール](https://docs.python.org/3/tutorial/modules.html#modules)」を参照してください。

#### Python パッケージ

Python パッケージは、Python モジュールを構造化する手法です。詳細は、「[パッケージ](https://docs.python.org/3/tutorial/modules.html#packages)」を参照してください。

#### パブリック TFF API

[TFF API ドキュメント](https://www.tensorflow.org/federated/api_docs/python/tff)で公開されている TFF API です。このドキュメントは、[explicit_package_contents_filter](https://github.com/tensorflow/docs/blob/master/tools/tensorflow_docs/api_generator/public_api.py;l=156) が定義するロジックを使用して [TensorFlow Docs](https://github.com/tensorflow/docs) によって生成されます。

#### プライベート TFF API

[TFF API ドキュメント](https://www.tensorflow.org/federated/api_docs/python/tff)が*公開していない* TFF API です。

#### TFF Python パッケージ

https://pypi.org で配布されている Python [パッケージ](https://pypi.org/project/tensorflow-federated/)です。

Python パッケージには、[パブリック TFF API](#public-tff-api)と[プライベート TFF API](#private-tff-api)の両方が含まれており、たとえばどの API がパブリックでどれがプライベートかは、*パッケージを調べる*だけではわかりません。

```python
import tensorflow_federated as tff

tff.Computation  # Public TFF API
tff.proto.v0.computation_pb2.Computation  # Private TFF API
```

したがって、TFF を使用する際は、[TFF API ドキュメント](https://www.tensorflow.org/federated/api_docs/python/tff)に留意しておくと有益です。

### ダイアグラム

#### 概要

```dot
<!--#include file="package_structure_overview.dot"-->
```

#### シミュレーション

```dot
<!--#include file="package_structure_simulation.dot"-->
```

#### 学習

```dot
<!--#include file="package_structure_learning.dot"-->
```

#### 分析

```dot
<!--#include file="package_structure_analytics.dot"-->
```

#### コア

```dot
<!--#include file="package_structure_core.dot"-->
```
