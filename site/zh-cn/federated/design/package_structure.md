# 软件包结构

[目录]

## 概述

### 术语

#### Python 模块

Python 模块是一个包含 Python 定义和语句的文件。请参阅[模块](https://docs.python.org/3/tutorial/modules.html#modules)，了解更多信息。

#### Python 软件包

Python 软件包是一种构建 Python 模块的方法。请参阅[软件包](https://docs.python.org/3/tutorial/modules.html#packages)，了解更多信息。

#### 公共 TFF API

[TTFF API 文档](https://www.tensorflow.org/federated/api_docs/python/tff)公开的 TFF API；本文档使用 [TensorFlow Docs](https://github.com/tensorflow/docs) 通过由 [explicit_package_contents_filter](https://github.com/tensorflow/docs/blob/master/tools/tensorflow_docs/api_generator/public_api.py;l=156) 定义的逻辑生成

#### 私有 TFF API

TFF [TFF API 文档](https://www.tensorflow.org/federated/api_docs/python/tff)中*未*公开的TFF API。

#### TFF Python 软件包

在 [PyPI](https://pypi.org) 上分发的 Python [软件包](https://pypi.org/project/tensorflow-federated/)。

请注意，Python 软件包同时包含[公共 TFF API](#public-tff-api) 和[私有 TFF API](#private-tff-api)，*检查软件包*无法明确哪些 API 是公共的，哪些是私有的，例如：

```python
import tensorflow_federated as tff

tff.Computation  # Public TFF API
tff.proto.v0.computation_pb2.Computation  # Private TFF API
```

因此，在使用 TFF 时维护 [TFF API 文档](https://www.tensorflow.org/federated/api_docs/python/tff)十分有用。

### 图表

```dot
<!--#include file="package_structure.dot"-->
```

**绿色**节点表示属于 [GitHub](https://github.com) 上 [TFF 仓库](https://github.com/tensorflow/federated)并使用[公共 TFF API](#public-tff-api) 的目录。

**蓝色**节点表示属于[公共 TFF API](#public-tff-api) 的软件包。

**灰色**节点表示不属于[公共 TFF API](#public-tff-api) 的目录或软件包。
