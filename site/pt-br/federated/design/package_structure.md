# Estrutura de pacotes

[TOC]

## Visão geral

### Terminologia

#### Módulo do Python

Um módulo do Python é um arquivo que contém definições e declarações do Python. Confira mais informações em [módulos](https://docs.python.org/3/tutorial/modules.html#modules) .

#### Pacote do Python

Os pacotes do Python são uma forma de estruturar os módulos do Python. Confira mais informações em [pacotes](https://docs.python.org/3/tutorial/modules.html#packages) .

#### API pública do TFF

API do TFF que é exposta pela [documentação da API do TFF](https://www.tensorflow.org/federated/api_docs/python/tff). Essa documentação é gerada com o [TensorFlow Docs](https://github.com/tensorflow/docs) usando a lógica definia pelo [explicit_package_contents_filter](https://github.com/tensorflow/docs/blob/master/tools/tensorflow_docs/api_generator/public_api.py;l=156) (filtro explícito de conteúdos do pacote).

#### API privada do TFF

API do TFF que *não* é exposta na [documentação do API do TFF](https://www.tensorflow.org/federated/api_docs/python/tff).

#### Pacote do Python no TFF

[Pacote](https://pypi.org/project/tensorflow-federated/) do Python distribuído em https://pypi.org.

Atenção: o pacote do Python contém tanto a [API pública do TFF](#public-tff-api) quanto a [API privada do TFF](#private-tff-api), e *ao analisar o pacote*, não é óbvio qual API deve ser pública e qual deve ser privada. Por exemplo:

```python
import tensorflow_federated as tff

tff.Computation  # Public TFF API
tff.proto.v0.computation_pb2.Computation  # Private TFF API
```

Portanto, é importante ter em mente a [documentação da API do TFF](https://www.tensorflow.org/federated/api_docs/python/tff) ao usar o TFF.

### Diagramas

#### Visão geral

```dot
<!--#include file="package_structure_overview.dot"-->
```

#### Simulação

```dot
<!--#include file="package_structure_simulation.dot"-->
```

#### Aprendizado

```dot
<!--#include file="package_structure_learning.dot"-->
```

#### Análise

```dot
<!--#include file="package_structure_analytics.dot"-->
```

#### Core

```dot
<!--#include file="package_structure_core.dot"-->
```
