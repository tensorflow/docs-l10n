# Estructura de paquete

[TOC]

## Descripción general

### Terminología

#### Módulo de Python

Un módulo de Python es un archivo que contiene definiciones y declaraciones de Python. Para más detalles, consulte la información sobre [módulos](https://docs.python.org/3/tutorial/modules.html#modules).

#### Paquete de Python

Los paquetes de Python ofrecen una forma de estructurar los módulos de Python. Para más detalles, consulte la información sobre [paquetes](https://docs.python.org/3/tutorial/modules.html#packages).

#### API de TFF pública

La API de TFF es la que se exhibe en la [documentación de las API de TFF](https://www.tensorflow.org/federated/api_docs/python/tff). Esta documentación se genera con [documentos de TensorFlow](https://github.com/tensorflow/docs) aplicando la lógica definida por [explicit_package_contents_filter](https://github.com/tensorflow/docs/blob/master/tools/tensorflow_docs/api_generator/public_api.py;l=156).

#### API de TFF privada

La API de TFF que *no* se exhibe en la [documentación de las API de TFF](https://www.tensorflow.org/federated/api_docs/python/tff).

#### Paquete de Python TFF

El [paquete](https://pypi.org/project/tensorflow-federated/) de Python distribuido en https://pypi.org.

Tenga en cuenta que el paquete de Python contiene tanto la [API pública de TFF](#public-tff-api) como la [privada](#private-tff-api) y que no es evidente, al *inspeccionar el paquete*, cuál de las API se pretende que sea pública y cuál privada. Por ejemplo:

```python
import tensorflow_federated as tff

tff.Computation  # Public TFF API
tff.proto.v0.computation_pb2.Computation  # Private TFF API
```

Por lo tanto, resulta útil tener la [documentación de las API de TTF](https://www.tensorflow.org/federated/api_docs/python/tff) en mente cuando se usa TFF.

### Diagramas

#### Descripción general

```dot
<!--#include file="package_structure_overview.dot"-->
```

#### Simulación

```dot
<!--#include file="package_structure_simulation.dot"-->
```

#### Aprendizaje

```dot
<!--#include file="package_structure_learning.dot"-->
```

#### Análisis

```dot
<!--#include file="package_structure_analytics.dot"-->
```

#### Núcleo

```dot
<!--#include file="package_structure_core.dot"-->
```
