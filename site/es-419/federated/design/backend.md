# Backend

[TOC]

Un <em>backend</em> es una estructura compuesta por un [compilador](compilation.md#compiler) y un [tiempo de ejecución](execution.md#runtime) en un [contexto](context.md#context) usado para [construir](tracing.md), [compilar](compilation.md) y [ejecutar](execution.md) un [AST](compilation.md#ast). Es decir, un <em>backend</em> construye entornos que evalúan a un AST (árbol de sintaxis abstracta).

El paquete de los [backend](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends) contiene varios <em>backend</em> que pueden extender el compilador de TFF o el tiempo de ejecución de TFF. Estas extensiones se pueden hallar en el <em>backend</em> correspondiente.

Si el [tiempo de ejecución](execution.md#runtime) de un <em>backend</em> se implementa como [pila de ejecución](execution.md#execution-stack), entonces, el <em>backend</em> puede construir un [ExecutionContext](context.md#executioncontext) para proporcionarle a TFF un entorno en el cual se pueda evaluar un AST. En este caso, el <em>backend</em> se integra con TFF mediante la abstracción de alto nivel. Sin embargo, si el tiempo de ejecución *no* se implementa como una pila de ejecución, el <em>backend</em> necesitará construir un [contexto](context.md#context) y se integrará con TFF mediante la abstracción de bajo nivel.

```dot
<!--#include file="backend.dot"-->
```

Los nodos en **azul** son proporcionados por el [<em>core</em>](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core) de TFF.

Los nodos **verde**, **rojo**, **amarillo** y **morado** son provistos por los <em>backend</em> [nativo](#native), [mapreduce](#mapreduce) y [de referencia](#reference) respectivamente.

Los nodos **líneas de rayas** son provistos por un sistema externo.

Las flechas **sólidas** indican la relación y las flechas con **líneas de rayas** indican la herencia.

## Nativo

El <em>backend</em> [nativo](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/native) está compuesto por el compilador y el tiempo de ejecución de TFF para compilar y ejecutar un AST de forma tal que sea razonablemente eficiente y posible de depurar.

### Forma nativa

Una forma nativa es un AST que topológicamente está ordenado en un Grafo acíclico dirigido (DAG, por sus siglas en inglés) de valores intrínsecos de TFF con algunas optimizaciones en la dependencia de esos valores intrínsecos.

### Compilador

La función [compiler.transform_to_native_form](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/native/compiler.py) compila un AST en una [forma nativa](#native-form).

### Tiempo de ejecución

El <em>backend</em> nativo no contiene extensiones específicas para el tiempo de ejecución de TFF, en cambio, se puede usar directamente una [pila de ejecución](execution.md#execution-stack).

### Contexto

Un contexto nativo es un [ExecutionContext](context.md#executioncontext) construido con un compilador (o no compilador) nativo y un tiempo de ejecución de TFF, por ejemplo:

```python
executor = eager_tf_executor.EagerTFExecutor()
factory = executor_factory.create_executor_factory(lambda _: executor)
context = execution_context.ExecutionContext(
    executor_fn=factory,
    compiler_fn=None)
set_default_context.set_default_context(context)
```

Sin embargo, hay algunas configuraciones comunes:

La función [execution_context.set_sync_local_cpp_execution_context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/native/execution_context.py) construye un `ExecutionContext` con un compilador nativo y una [pila de ejecución local](execution.md#local-execution-stack).

## MapReduce

El <em>backend</em> [mapreduce](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce) contiene las estructuras de datos y el compilador necesarios para construir una forma que se pueda ejecutar con tiempos de ejecución similares a los de MapReduce.

### `MapReduceForm`

Una [forms.MapReduceForm](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/forms.py) es una estructura de datos que define la representación de lógica que se puede ejecutar en tiempos de ejecución como los de MapReduce. Esta lógica está organizada como una colección de funciones de TensorFlow. Para más información sobre la naturaleza de estas funciones, consulte el módulo de [formas](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/forms.py).

### Compilador

El módulo [compilador](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/compiler.py) contiene las transformaciones del [bloque de construcción](compilation.md#building-block) y de [cálculos de TensorFlow](compilation.md#tensorflow-computation) necesarias para compilar un AST en una [MapReduceForm](#canonicalform).

El módulo [form_utils](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/form_utils.py) contiene el compilador para el <em>backend</em> MapReduce y construye una forma [MapReduceForm](#canonicalform).

### Tiempo de ejecución

TFF no proporciona ningún tiempo de ejecución de MapReduce. Debe proporcionarlo un sistema externo similar a MapReduce.

### Contexto

TFF no proporciona ningún contexto de MapReduce.
