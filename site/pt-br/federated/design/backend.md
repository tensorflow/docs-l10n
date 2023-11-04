# Back-end

[TOC]

Um back-end é a composição de um [compilador](compilation.md#compiler) e um [runtime](execution.md#runtime) em um [contexto](context.md#context) usado para [construir](tracing.md), [compilar](compilation.md) e [executar](execution.md) uma [AST](compilation.md#ast) (árvore de sintaxe abstrata), ou seja, um back-end constrói ambientes que avaliam uma AST.

O pacote de [back-ends](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends) contém back-ends que podem estender o compilador do TFF e/ou o runtime do TFF. Essas extensões são encontradas no back-end correspondente.

Se o [runtime](execution.md#runtime) de um back-end for implementado como uma [pilha de execução](execution.md#execution-stack), então o back-end pode construir um [ExecutionContext](context.md#executioncontext) (contexto de execução) para fornecer ao TFF um ambiente para avaliar uma AST. Neste caso, o back-end se integra ao TFF usndo a abstração de alto nível. Porém, se o runtime *não* for implementado como uma pilha de execução, então o back-end precisará construir um [contexto](context.md#context) e se integará ao TFF usando a abstração de baixo nível.

```dot
<!--#include file="backend.dot"-->
```

Os nós **azuis** são fornecidos pelo [core](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core) do TFF.

Os nós **verdes**, **vermelhos**, **amarelos** e **roxos** são fornecidos pelos back-ends [native](#native) (nativo), [mapreduce](#mapreduce) (redução do mapa) e [reference](#reference) (de referência), respectivamente.

Os nós **tracejados** são fornecidos por um sistema externo.

As setas **sólidas** indicam o relacionamento, e as setas **tracejadas** indicam a herança.

## Native (Nativo)

O back-end [native](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/native) é composto pelo compilador do TFF e runtime do TFF para compilar e executar uma AST de uma forma que seja razoavelmente eficiente e que possa ser depurada.

### Forma nativa

Uma forma nativa é uma AST que é ordenada topologicamente em um grafo acíclico dirigido (DAG, na sigla em inglês) ou intrínsecos do TFF com algumas otimizações da dependência desses intrínsecos.

### Compilador

A função [compiler.transform_to_native_form](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/native/compiler.py) compila uma AST em uma [forma nativa](#native-form).

### Runtime

O back-end nativo não contém extensões específicas de back-end do runtime do TFF. Em vez disso, uma [pilha de execução](execution.md#execution-stack) pode ser usada diretamente.

### Contexto

Um contexto nativo é um [ExecutionContext](context.md#executioncontext) (contexto de execução) construído com um compilador nativo (ou sem compilador) e o runtime do TFF. Por exemplo:

```python
executor = eager_tf_executor.EagerTFExecutor()
factory = executor_factory.create_executor_factory(lambda _: executor)
context = execution_context.ExecutionContext(
    executor_fn=factory,
    compiler_fn=None)
set_default_context.set_default_context(context)
```

Porém, há algumas configurações comuns:

A função [execution_context.set_sync_local_cpp_execution_context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/native/execution_context.py) constrói um `ExecutionContext` com um compilador nativo e uma [pilha de execução local](execution.md#local-execution-stack).

## MapReduce

O back-end [mapreduce](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce) (redução do mapa) contém as estruturas de dados e o compilador necessários para construir uma forma que possa ser executada em runtimes do tipo MapReduce.

### `MapReduceForm`

Uma [forms.MapReduceForm](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/forms.py) é uma estrutura de dados que define a representação da lógica que pode ser executada em runtimes do tipo MapReduce. Essa lógica é organizada como uma coleção de funções do TensorFlow. Confira mais informações sobre a natureza dessas funções no módulo [forms](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/forms.py) (formas).

### Compilador

O módulo [compiler](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/compiler.py) contém transformações de [bloco de construção](compilation.md#building-block) e [computação do TensorFlow](compilation.md#tensorflow-computation) necessárias para compilar uma AST em uma [MapReduceForm](#canonicalform).

O módulo [form_utils](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/form_utils.py) contém o compilador para o back-end MapReduce e constrói uma [MapReduceForm](#canonicalform).

### Runtime

O runtime do MapReduce não é fornecido pelo TFF. Em vez disso, ele deve ser fornecido pelo sistema externo tipo MapReduce.

### Contexto

O contexto do MapReduce não é fornecido pelo TFF.
