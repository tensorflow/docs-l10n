# Contexto

[TOC]

## `Context`

Um [context_base.SyncContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) ou [context_base.AsyncContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) é um ambiente que pode [construir](tracing.md), [compilar](compilation.md) ou [executar](execution.md) uma [AST](compilation.md#ast).

A API define uma **abstração de baixo nível** que deve ser usada quando um [executor](execution.md#executor) **não** é usado para a execução; o back-end [Reference](backend.md#reference) (de referência) se integra neste nível.

### `ExecutionContext`

Um [execution_context.ExecutionContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/execution_contexts/execution_context.py) é um [context_base.SyncContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) ou um [context_base.AsyncContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) que compila computações usando uma função de compilação e executa computações usando um [executor](execution.md#executor).

A API define uma **abstração de alto nível** que deve ser usada quando um [executor](execution.md#executor) é usado para a execução; o back-end [native](backend.md#native) (nativo) se integra neste nível.

### `FederatedComputationContext`

Um [federated_computation_context.FederatedComputationContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation_context.py) é um contexto que constrói computações federadas. Esse contexto é usado para fazer o tracing de funções do Python decoradas com o decorador [federated_computation.federated_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation.py) decorator.

### `TensorFlowComputationContext`

Um [tensorflow_computation_context.TensorFlowComputationContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/tensorflow_context/tensorflow_computation_context.py) é um contexto que constrói computações do TensorFlow. Esse contexto é usado para serializar funções do Python decoradas com o decorador [tensorflow_computation.tf_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/tensorflow_context/tensorflow_computation.py).

## `ContextStack`

Una [context_stack_base.ContextStack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py) é uma estrutura de dados para interagir com uma pilha de [contextos](#context).

Você pode chamar o contexto que o TFF usará para [construir](tracing.md), [compilar](compilation.md) ou [executar](execution.md) uma [AST](compilation.md#ast) das seguintes formas:

- Invocando [set_default_context.set_default_context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/set_default_context.py) para definir o contexto padrão. Essa API geralmente é usada para instalar um contexto que compilará ou executará uma computação.

- Invocando [get_context_stack.get_context_stack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/get_context_stack.py) para obter a pilha de contextos atual e depois invocando [context_stack_base.ContextStack.install](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py) para instalar temporariamente um contexto no topo da pilha. Por exemplo, os decoradores [federated_computation.federated_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation.py) e [tensorflow_computation.tf_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/tensorflow_context/tensorflow_computation.py) colocam os contextos correspondentes na pilha de contextos atual enquanto está sendo feito o tracing da função decorada.

### `ContextStackImpl`

Uma [context_stack_impl.ContextStackImpl](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_impl.py) é uma [context_stack_base.ContextStack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py) implementada como uma pilha thread-local comum.
