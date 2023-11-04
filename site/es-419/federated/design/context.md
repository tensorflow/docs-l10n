# Contexto

[TOC]

## `Context`

Un [context_base.SyncContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) o [context_base.AsyncContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) es un entorno que puede [construir](tracing.md), [compilar](compilation.md) o [ejecutar](execution.md) un [AST](compilation.md#ast).

Con esta API se define una **abstracción de bajo nivel** que se debería usar cuando un [ejecutor](execution.md#executor) **no** se usa para la ejecución. El <em>backend</em> de [referencia](backend.md#reference) se integra en este nivel.

### `ExecutionContext`

Un [execution_context.ExecutionContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/execution_contexts/execution_context.py) es un [context_base.SyncContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) o un [context_base.AsyncContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) que compila los cálculos con una función de compilación y ejecuta los cálculos con un [ejecutor](execution.md#executor).

Con esta API se define una **abstracción de alto nivel** que se debería usar cuando se usa un [ejecutor](execution.md#executor) para la ejecución. El [nativo](backend.md#native) se integra en este nivel.

### `FederatedComputationContext`

Un [federated_computation_context.FederatedComputationContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation_context.py) es un contexto que construye cálculos federados. Este contexto se usa con funciones Python de rastreo decoradas con el decorador [federated_computation.federated_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation.py).

### `TensorFlowComputationContext`

Un [tensorflow_computation_context.TensorFlowComputationContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/tensorflow_context/tensorflow_computation_context.py) es un contexto que construye cálculos de TensorFlow. Este contexto se usa para serializar funciones Python decoradas con el decorador [tensorflow_computation.tf_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/tensorflow_context/tensorflow_computation.py).

## `ContextStack`

Un [context_stack_base.ContextStack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py) es una estructura de datos que se usa para interactuar con una pila de [contextos](#context).

Se puede configurar el contexto que usará TFF para [construir](tracing.md), [compilar](compilation.md) o [ejecutar](execution.md) un [AST](compilation.md#ast) mediante lo siguiente:

- Invocando a [set_default_context.set_default_context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/set_default_context.py) para establecer el contexto predeterminado. Por lo común, esta API se usa para instalar un contexto que compilará o ejecutará un cálculo.

- Invocando [get_context_stack.get_context_stack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/get_context_stack.py) para obtener la pila de contextos actuales, invocando después [context_stack_base.ContextStack.install](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py) para instalar temporalmente un contexto sobre la parte superior de la pila. Por ejemplo, los decoradores [federated_computation.federated_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation.py) y [tensorflow_computation.tf_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/tensorflow_context/tensorflow_computation.py) empuja e instala los contextos correspondientes sobre la pila de contextos actuales mientras se rastrea la función decorada.

### `ContextStackImpl`

Un [context_stack_impl.ContextStackImpl](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_impl.py) es un [context_stack_base.ContextStack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py) que se implementa como una pila de hilo (<em>thread</em>) local común.
