# Context

[TOC]

## `Context`

A [context_base.SyncContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) or [context_base.AsyncContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) is an environment that can [construct](tracing.md), [compile](compilation.md), or [execute](execution.md) an [AST](compilation.md#ast).

この API は、[Executor](execution.md#executor) が実行に**使用されない**場合に使用される **low-level abstraction** を定義します。バックエンドの [Reference](backend.md#reference) はこのレベルで統合されます。

### `ExecutionContext`

An [execution_context.ExecutionContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/execution_contexts/execution_context.py) is [context_base.SyncContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) or [context_base.AsyncContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) that compiles computations using a compilation function and executes computations using an [Executor](execution.md#executor).

この API は、[Executor](execution.md#executor) が実行に使用される時に使用される **高レベルの抽象化**を定義します。[native](backend.md#native) はこのレベルで統合されます。

### `FederatedComputationContext`

A [federated_computation_context.FederatedComputationContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation_context.py) is a context that constructs federated computations. This context is used trace Python functions decorated with the [federated_computation.federated_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation.py) decorator.

### `TensorFlowComputationContext`

A [tensorflow_computation_context.TensorFlowComputationContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/tensorflow_context/tensorflow_computation_context.py) is a context that constructs TensorFlow computations. This context is used to serialize Python functions decorated with the [tensorflow_computation.tf_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/tensorflow_context/tensorflow_computation.py) decorator.

## `ContextStack`

[context_stack_base.ContextStack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py) は[Contexts](#context) スタックを操作するためのデータ構造です。

TFF が[構築](tracing.md)、[コンパイル](compilation.md)、または[実行](execution.md)するために使用するコンテキストは、以下のようにして設定できます。

- [set_default_context.set_default_context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/set_default_context.py) を呼び出してデフォルトのコンテキストを設定します。この API は通常、計算をコンパイルまたは実行するコンテキストをインストールする際に使用されます。

- [get_context_stack.get_context_stack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/get_context_stack.py) を呼び出して現在のコンテキストスタックを取得し、[context_stack_base.ContextStack.install](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py) を呼び出してコンテキストをスタックの上に一時的にインストールします。たとえば、[federated_computation.federated_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation.py) と [tensorflow_computation.tf_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/tensorflow_context/tensorflow_computation.py) デコレータは、デコレートされた関数がトレースされている間に、対応するコンテキストを現在のコンテキストスタックにプッシュします。

### `ContextStackImpl`

[context_stack_impl.ContextStackImpl](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_impl.py) は、一般的なスレッドローカルスタックとして実装される [context_stack_base.ContextStack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py) です。
