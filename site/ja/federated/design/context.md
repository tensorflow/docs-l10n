# コンテキスト

[TOC]

## `Context`

[context_base.Context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) は [AST](compilation.md#ast) を[構築](tracing.md)、[コンパイル](compilation.md)、または[実行](execution.md)できる環境です。

この API は、[Executor](execution.md#executor) が実行に**使用されない**場合に使用される **low-level abstraction** を定義します。バックエンドの [Reference](backend.md#reference) はこのレベルで統合されます。

### `ExecutionContext`

[execution_context.ExecutionContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/execution_contexts/sync_execution_context.py) は、コンパイル関数を使用して計算をコンパイルし、[Executor](execution.md#executor) を使用して計算を実行する [context_base.Context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) です。

この API は、[Executor](execution.md#executor) が実行に使用される場合に使用される **high-level abstraction** を定義します。[native](backend.md#native) と [IREE](backend.md#iree) のバックエンドはこのレベルで統合されます。

### `FederatedComputationContext`

[federated_computation_context.FederatedComputationContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation_context.py) は連合コンピュテーションを構築する [context_base.Context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) です。このコンテキストは、[computations.federated_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/api/computations.py) デコレータが使用された Python 関数をトレースするために使用されます。

### `TensorFlowComputationContext`

[tensorflow_computation_context.TensorFlowComputationContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/tensorflow_context/tensorflow_computation_context.py) は TensorFlow computations を構築する [context_base.Context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) です。このコンテキストは、[computations.tf_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/api/computations.py) デコレータが使用された Python をシリアル化するために使用されます。

## `ContextStack`

[context_stack_base.ContextStack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py) は[Contexts](#context) スタックを操作するためのデータ構造です。

TFF が[構築](tracing.md)、[コンパイル](compilation.md)、または[実行](execution.md)するために使用するコンテキストは、以下のようにして設定できます。

- [set_default_context.set_default_context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/set_default_context.py) を呼び出してデフォルトのコンテキストを設定します。この API は通常、計算をコンパイルまたは実行するコンテキストをインストールする際に使用されます。

- [get_context_stack.get_context_stack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/get_context_stack.py) を呼び出して現在のコンテキストスタックを取得し、[context_stack_base.ContextStack.install](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py) を呼び出してコンテキストをスタックの上に一時的にインストールします。たとえば、[computations.federated_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/api/computations.py) と [computations.tf_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/api/computations.py) デコレータは、デコレートされた関数がトレースされている間に、対応するコンテキストを現在のコンテキストスタックにプッシュします。

### `ContextStackImpl`

[context_stack_impl.ContextStackImpl](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_impl.py) は、一般的なスレッドローカルスタックとして実装される [context_stack_base.ContextStack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py) です。
