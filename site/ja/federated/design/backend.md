# バックエンド

[TOC]

バックエンドは[コンパイラ](compilation.md#compiler)と[ランタイム](execution.md#runtime)の構成で、[コンテキスト](context.md#context)内で AST を[構築](tracing.md)、[コンパイル](compilation.md)、および[実行](execution.md)するために使用されます。つまり、バックエンドは、[AST](compilation.md#ast) を評価する環境を構築します。

[バックエンド](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends)パッケージにはTFF コンパイラや TFF ランタイムを拡張できるバックエンドが含まれます。これらの拡張機能は、対応するバックエンドにあります。

バックエンドの[ランタイム](execution.md#runtime)が[実行スタック](execution.md#execution-stack)として実装されている場合、そのバックエンドは AST を評価する環境を TFF に提供する [ExecutionContext](context.md#executioncontext) を構築できます。この場合、バックエンドは、高位の抽出を使用して TFF に統合していますが、ランタイムが実行スタックとして*実装されていない*場合は、[コンテキスト](context.md#context)を構築する必要があり、低位の抽出を使用して TFF と統合しています。

```dot
<!--#include file="backend.dot"-->
```

**青**のノードは、TFFの[コア](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core)によって提供されます。

**緑**、**赤**、**黄色**、および**紫**のノードはそれぞれ、[native](#native)、[mapreduce](#mapreduce)、[iree](#iree)、および [reference](#reference) バックエンドによって提供されます。

**破線**のノードは、外部システムによって提供されます。

**実線**の矢印は関係を示し、**破線**の矢印は継承を示します。

## Native

[native](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/native) バックエンドは、適度に効率的でデバッグ可能な方法で AST をコンパイルして実行するために、TFF コンパイラと TFF ランタイムで構成されています。

### ネイティブ形態

ネイティブ形態とは、TFF 組み込み関数の有向非巡回グラフ（DAG）にトポロジー的にソートされ、それらの組み込み関数の依存関係にいくつかの最適化が加えられた AST です。

### コンパイラ

[compiler.transform_to_native_form](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/native/compiler.py) 関数は AST を[ネイティブ形態](#native-form)にコンパイルします。

### ランタイム

native バックエンドには TFF ランタイムに対するバックエンド固有の拡張機能が含まれない代わりに、[実行スタック](execution.md#execution-stack)を直接使用できます。

### コンテキスト

native コンテキストは、native コンパイラ（またはコンパイラなし）と TFF ランタイムで構成される [ExecutionContext](context.md#executioncontext) です。以下に例を示します。

```python
executor = eager_tf_executor.EagerTFExecutor()
factory = executor_factory.create_executor_factory(lambda _: executor)
context = execution_context.ExecutionContext(
    executor_fn=factory,
    compiler_fn=None)
set_default_context.set_default_context(context)
```

ただし、共通する構成がいくつかあります。

[execution_context.set_local_python_execution_context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/native/execution_context.py) 関数は、`ExecutionContext` を native コンパイラと[ローカルの実行スタック](execution.md#local-execution-stack)を使って構築します。

## MapReduce

[mapreduce](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce) バックエンドには、MapReduce に似たランタイムで実行できる形態を構築するために必要なデータ構造とコンパイラが含まれます。

### `MapReduceForm`

[forms.MapReduceForm](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/forms.py) は、MapReduce のようなランタイムで実行できるロジックの表現を定義するデータ構造です。このロジックは、TensorFlow 関数のコレクションとして編成されています。これらの関数の性質に関する詳細については、[forms](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/forms.py) モジュールをご覧ください。

### コンパイラ

[transformations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/transformations.py) モジュールには、[ビルディングブロック](compilation.md#building-block)と [TensorFlow Computation](compilation.md#tensorflow-computation) 変換が含まれます。これらは、AST を [MapReduceForm](#canonicalform) に変換するために必要です。

[form_utils](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/form_utils.py) モジュールには MapReduce バックエンドのコンパイラが含まれており、[MapReduceForm](#canonicalform) を構築します。

### ランタイム

MapReduce ランタイムは、TFF によってではなく外部の MapReduce のようなシステムによって提供されます。

### コンテキスト

MapReduce コンテキストは TFF によって提供されません。

## IREE

[IREE](https://github.com/google/iree) は [MLIR](https://mlir.llvm.org/) の実験的なコンパイラバックエンドです。

[iree](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/iree) バックエンドには、AST を実行するために必要なデータ構造、コンパイラ、およびランタイムが含まれます。

### コンパイラ

[compiler](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/iree/compiler.py) モジュールには、[executor.IreeExecutor](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/iree/executor.py) を使用して実行できる形態に AST をコンパイルするために必要な変換が含まれます。

### ランタイム

[executor.IreeExecutor](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/iree/executor.py) は、IREE ランタイムにデリゲートすることで計算を実行する [Executor](execution.md#executor) です。この executor は、TFF ランタイムのほかの [Executor](execution.md#executor) で構成して、IREE ランタイムを表現する[実行スタック](execution.md#execution-stack)を構成することが可能です。

### コンテキスト

iree コンテキストは [ExecutionContext](context.md#executioncontext) で、iree コンパイラと[実行スタック](execution.md#execution-stack)で構成されています。この実行スタックには、外部 IREE ランタイムにデリゲートする [executor.IreeExecutor](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/iree/executor.py) が備わっています。
