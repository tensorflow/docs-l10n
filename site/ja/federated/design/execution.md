# 実行

[TOC]

[executors](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors) パッケージには、コア [Executors](#executor) クラスとと[ランタイム](#runtime)関連の機能が含まれます。

## ランタイム

ランタイムは、計算を実行するシステムを記述する論理的概念です。

### TFF ランタイム

TFF ランタイムは通常、[AST](compilation.md#ast) の実行を処理し、数学的計算の実行を [TensorFlow](#tensorflow) などの[外部ランタイム](#external-runtime)にデリゲートします。

### 外部ランタイム

外部ランタイムは、TFF ランタイムが実行をデリゲートする先のシステムです。

#### TensorFlow

[TensorFlow](https://www.tensorflow.org/) は機械学習用のオープンソースプラットフォームです。今日、TFF ランタイムは、[実行スタック](#execution-stack)と呼ばれる階層に構成できる [eager_tf_executor.EagerTFExecutor](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/eager_tf_executor.py) を使用して、数学的計算を TensorFlow にデリゲートしています。

## `Executor`

[executor_base.Executor](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_base.py) は、[AST](compilation.md#ast) を実行するための API を定義する抽象インターフェースです。[executors](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors) パッケージには、このインターフェースの具体的な実装のコレクションが含まれます。

## `ExecutorFactory`

[executor_factory.ExecutorFactory](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_factory.py) は、[Executor](#executor) を構築するための API を定義する抽象インターフェースです。これらのファクトリーは Executor を遅延的に構築し、その Executor のライフサイクルを管理します。Executor を遅延構築するのは、実行時にクライアント数を推論するためです。

## 実行スタック

実行スタックは、[Executor](#executor) の階層です。[executor_stacks](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_stacks.py) モジュールには、特定の実行スタックを構築・作成するためのロジックが含まれます。

### ローカル実行スタック

[executor_stacks.local_executor_factory](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_stacks.py) 関数は、クライアントで [AST](compilation.md#ast) を実行するローカル実行スタックを構築します。

### リモート実行スタック

[executor_stacks.remote_executor_factory](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_stacks.py) 関数は、サービスで [AST](compilation.md#ast) を実行するリモート実行スタックを構築します。
