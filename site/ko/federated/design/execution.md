# 실행

[TOC]

[executors](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/executors) 패키지에는 핵심 [Executors](#executor) 클래스 및 [런타임](#runtime) 관련 기능이 포함되어 있습니다.

## Runtime

런타임은 계산을 실행하는 시스템을 설명하는 논리적 개념입니다.

### TFF 런타임

A TFF runtime typically handles executing an [AST](compilation.md#ast) and delegates executing mathematical computations to a [external runtime](#external-runtime) such as [TensorFlow](#tensorflow).

### 외부 런타임

외부 런타임은 TFF 런타임이 실행을 위임하는 시스템입니다.

#### TensorFlow

[TensorFlow](https://www.tensorflow.org/) is an open source platform for machine learning. Today the TFF runtime delegates mathematical computations to TensorFlow using an [eager_tf_executor.EagerTFExecutor](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/executors/eager_tf_executor.py) that can be composed into a hierarchy, referred to as an [execution stack](#execution-stack).

## `Executor`

An [executor_base.Executor](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/executors/executor_base.py) is an abstract interface that defines the API for executing an [AST](compilation.md#ast). The [executors](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/executors) package contains a collection of concrete implementations of this interface.

## `ExecutorFactory`

An [executor_factory.ExecutorFactory](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/executors/executor_factory.py) is an abstract interface that defines the API for constructing an [Executor](#executor). These factories construct the executor lazily and manage the lifecycle of the executor; the motivation to lazily constructing executors is to infer the number of clients at execution time.

## 실행 스택

An execution stack is a hierarchy of [Executors](#executor). The [executor_stacks](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/executors/executor_stacks.py) module contains logic for constructing and composing specific execution stacks.

### 로컬 실행 스택

The [executor_stacks.local_executor_factory](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/executors/executor_stacks.py) function constructs a local execution stack that executes an [AST](compilation.md#ast) on some number of clients.

### 원격 실행 스택

The [executor_stacks.remote_executor_factory](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/executors/executor_stacks.py) function constructs a remote execution stack that executes an [AST](compilation.md#ast) on some service.
