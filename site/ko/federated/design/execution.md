# 실행

[TOC]

[executors](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors) 패키지에는 핵심 [Executors](#executor) 클래스 및 [런타임](#runtime) 관련 기능이 포함되어 있습니다.

## 런타임

런타임은 계산을 실행하는 시스템을 설명하는 논리적 개념입니다.

### TFF 런타임

TFF 런타임은 일반적으로 [AST](compilation.md#ast) 실행을 처리하고 수학적 계산의 실행을 [TensorFlow](#external-runtime)와 같은 [외부 런타임](#tensorflow)에 위임합니다.

### 외부 런타임

외부 런타임은 TFF 런타임이 실행을 위임하는 시스템입니다.

#### TensorFlow

[TensorFlow](https://www.tensorflow.org/)는 머신러닝을 위한 오픈 소스 플랫폼입니다. 오늘날 TFF 런타임은 [실행 스택](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/eager_tf_executor.py)이라고 하는 계층 구조로 구성될 수 있는 [eager_tf_executor.EagerTFExecutor](#execution-stack)를 사용하여 수학적 계산을 TensorFlow에 위임합니다.

## `Executor`

[executor_base.Executor](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_base.py)는 [AST](compilation.md#ast) 실행을 위한 API를 정의하는 추상적인 인터페이스입니다. [executors](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors) 패키지에는 이 인터페이스의 구체적인 구현 모음이 포함되어 있습니다.

## `ExecutorFactory`

[executor_factory.ExecutorFactory](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_factory.py)는 [Executor](#executor) 구성을 위한 API를 정의하는 추상적인 인터페이스입니다. 이들 팩토리는 executor를 느리게 구성하고 executor의 라이프 사이클을 관리합니다. executor를 느리게 구성하는 이유는 실행 시간에 클라이언트의 수를 추론하기 위함입니다.

## 실행 스택

실행 스택은 [Executors](#executor)의 계층입니다. [executor_stacks](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_stacks.py) 모듈은 특정 실행 스택을 구성하고 작성하기 위한 로직을 포함합니다.

### 로컬 실행 스택

[executor_stacks.local_executor_factory](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_stacks.py) 함수는 일부 클라이언트에서 [AST](compilation.md#ast)를 실행하는 로컬 실행 스택을 구성합니다.

### 원격 실행 스택

[executor_stacks.remote_executor_factory](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_stacks.py) 함수는 일부 서비스에서 [AST](compilation.md#ast)를 실행하는 원격 실행 스택을 구성합니다.
