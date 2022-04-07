# 컨텍스트

[TOC]

## `Context`

[context_base.Context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py)는 [AST](tracing.md)를 [구성](compilation.md), [컴파일](execution.md) 또는 [실행](compilation.md#ast)할 수 있는 환경입니다.

이 API는 [Executor](execution.md#executor)가 실행에 사용되지 **않을 때** 사용해야 하는 **하위 수준의 추상화**를 정의합니다. [Reference](backend.md#reference) 백엔드는 이 수준에서 통합됩니다.

### `ExecutionContext`

[execution_context.ExecutionContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/execution_contexts/sync_execution_context.py)는 컴파일 함수를 사용하여 계산을 컴파일하고 [Executor](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py)를 사용하여 계산을 실행하는 [context_base.Context](execution.md#executor)입니다.

이 API는 [Executor](execution.md#executor)가 실행에 사용될 때 사용해야 하는 **상위 수준의 추상화**를 정의합니다. [native](backend.md#native) 및 [IREE](backend.md#iree) 백엔드는 이 수준에서 통합됩니다.

### `FederatedComputationContext`

[federated_computation_context.FederatedComputationContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation_context.py)는 페더레이션 계산을 구성하는 [context_base.Context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py)입니다. 이 컨텍스트는 [computations.federated_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/api/computations.py) 데코레이터로 데코레이팅된 Python 함수를 추적하는 데 사용됩니다.

### `TensorFlowComputationContext`

[tensorflow_computation_context.TensorFlowComputationContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/tensorflow_context/tensorflow_computation_context.py)는 TensorFlow 계산을 구성하는 [context_base.Context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py)입니다. 이 컨텍스트는[computations.tf_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/api/computations.py) 데코레이터로 데코레이팅된 Python 함수를 직렬화하는 데 사용됩니다.

## `ContextStack`

[context_stack_base.ContextStack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py)은 [Context](#context) 스택과 상호 작용하기 위한 데이터 구조입니다.

TFF가 [AST](execution.md)를 [구성](compilation.md#ast), [컴파일](tracing.md) 또는 [실행](compilation.md)하는 데 사용할 컨텍스트를 다음과 같이 설정할 수 있습니다.

- [set_default_context.set_default_context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/set_default_context.py)를 호출하여 기본 컨텍스트를 설정합니다. 이 API는 종종 계산을 컴파일하거나 실행할 컨텍스트를 설치하는 데 사용됩니다.

- [get_context_stack.get_context_stack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/get_context_stack.py)을 호출하여 현재 컨텍스트 스택을 가져온 다음 [context_stack_base.ContextStack.install](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py)을 호출하여 스택의 맨 위에 컨텍스트를 임시로 설치합니다. 예를 들어, [calculations.federated_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/api/computations.py) 및 [calculations.tf_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/api/computations.py) 데코레이터는 데코레이팅된 함수가 추적되는 동안 해당 컨텍스트를 현재 컨텍스트 스택으로 푸시합니다.

### `ContextStackImpl`

[context_stack_impl.ContextStackImpl](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_impl.py)은 공통 스레드 로컬 스택으로 구현되는 [context_stack_base.ContextStack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py)입니다.
