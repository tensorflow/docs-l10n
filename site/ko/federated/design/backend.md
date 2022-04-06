# 백엔드

[TOC]

백엔드는 [AST](compilation.md#ast)를 [구성](tracing.md), [컴파일](compilation.md) 및 [실행](execution.md)하는 데 사용되는 [Context](context.md#context)에서 [컴파일러](compilation.md#compiler)와 [런타임](execution.md#runtime)으로 구성됩니다. 즉, 백엔드가 AST를 평가하는 환경을 구성합니다.

[백엔드](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends) 패키지에는 TFF 컴파일러 및/또는 TFF 런타임을 확장할 수 있는 백엔드가 포함되어 있습니다. 이들 확장은 해당 백엔드에서 찾을 수 있습니다.

백엔드의 [런타임](execution.md#runtime)이 [실행 스택](execution.md#execution-stack)으로 구현되면, 백엔드는 [ExecutionContext](context.md#executioncontext)를 구성하여 AST를 평가할 환경을 TFF에 제공할 수 있습니다. 이 경우 백엔드는 높은 수준의 추상화를 사용하여 TFF와 통합됩니다. 그러나 런타임이 실행 스택으로 구현*되지 않은* 경우, 백엔드는 [컨텍스트](context.md#context)를 구성해야 하며 하위 수준의 추상화를 사용하여 TFF와 통합됩니다.

```dot
<!--#include file="backend.dot"-->
```

**파란색** 노드는 TFF [코어](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core)에서 제공됩니다.

**녹색** , **빨간색** , **노란색** 및 **보라색** 노드는 각각 [native](#native), [mapreduce](#mapreduce), [iree](#iree) 및 [reference](#reference) 백엔드에서 제공됩니다.

**점선** 노드는 외부 시스템에서 제공됩니다.

**실선** 화살표는 관계를 나타내고 **점선** 화살표는 상속을 나타냅니다.

## Native

[native](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/native) 백엔드는 합리적인 수준에서 효율적이고 디버깅 가능한 방식으로 AST를 컴파일하고 실행하기 위해 TFF 컴파일러와 TFF 런타임으로 구성됩니다.

### 네이티브 형식

네이티브 형식은 토폴로지별로 TFF 내장 함수의 방향성 비순환 그래프(DAG)로 정렬된 AST이며, 내장 함수의 종속성에 대한 최적화가 있습니다.

### 컴파일러

[compiler.transform_to_native_form](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/native/compiler.py) 함수는 AST를 [네이티브 형식](#native-form)으로 컴파일합니다.

### 런타임

기본 백엔드에는 TFF 런타임에 대한 백엔드별 확장이 포함되지 않고 대신 [실행 스택](execution.md#execution-stack)을 직접 사용할 수 있습니다.

### 컨텍스트

네이티브 컨텍스트는 네이티브 컴파일러(또는 컴파일러 없음) 및 TFF 런타임으로 구성된 [ExecutionContext](context.md#executioncontext)입니다. 예를 들면, 다음과 같습니다.

```python
executor = eager_tf_executor.EagerTFExecutor()
factory = executor_factory.create_executor_factory(lambda _: executor)
context = execution_context.ExecutionContext(
    executor_fn=factory,
    compiler_fn=None)
set_default_context.set_default_context(context)
```

그러나 몇 가지 일반적인 구성이 있습니다.

[execution_context.set_local_python_execution_context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/native/execution_context.py) 함수는 네이티브 컴파일러와 [로컬 실행 스택](execution.md#local-execution-stack)으로 `ExecutionContext`를 구성합니다.

## MapReduce

[mapreduce](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce) 백엔드에는 MapReduce와 유사한 런타임에서 실행할 수 있는 형식을 구성하는 데 필요한 데이터 구조와 컴파일러가 포함되어 있습니다.

### `MapReduceForm`

[forms.MapReduceForm](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/forms.py)은 MapReduce와 유사한 런타임에서 실행할 수 있는 논리의 표현을 정의하는 데이터 구조입니다. 이 로직은 TensorFlow 함수의 모음으로 구성됩니다. 함수의 특성에 대한 자세한 내용은 [forms](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/forms.py) 모듈을 참조하세요.

### 컴파일러

[transformations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/transformations.py) 모듈에는 AST를 [MapReduceForm](#canonicalform)으로 컴파일하는 데 필요한 [빌딩 블록(Building Block)](compilation.md#building-block) 및 [TensorFlow 계산(Computation)](#canonicalform) 변환이 포함되어 있습니다.

[form_utils](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/form_utils.py) 모듈은 MapReduce 백엔드용 컴파일러를 포함하고 [MapReduceForm](#canonicalform)을 구성합니다.

### 런타임

MapReduce 런타임은 TFF에서 제공하지 않습니다. 대신 외부 MapReduce와 유사한 시스템에서 제공해야 합니다.

### 컨텍스트

MapReduce 컨텍스트는 TFF에서 제공하지 않습니다.

## IREE

[IREE](https://github.com/google/iree)는 [MLIR](https://mlir.llvm.org/)를 위한 실험용 컴파일러 백엔드입니다.

[iree](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/iree) 백엔드에는 AST를 실행하는 데 필요한 데이터 구조, 컴파일러 및 런타임이 포함됩니다.

### 컴파일러

[컴파일러](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/iree/compiler.py) 모듈에는 [executor.IreeExecutor](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/iree/executor.py)를 사용하여 예상할 수 있는 형식으로 AST를 컴파일하는 데 필요한 변환이 포함되어 있습니다.

### 런타임

[executor.IreeExecutor](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/iree/executor.py)는 IREE 런타임에 위임하여 계산을 실행하는 [Executor](execution.md#executor)입니다. 이 실행기는 IREE 런타임을 나타내는 [실행 스택](execution.md#executor)을 구성하기 위해 TFF 런타임의 다른 [실행기](execution.md#execution-stack)와 함께 구성될 수 있습니다.

### 컨텍스트

iree 컨텍스트는 외부 IREE 런타임에 위임하는 [executor.IreeExecutor](context.md#executioncontext)가 있는 [실행 스택](execution.md#execution-stack)과 iree 컴파일러로 구성된 [ExecutionContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/iree/executor.py)입니다.
