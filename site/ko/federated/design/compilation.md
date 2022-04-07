# 컴파일

[TOC]

[컴파일러](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler) 패키지에는 [AST](#ast)의 Python 표현, 핵심 [transformation](#transformation) 함수 및 [컴파일러](#compiler) 관련 기능을 정의하는 데이터 구조가 포함되어 있습니다.

## AST

TFF의 추상 구문 트리(AST)는 페더레이션 계산의 구조를 설명합니다.

### 빌딩 블록

[building_block.ComputationBuildingBlock](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py)은 [AST](#ast)의 Python 표현입니다.

#### `CompiledComputation`

[building_block.CompiledComputation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py)은 [외부 런타임](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py)에 위임될 계산을 나타내는 [building_block.ComputationBuildingBlock](execution.md#external-runtime)입니다. 현재 TFF는 [TensorFlow 계산](#tensorFlow-computation)만 지원하지만, 다른 외부 런타임에서 지원하는 [Computations](#computation)을 지원하도록 확장할 수 있습니다.

### `Computation`

[pb.Computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/computation.proto)은 [AST](#ast)의 Proto 또는 직렬화된 표현입니다.

#### TensorFlow 계산

[TensorFlow](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/computation.proto) 런타임에 위임될 [Computations](#computation)을 나타내는 [pb.Computation](execution.md#tensorflow)입니다.

## 변환

변환(transformation)은 변경 모음을 적용한 후 주어진 AST의 새 [AST](#ast)를 구성합니다. 변환은 AST의 Python 표현을 변환하기 위해 [빌딩 블록](#building-block)에서 작동하거나 `tf.Graph`를 변환하기 위해 [TensorFlow 계산](#tensorFlow-computation)에서 작동할 수 있습니다.

**원자** 변환은 주어진 입력에 단일 변경(가능하면 두 번 이상)를 적용하는 변환입니다.

**복합** 변환은 일부 특성 또는 어설션을 제공하기 위해 지정된 입력에 여러 변환을 적용하는 변환입니다.

참고: 변환은 직렬 또는 병렬로 구성될 수 있습니다. 즉, AST를 통해 한 번에 여러 변환을 수행하는 복합 변환을 구성할 수 있습니다. 그러나 변환을 적용하는 순서와 이들 변환이 병렬화되는 방식은 추론하기 어렵습니다. 결과적으로, 복합 변형은 수작업으로 이루어지며 대부분은 다소 취약합니다.

[tree_transformations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/tree_transformations.py) 모듈에는 원자 [building block](#building-block) 변환이 포함됩니다.

[transformations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/transformations.py) 모듈에는 복합 [building block](#building-block) 변환이 포함되어 있습니다.

[tensorflow_computation_transformations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/tensorflow_computation_transformations.py) 모듈에는 원자 [TensorFlow 계산](#tensorflow-computation) 변환이 포함되어 있습니다.

[compiled_computation_transforms](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/compiled_computation_transforms.py) 모듈에는 원자 및 복합 [컴파일 계산](#compiled-computation) 변환이 포함되어 있습니다.

[transformation_utils](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/transformation_utils.py) 모듈에는 다른 변환 모듈에서 사용하는 함수, 순회 논리 및 데이터 구조가 포함되어 있습니다.

## 컴파일러

컴파일러는 실행할 수 있는 양식을 구성하는 [transformations](#transformation) 모음입니다.

### `CompilerPipeline`

[compiler_pipeline.CompilerPipeline](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/compiler_pipeline.py)은 [AST](#ast)를 컴파일하고 컴파일된 결과를 캐시하는 데이터 구조입니다. AST를 컴파일하는 성능은 컴파일 함수의 복잡성에 따라 달라집니다. `CompilerPipeline`은 같은 AST를 여러 번 컴파일해도 시스템 성능에 영향을 미치지 않도록 합니다.
