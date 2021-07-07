# 编译

[TOC]

[编译器](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler)软件包中包含用于定义 [AST](#ast) 的 Python 表示形式的数据结构、核心[转换](#transformation)函数和[编译器](#compiler)相关功能。

## AST

TFF 中的抽象语法树 (AST) 描述了联合计算的结构。

### 构建块

[building_block.ComputationBuildingBlock](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py) 是 [AST](#ast) 的 Python 表示形式。

#### `CompiledComputation`

[building_block.CompiledComputation](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py) 是表示将被委托至[外部运行时](execution.md#external-runtime)的计算的 [building_block.ComputationBuildingBlock](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py)。目前，TFF 仅支持 [TensorFlow 计算](#tensorFlow-computation)，但可以扩展以支持由其他外部运行时执行的[计算](#computation)。

### `Computation`

[pb.Computation](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/proto/v0/computation.proto) 是 [AST](#ast) 的 proto 或序列化表示形式。

#### TensorFlow 计算

表示将被委托至 [TensorFlow](execution.md#tensorflow) 运行时的[计算](#computation)的 [pb.Computation](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/proto/v0/computation.proto)。

## 转换

转换可在应用一系列变更后为给定的 AST 构造新的 [AST](#ast)。转换可在[构建块](#building-block)上运行，从而转换 AST 的 Python 表示形式；或在 [TensorFlow 计算](#tensorFlow-computation)上运行，转换 `tf.Graph`。

**原子**转换是一种将单一变更应用于给定输入（可以多次应用）的转换。

**复合**转换是一种将多项转换应用于给定输入以提供某些特征或断言的转换。

注：转换可以采用串行或并行模式，这意味着您可以构造一个复合转换，通过 AST 一次执行多项转换。但是，很难推算出转换的应用顺序以及这些转换的并行方式。因此，复合转换属于人工编制的转换，多数都较为脆弱。

[tree_transformations](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler/tree_transformations.py) 模块中包含原子[构建块](#building-block)转换。

[transformations](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler/transformations.py) 模块中包含复合[构建块](#building-block)转换。

[tensorflow_computation_transformations](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler/tensorflow_computation_transformations.py) 模块中包含原子 [TensorFlow 计算](#tensorflow-computation)转换。

[compiled_computation_transforms](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler/compiled_computation_transforms.py) 模块中包含原子与复合[编译计算](#compiled-computation)转换。

[tree_to_cc_transformations](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/tree_to_cc_transformations.py) 模块中包含表示语法制导定义 (SDD) 逻辑的复合[构建块](#building-block)转换。

[transformation_utils](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler/transformation_utils.py) 模块中包含由其他转换模块使用的函数、遍历逻辑和数据结构。

## 编译器

编译器是用于构造可执行表单的一系列[转换](#transformation)的集合。

### `CompilerPipeline`

[compiler_pipeline.CompilerPipeline](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler/compiler_pipeline.py) 是一种用于编译 [AST](#ast) 并缓存编译结果的数据结构。编译 AST 的性能取决于编译函数的复杂程度；`CompilerPipeline` 可确保多次编译相同的 AST 不会影响系统性能。
