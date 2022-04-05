# コンパイル

[TOC]

[コンパイラ](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler)パッケージには、[AST](#ast)、コア[変換](#transformation)関数、および[コンパイラ](#compiler)関連の機能の Python 表現を定義するデータ構造が含まれます。

## AST

TFF の抽象構文木（AST）は、連合コンピュテーションの構造を説明します。

### ビルディングブロック

[building_block.ComputationBuildingBlock](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) は [AST](#ast) の Python 表現です。

#### `CompiledComputation`

[building_block.CompiledComputation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) は、[外部ランタイム](execution.md#external-runtime)にデリゲートされる計算を表現する [building_block.ComputationBuildingBlock](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) です。現在 TFF は [TensorFlow computations](#tensorFlow-computation) しかサポートしていませんが、ほかの外部ランタイムを使用することで、[Computations](#computation) をサポートするように拡張できる可能性があります。

### `Computation`

[pb.Computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/computation.proto) は [AST](#ast) のプロトまたはシリアル化表現です。

#### TensorFlow Computation

[TensorFlow](execution.md#tensorflow) ランタイムにデリゲートされる [Computations](#computation) を表現する [pb.Computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/computation.proto) です。

## 変換

変換は、一連のミューテーションを適用した後に、特定の AST に新しい [AST](#ast) を構築します。変換は、AST の Python 表現を変換する場合は[ビルディングブロック](#building-block)に対して、または `tf.Graph` を変換する場合は [TensorFlow computations](#tensorFlow-computation) で動作できます。

**アトミック**変換は、特定の入力に単一のミューテーションを（おそらく何度も）適用する変換を指します。

**複合**変換は、特徴量またはアサーションを提供するために、特定の入力に複数の変換を適用する変換です。

注意: 変換は、シリアルまたはパラレルで構成できます。つまり、AST を 1 回通過するだけで複数の変換を実行する複合変換を構築できます。ただし、変換を適用する順序と、それらの変換がどのように並列化されるかについては、根拠づけることが困難です。そのため、複合変換は手動で作成する必要があり、ほとんどがあまり堅牢ではありません。

[tree_transformations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/tree_transformations.py) モジュールには、アトミック[ビルディングブロック](#building-block)変換が含まれます。

[transformations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/transformations.py) モジュールには、複合[ビルディングブロック](#building-block)変換が含まれます。

[tensorflow_computation_transformations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/tensorflow_computation_transformations.py) モジュールには、アトミック [TensorFlow computation](#tensorflow-computation) 変換が含まれます。

[compiled_computation_transforms](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/compiled_computation_transforms.py) モジュールには、アトミックと複合の [Compiled Computation](#compiled-computation) 変換が含まれます。

[transformation_utils](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/transformation_utils.py) モジュールには、他の変換モジュールが使用する関数、トラバーサルロジック、およびデータ構造が含まれます。

## コンパイラ

コンパイラは、実行可能な形態を構築する[変換](#transformation)のコレクションです。

### `CompilerPipeline`

[compiler_pipeline.CompilerPipeline](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/compiler_pipeline.py) は、[AST](#ast) をコンパイルしてコンパイルされたものをキャッシュするデータ構造です。AST のコンパイルパフォーマンスは、コンパイル関数の複雑さに依存し、`CompilerPipeline` によって、同じ AST を何度もコンパイルしてもシステムのパフォーマンスに影響がないように保証されています。
