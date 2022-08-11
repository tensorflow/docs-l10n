# 将 TensorFlow RNN 转换为 TensorFlow Lite

## 概述

TensorFlow Lite 支持将 TensorFlow RNN 模型转换为 TensorFlow Lite 的融合 LSTM 运算。融合运算的存在是为了最大限度地提高其底层内核实现的性能，同时也提供了一个更高级别的接口来定义如量化之类的复杂转换。

由于 TensorFlow 中 RNN API 的变体很多，我们的转换方式包括两个方面：

1. **为标准 TensorFlow RNN API（如 Keras LSTM）提供原生支持**。这是推荐的选项。
2. 提供**进入转换基础架构**的**接口**，用于插入**用户定义的** **RNN 实现**并转换为 TensorFlow Lite。我们提供了几个有关此类转换的开箱即用的示例，这些示例使用的是 lingvo 的 [LSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/82abf0dbf316526cd718ae8cd7b11cfcb805805e/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L130) 和 [LayerNormalizedLSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L137) RNN 接口。

## 转换器 API

该功能是 TensorFlow 2.3 版本的一部分。它也可以通过 [tf-nightly](https://pypi.org/project/tf-nightly/) pip 或从 head 获得。

当通过 SavedModel 或直接从 Keras 模型转换到 TensorFlow Lite 时，可使用此转换功能。请参阅用法示例。

### 从保存的模型

<a id="from_saved_model"></a>

```
# build a saved model. Here concrete_function is the exported function
# corresponding to the TensorFlow model containing one or more
# Keras LSTM layers.
saved_model, saved_model_dir = build_saved_model_lstm(...)
saved_model.save(saved_model_dir, save_format="tf", signatures=concrete_func)

# Convert the model.
converter = TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
```

### 从 Keras 模型

```
# build a Keras model
keras_model = build_keras_lstm(...)

# Convert the model.
converter = TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

```

## 示例

Keras LSTM 到 TensorFlow Lite [Colab](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/experimental_new_converter/Keras_LSTM_fusion_Codelab.ipynb) 说明了 TensorFlow Lite 解释器的端到端用法。

## 支持的 TensorFlow RNN API

<a id="rnn_apis"></a>

### Keras LSTM 转换（推荐）

我们支持 Keras LSTM 到 TensorFlow Lite 的开箱即用的转换。有关工作原理的详细信息，请参阅 [Keras LSTM 接口](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/python/keras/layers/recurrent_v2.py#L1238)<span style="text-decoration:space;"></span>和[转换逻辑](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/compiler/mlir/lite/utils/lstm_utils.cc#L627)。

另外，强调与 Keras 运算定义相关的 TensorFlow Lite 的 LSTM 协定也很重要：

1. **input** 张量的维度 0 是批次大小。
2. **recurrent_weight** 张量的维度 0 是输出的数量。
3. **weight** 和 **recurrent_kernel** 张量进行了转置。
4. 转置后的 weight 张量、转置后的  recurrent_kernel 张量，以及 **bias** 张量沿着维度 0 被拆分成了 4 个大小相等的张量。这些张量分别对应 **input gate、forget gate、cell 和 output gate**。

#### Keras LSTM 变体

##### 时间为主

用户可以选择 time-major 或非 time-major。Keras LSTM 在函数 def attributes 中增加了一个 time-major 特性。对于单向序列 LSTM，我们可以简单地映射到 unidirecional_sequence_lstm 的[ time major 特性](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/compiler/mlir/lite/ir/tfl_ops.td#L3902)中。

##### 双向 LSTM

双向 LSTM 可以用两个 Keras LSTM 层来实现，一个是前向 LSTM 层，一个是后向 LSTM 层，请参阅[此处](https://tensorflow.google.cn/lite/guide/ops_custom#converting_tensorflow_models_to_convert_graphs)。当我们看到 go_backward 特性后，就把它识别为后向 LSTM，然后我们将前向和后向 LSTM 归为一组。**这是未来的工作**。目前，这会在 TensorFlow Lite 模型中创建两个 UnidirectionalSequenceLSTM 运算。

### 用户定义的 LSTM 转换示例

TensorFlow Lite 还提供了一种转换用户定义的 LSTM 实现的方式。在这里，我们以 Lingvo 的 LSTM 为例来说明如何实现。有关详细信息，请参阅 [lingvo.LSTMCellSimple 接口](https://github.com/tensorflow/lingvo/blob/91a4609dbc2579748a95110eda59c66d17c594c5/lingvo/core/rnn_cell.py#L228)和[转换逻辑](https://github.com/tensorflow/tensorflow/blob/82abf0dbf316526cd718ae8cd7b11cfcb805805e/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L130)。我们还在 [lingvo.LayerNormalizedLSTMCellSimple 接口](https://github.com/tensorflow/lingvo/blob/91a4609dbc2579748a95110eda59c66d17c594c5/lingvo/core/rnn_cell.py#L1173)及其[转换逻辑](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L137)中提供了 Lingvo 的 LSTM 定义的另一个示例。

## “将您自己的 TensorFlow RNN”转换为 TensorFlow Lite

如果用户的 RNN 接口与支持的标准接口不同，有如下几个选项：

**选项 1**：在 TensorFlow Python 中编写适配器代码，以使 RNN 接口适配 Keras RNN 接口。这意味着在生成的 RNN 接口函数上具有 [tf_implements 注解](https://github.com/tensorflow/community/pull/113)的 tf.function 与 Keras LSTM 层生成的相同。此后，将使用与 Keras LSTM 所用相同的转换 API。

**选项 2**：如果上述操作不可行（例如 Keras LSTM 缺少某些目前由 TensorFlow Lite 的融合 LSTM 运算公开的功能，如层归一化），则通过编写自定义转换代码来扩展 TensorFlow Lite 转换器并将其插入[此处的](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L115) prepare-composite-functions MLIR-pass 中。函数的接口应被视为 API 协定，并应包含转换为融合 TensorFlow Lite LSTM 运算所需的参数，即输入、偏差、权重、投影、层归一化等。作为参数传递给该函数的张量最好具有已知的秩（即 MLIR 中的 RankedTensorType）。这使编写转换代码变得更加容易，可以将这些张量假定为 RankedTensorType，并帮助将它们转换为对应于融合 TensorFlow Lite 算子的运算对象的有秩张量。

这种转换流程的一个完整例子是 Lingvo 的 LSTMCellSimple 到 TensorFlow Lite 的转换。

[此处](https://github.com/tensorflow/lingvo/blob/91a4609dbc2579748a95110eda59c66d17c594c5/lingvo/core/rnn_cell.py#L228)定义了 Lingvo 中的 LSTMCellSimple。用这个 LSTM 单元训练的模型可以按照以下方式转换为 TensorFlow Lite：

1. 将 LSTMCellSimple 的所有用法都用 tf_implements 注解封装在 tf.function 中，并标注为 tf_implements 注解（例如这里 lingvo.LSTMCellSimple 是一个很好的注解名称）。确保生成的 tf.function 与转换代码中预期的函数接口相匹配。这是添加注解的模型作者和转换代码之间的协定。

2. 扩展 prepare-composite-functions 传递，以插入从自定义复合运算到 TensorFlow Lite 融合 LSTM 运算的转换。请参阅 [LSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/82abf0dbf316526cd718ae8cd7b11cfcb805805e/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L130) 转换代码。

    转换协定：

3. **Weight** 和 **projection** 张量进行了转置。

4. 通过对转置后的权重张量进行切分，提取出从 **{input, recurrent}** 到 **{cell, input gate, forget gate, output gate}**。

5. 通过对偏差张量进行切分，提取出从 **{bias}** 到 **{cell, input gate, forget gate, output gate}**。

6. 通过切分转置后的投影张量提取出 **projection**。

7. 为 [LayerNormalizedLSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L137) 编写了类似的转换。

8. TensorFlow Lite 转换基础架构的其余部分，包括所有已定义的 [MLIR 传递](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/compiler/mlir/lite/tf_tfl_passes.cc#L57)以及最终导出到 TensorFlow Lite 平面缓冲区的部分都可以重用。

## 已知问题/限制

1. 目前只支持转换无状态的 Keras LSTM（Keras 的默认行为）。有状态的 Keras LSTM 转换是未来的工作。
2. 仍然可以使用底层的无状态 Keras LSTM 层对有状态的 Keras LSTM 层进行建模，并在用户程序中显式管理状态。仍然可以使用此处描述的功能将此类 TensorFlow 程序转换为 TensorFlow Lite。
3. 双向 LSTM 目前在 TensorFlow Lite 中被建模为两个 UnidirectionalSequenceLSTM 运算。将由单个 BidirectionalSequenceLSTM 运算代替。
