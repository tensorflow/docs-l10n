# TensorFlow 运算融合

## 概述

本文介绍了将 TensorFlow 中的复合运算转换为 TensorFlow Lite 中的融合运算所需的设计和步骤。此基础架构是通用的，支持将 TensorFlow 中的任何复合运算转换为 TensorFlow Lite 中的相应融合运算。

An example use of this infrastructure is TensorFlow RNN operation fusion to TensorFlow Lite, as detailed [here](https://www.tensorflow.org/lite/models/convert/rnn).

### 什么是融合运算

![drawing](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/lite/images/convert/op_fusion_banner.jpg?raw=true)

TensorFlow 运算既可以是基元运算（例如 [tf.add](https://www.tensorflow.org/api_docs/python/tf/math/add)），也可以由其他基元运算（例如 [tf.einsum](https://www.tensorflow.org/api_docs/python/tf/einsum)）组成。基元运算在 TensorFlow 计算图中显示为单个节点，而复合运算则是 TensorFlow 计算图中节点的集合。执行复合运算相当于执行组成该复合运算的每个基元运算。

融合运算对应于这样一种运算：将每个基元运算执行的所有计算都纳入相应的复合运算中。

### 融合运算的好处

通过优化整体计算并减少内存占用，融合运算可以最大程度地提高其底层内核实现的性能。这非常有价值，特别适合低延迟推理工作负载和资源受限的移动平台。

融合运算还提供了一个更高级别的接口来定义像量化一样的复杂转换，如果不使用融合运算，便无法或很难在更细粒度的级别上实现这种转换。

出于上述原因，TensorFlow Lite 中具有许多融合运算的实例。这些融合运算通常对应于源 TensorFlow 程序中的复合运算。TensorFlow 中的复合运算在 TensorFlow Lite 中以单个融合运算的形式实现，示例包括各种 RNN 运算，如单向和双向序列 LSTM、卷积（conv2d、bias add、relu）、全连接（matmul、bias add、relu）等。在 TensorFlow Lite 中，LSTM 量化目前仅在 LSTM 融合运算中实现。

### 融合运算面临的挑战

将 TensorFlow 中的复合运算转换为 TensorFlow Lite 中的融合运算是一个难题。原因如下：

1. TensorFlow 计算图中的复合运算表示为一组没有明确定义边界的基元运算。识别（例如，通过模式匹配）与这种复合运算对应的子计算图极具挑战。

2. 可能有多个 TensorFlow 实现将某个 TensorFlow Lite 融合运算作为目标。例如，TensorFlow 中有许多 LSTM 实现（Keras、Babelfish/lingvo 等），每个实现都由不同的基元运算组成，但它们仍然可转换为 TensorFlow Lite 中的相同 LSTM 融合运算。

因此，融合运算的转换已被证实非常困难。

## Converting from composite op to a TFLite custom operation (recommended)

### 将复合运算包装在 `tf.function` 中

In many cases, some part of the model can be mapped to a single operation in TFLite. This can help with performance when writing an optimized implementation for specific operations. To be able to create a fused operation in TFLite, identify the part of the graph that represents a fused operation and wrap it in a `tf.function` with "experimental_implements" attribute to a `tf.function`, which has attribute value `tfl_fusable_op` with value `true`. If the custom operation takes attributes then pass them as part of the same "experimental_implements".

Example,

```python
def get_implements_signature():
  implements_signature = [
    # 'name' will be used as a name for the operation.
    'name: "my_custom_fused_op"',
    # attr "tfl_fusable_op" is required to be set with true value.
    'attr {key: "tfl_fusable_op" value { b: true } }',
    # Example attribute "example_option" that the op accepts.
    'attr {key: "example_option" value { i: %d } }' % 10
  ]
  return ' '.join(implements_signature)

@tf.function(experimental_implements=get_implements_signature())
def my_custom_fused_op(input_1, input_2):
  # An empty function that represents pre/post processing example that
  # is not represented as part of the Tensorflow graph.
  output_1 = tf.constant(0.0, dtype=tf.float32, name='first_output')
  output_2 = tf.constant(0.0, dtype=tf.float32, name='second_output')
  return output_1, output_2

class TestModel(tf.Module):
  def __init__(self):
    super(TestModel, self).__init__()
    self.conv_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3))
    self.conv_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3))

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[1, 28, 28, 3], dtype=tf.float32),
      tf.TensorSpec(shape=[1, 28, 28, 3], dtype=tf.float32),
  ])
  def simple_eval(self, input_a, input_b):
    return my_custom_fused_op(self.conv_1(input_a), self.conv_2(input_b))
```

Note that you don't need to set `allow_custom_ops` on the converter as `tfl_fusable_op` attribute imply this already.

### Implement custom op and register with TFLite Interpreter

Implement your fused operation as a TFLite Custom operation - see [instructions](https://www.tensorflow.org/lite/guide/ops_custom).

Note that, the name to register the op with should be similar to the name specified in the `name` attribute in the implements signature.

An example for the op in the example is

```c++
  TfLiteRegistration reg;
  // This name must match the name specified in the implements signature.
  static constexpr char kOpName[] = "my_custom_fused_op";
  reg.custom_name = kOpName;
  reg.prepare = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
    // Add your code.
    return kTfLiteOk;
  };
  reg.invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
    // Add your coder.
    return kTfLiteOk;
  };
  reg.builtin_code = kTfLiteCustom;
  resolver->AddCustom(kOpName, &reg);
```

## Converting from composite to fused operation (Advanced)

将 TensorFlow 复合运算转换为 TensorFlow Lite 融合运算的整体架构如下：

![drawing](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/lite/images/convert/op_fusion.png?raw=true)

### Wrap the composite operation in a `tf.function`

在 TensorFlow 模型源代码中，使用 [experimental_implements](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/python/eager/def_function.py#L470) 注解识别复合运算并将其抽象为 `tf.function`。请参见[嵌入向量查找](#composing_ops)的示例。该函数定义了接口，其参数应当用于实现转换逻辑。

### 编写转换代码

使用 `implements` 注解为函数的接口编写转换代码。请参见[嵌入向量查找](#fusion_code)的融合示例。从概念上讲，转换代码用融合实现替代了此接口的复合实现。

在“准备-复合-函数”传递中，插入[转换代码](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L115)。

在更高级的用法中，可以实现复合运算的运算对象的复杂转换，以便派生融合运算的运算对象。请参见 [Keras LSTM](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/lite/utils/lstm_utils.cc#L627) 转换代码示例。

### 转换为 TensorFlow Lite

使用 [TFLiteConverter.from_saved_model](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#from_saved_model) API 转换为 TensorFlow Lite。

## 底层细节

<a id="under_the_hood"></a>

现在，我们将介绍转换为 TensorFlow Lite 中的融合运算时整体设计的高层次细节。

### TensorFlow 中的复合运算

<a id="composing_ops"></a>

将 `tf.function` 与 [experimental_implements](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/python/eager/def_function.py#L470) 函数特性一起使用时，用户可以利用 TensorFlow 基元运算显式组成新运算，并指定由此产生的复合运算实现的接口。这非常有用，因为它可以提供：

1. 底层 TensorFlow 计算图中复合运算的明确定义边界。
2. 显式指定此运算实现的接口。`tf.function` 的参数对应于此接口的参数。

举例来说，我们考虑一个定义为实现嵌入向量查找的复合运算。它映射到 TensorFlow Lite 中的融合运算。

```python
  @tf.function(
        experimental_implements="embedding_lookup")
    def EmbFprop(embs, ids_vec):
      """Embedding forward prop.

      Effectively, it computes:
        num = size of ids_vec
        rets = zeros([num, embedding dim])
        for i in range(num):
          rets[i, :] = embs[ids_vec[i], :]
        return rets

      Args:
        embs: The embedding matrix.
        ids_vec: A vector of int32 embedding ids.

      Returns:
        The result of embedding lookups. A matrix of shape
        [num ids in ids_vec, embedding dims].
      """
      num = tf.shape(ids_vec)[0]
      rets = inplace_ops.empty([num] + emb_shape_suf, py_utils.FPropDtype(p))

      def EmbFpropLoop(i, embs, ids_vec, rets):
        # row_id = ids_vec[i]
        row_id = tf.gather(ids_vec, i)
        # row = embs[row_id]
        row = tf.reshape(tf.gather(embs, row_id), [1] + emb_shape_suf)
        # rets[i] = row
        rets = inplace_ops.alias_inplace_update(rets, [i], row)
        return embs, ids_vec, rets

      _, _, rets = functional_ops.For(
          start=0,
          limit=num,
          delta=1,
          inputs=[embs, ids_vec, rets],
          body=EmbFpropLoop,
          rewrite_with_while=compiled)
      if len(weight_shape) > 2:
        rets = tf.reshape(rets, [num, symbolic.ToStatic(p.embedding_dim)])
      return rets
```

如上文所述，我们让模型通过 `tf.function` 使用复合运算，这样便可构建一个通用基础架构来**识别此类运算并将其转换**为 TensorFlow Lite 融合运算。

### 扩展 TensorFlow Lite 转换器

今年早些时候发布的 TensorFlow Lite 转换器仅支持将 TensorFlow 模型作为计算图导入，其中的所有变量都会替换为其对应的常量值。这不适用于运算融合，因为此类计算图内嵌了所有函数，因此可以将变量转换为常量。

为了在转换过程中将 `tf.function` 与 `experimental_implements` 函数一起使用，需要保留这些函数，直到转换过程的后期。

因此，我们实现了一个在转换器中导入和转换 TensorFlow 模型的新工作流，以支持复合运算融合用例。具体地说，添加的新功能包括：

1. 将 TensorFlow [保存的模型导入 MLIR](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/tensorflow/translate/import_model.cc#L3748)
2. [fuse composite operations](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L103)
3. [variable mutability analysis](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/tensorflow/transforms/optimize_global_tensors.cc#L43)
4. [freeze all read-only variables](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/tensorflow/transforms/freeze_global_tensors.cc#L44)

这样，我们便能够在函数内嵌和变量冻结之前使用代表复合运算的函数执行运算融合。

### 实现运算融合

我们来更详细地了解运算融合传递。此传递执行以下操作：

1. 遍历 MLIR 模块中的所有函数。
2. 如果一个函数具有 tf._implements 特性，则基于该特性值，调用适当的运算融合效用函数。
3. 运算融合效用函数对函数的运算对象和特性（用作转换的接口）执行运算，并用包含融合运算的等效函数体替换函数的主体。
4. 在许多情况下，替换后的主体将包含融合运算以外的其他运算。这些运算对应于函数运算对象上的一些静态转换，这些转换的目的是获得融合运算的运算对象。由于这些计算均支持常量折叠，因此它们不会出现在仅存在融合运算的已导出 flatbuffer 中。

下面是传递中的代码段，展示了主工作流：

```
void PrepareCompositeFunctionsPass::ConvertTFImplements(FuncOp func,
                                                        StringAttr attr) {
  if (attr.getValue() == "embedding_lookup") {
    func.eraseBody();
    func.addEntryBlock();
    // Convert the composite embedding_lookup function body to a
    // TFLite fused embedding_lookup op.
    ConvertEmbeddedLookupFunc convert_embedded_lookup(func);
    if (failed(convert_embedded_lookup.VerifySignature())) {
      return signalPassFailure();
    }
    convert_embedded_lookup.RewriteFunc();
  } else if (attr.getValue() == mlir::TFL::kKerasLstm) {
     func.eraseBody();
     func.addEntryBlock();
     OpBuilder builder(func.getBody());
     if (failed(ConvertKerasLSTMLayer(func, &builder))) {
       return signalPassFailure();
     }
  } else if (.....) /* Other fusions can plug in here */
}
```

下面的代码段展示了利用函数作为转换接口将此复合运算映射到 TensorFlow Lite 中的融合运算。

<a id="fusion_code"></a>

```c++
void RewriteFunc() {
    Value lookup = func_.getArgument(1);
    Value value = func_.getArgument(0);
    auto output_type = func_.getType().getResult(0);

    OpBuilder builder(func_.getBody());
    auto op = builder.create<mlir::TFL::EmbeddingLookupOp>(
        func_.getLoc(), output_type, lookup, value);

    builder.create<mlir::ReturnOp>(func_.getLoc(), op.getResult());
  }
```
