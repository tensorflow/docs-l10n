# 自定义算子

由于 TensorFlow Lite 内置算子库仅支持有限数量的 TensorFlow 算子，所以并非所有模型都可以转换。有关详细信息，请参阅[算子兼容性](ops_compatibility.md)。

为了进行转换，用户可以在 TensorFlow Lite 中提供自己不受支持的 TensorFlow 算子的自定义实现（即自定义算子）。*而如果要将一系列不受支持（或受支持）的 TensorFlow 算子组合到单个融合的优化后的自定义算子中，请参阅[算子融合](https://www.tensorflow.org/lite/convert/operation_fusion)*。

使用自定义算子包括四个步骤。

- [创建 TensorFlow 模型。](#create-a-tensorflow-model)确保 Saved Model（或 Graph Def）引用正确命名的 TensorFlow Lite 算子。

- [转换为 TensorFlow Lite 模型。](#convert-to-a-tensorflow-lite-model)确保设置正确的 TensorFlow Lite 转换器特性，以便成功转换模型。

- [创建并注册该算子。](#create-and-register-the-operator)这是为了使 TensorFlow Lite 运行时知道如何将计算图中的算子和参数映射到可执行的 C/C++ 代码。

- [对算子进行测试和性能分析。](#test-and-profile-your-operator)如果只想测试您的自定义算子，最好仅使用您的自定义算子来创建模型，并使用 [benchmark_model](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/benchmark_model.cc) 程序。

Let’s walk through an end-to-end example of running a model with a custom operator `tf.atan` (named as `Atan`, refer to #create-a-tensorflow-model) which is supported in TensorFlow, but unsupported in TensorFlow Lite.

Note: The `tf.atan` function is **not** a custom operator. It is a regular operator which is supported by both TensorFlow and TensorFlow Lite. But we **assume** that it is a custom operator in the following example in order to demonstrate a simple workflow.

The TensorFlow Text operator is an example of a custom operator. See the <a href="https://tensorflow.org/text/guide/text_tf_lite" class="external"> Convert TF Text to TF Lite</a> tutorial for a code example.

## Example: Custom `Atan` operator

Let’s walk through an example of supporting a TensorFlow operator that TensorFlow Lite does not have. Assume we are using the `Atan` operator and that we are building a very simple model for a function `y = atan(x + offset)`, where `offset` is trainable.

### 创建 TensorFlow 模型

The following code snippet trains a simple TensorFlow model. This model just contains a custom operator named `Atan`, which is a function `y = atan(x + offset)`, where `offset` is trainable.

```python
import tensorflow as tf

# Define training dataset and variables
x = [-8, 0.5, 2, 2.2, 201]
y = [-1.4288993, 0.98279375, 1.2490457, 1.2679114, 1.5658458]
offset = tf.Variable(0.0)

# Define a simple model which just contains a custom operator named `Atan`
@tf.function(input_signature=[tf.TensorSpec.from_tensor(tf.constant(x))])
def atan(x):
  return tf.atan(x + offset, name="Atan")

# Train model
optimizer = tf.optimizers.Adam(0.01)
def train(x, y):
    with tf.GradientTape() as t:
      predicted_y = atan(x)
      loss = tf.reduce_sum(tf.square(predicted_y - y))
    grads = t.gradient(loss, [offset])
    optimizer.apply_gradients(zip(grads, [offset]))

for i in range(1000):
    train(x, y)

print("The actual offset is: 1.0")
print("The predicted offset is:", offset.numpy())
```

```python
The actual offset is: 1.0
The predicted offset is: 0.99999905
```

此时，如果尝试使用默认转换器标志生成 TensorFlow Lite 模型，则会收到以下错误消息：

```none
Error:
error: 'tf.Atan' op is neither a custom op nor a flex op.
```

### 转换为 TensorFlow Lite 模型

通过设置转换器特性 `allow_custom_ops`，创建一个具有自定义算子的 TensorFlow Lite 模型，如下所示：

<pre>converter = tf.lite.TFLiteConverter.from_concrete_functions([atan.get_concrete_function()], atan)
&lt;b&gt;converter.allow_custom_ops = True&lt;/b&gt;
tflite_model = converter.convert()
</pre>

At this point, if you run it with the default interpreter using commands such as follows:

```python
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
```

You will still get the error:

```none
Encountered unresolved custom op: Atan.
```

### 创建并注册算子

所有 TensorFlow Lite 算子（自定义和内置算子）都使用由四个函数组成的简单纯 C 接口进行定义：

```c++
typedef struct {
  void* (*init)(TfLiteContext* context, const char* buffer, size_t length);
  void (*free)(TfLiteContext* context, void* buffer);
  TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);
} TfLiteRegistration;
```

请参阅 [`common.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h)，了解有关 <code>TfLiteContext</code> 和 `TfLiteNode` 的详细信息。前者提供错误报告功能和对全局对象（包括所有张量）的访问。后者允许实现访问其输入和输出。

当解释器加载模型时，它会为计算图中的每个节点调用一次 `init()`。如果算子在计算图中被多次使用，则会多次调用给定的 `init()`。对于自定义算子，将提供配置缓冲区，其中包含将参数名称映射到参数值的 FlexBuffer。内置算子的缓冲区为空，因为解释器已经解析了算子参数。需要状态的内核实现应在此处对其进行初始化，并将所有权转移给调用者。对于每个 `init()` 调用，都会有一个相应的 `free()` 调用，允许实现释放其可能在 `init()` 中分配的缓冲区。

每当调整输入张量的大小后，解释器都将遍历计算图，将更改通知给实现。这使它们有机会调整其内部缓冲区的大小，检查输入形状和类型的有效性，以及重新计算输出形状。这一切都通过 `prepare()` 完成，且实现可以使用 `node->user_data` 访问其状态。

最后，每次运行推断时，解释器都会通过调用 `invoke()` 来遍历计算图，同样，此处的状态也可作为 `node->user_data` 使用。

通过定义上述四个函数和全局注册函数，自定义算子可以使用与内置算子完全相同的方式实现，通常如下所示：

```c++
namespace tflite {
namespace ops {
namespace custom {
  TfLiteRegistration* Register_MY_CUSTOM_OP() {
    static TfLiteRegistration r = {my_custom_op::Init,
                                   my_custom_op::Free,
                                   my_custom_op::Prepare,
                                   my_custom_op::Eval};
    return &r;
  }
}  // namespace custom
}  // namespace ops
}  // namespace tflite
```

请注意，注册不会自动进行，而应显式调用 `Register_MY_CUSTOM_OP`。标准的 `BuiltinOpResolver`（可从 `:builtin_ops` 目标获得）负责内置算子的注册，而自定义算子必须收集到单独的自定义库中。

### 在 TensorFlow Lite 运行时中定义内核

要在 TensorFlow Lite 中使用算子，我们只需定义两个函数（`Prepare` 和 `Eval`），并构造 `TfLiteRegistration`：

```cpp
TfLiteStatus AtanPrepare(TfLiteContext* context, TfLiteNode* node) {
  using namespace tflite;
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  int num_dims = NumDimensions(input);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(num_dims);
  for (int i=0; i<num_dims; ++i) {
    output_size->data[i] = input->dims->data[i];
  }

  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus AtanEval(TfLiteContext* context, TfLiteNode* node) {
  using namespace tflite;
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  float* input_data = GetTensorData<float>(input);
  float* output_data = GetTensorData<float>(output);

  size_t count = 1;
  int num_dims = NumDimensions(input);
  for (int i = 0; i < num_dims; ++i) {
    count *= input->dims->data[i];
  }

  for (size_t i=0; i<count; ++i) {
    output_data[i] = atan(input_data[i]);
  }
  return kTfLiteOk;
}

TfLiteRegistration* Register_ATAN() {
  static TfLiteRegistration r = {nullptr, nullptr, AtanPrepare, AtanEval};
  return &r;
}
```

When initializing the `OpResolver`, add the custom op into the resolver (see below for an example). This will register the operator with Tensorflow Lite so that TensorFlow Lite can use the new implementation. Note that the last two arguments in `TfLiteRegistration` correspond to the `AtanPrepare` and `AtanEval` functions you defined for the custom op. If you used `AtanInit` and `AtanFree` functions to initialize variables used in the op and to free up space, respectively, then they would be added to the first two arguments of `TfLiteRegistration`; those arguments are set to `nullptr` in this example.

### 在内核库中注册算子

接下来我们需要在内核库中注册算子。此操作可通过 `OpResolver` 来完成。在后台，解释器将加载内核库，该库将被指定来执行模型中的每个算子。虽然默认库仅包含内置内核，但是可以使用自定义库来替换/增强默认库。

`OpResolver` 类会将算子代码和名称翻译成实际代码，其定义如下：

```c++
class OpResolver {
  virtual TfLiteRegistration* FindOp(tflite::BuiltinOperator op) const = 0;
  virtual TfLiteRegistration* FindOp(const char* op) const = 0;
  virtual void AddBuiltin(tflite::BuiltinOperator op, TfLiteRegistration* registration) = 0;
  virtual void AddCustom(const char* op, TfLiteRegistration* registration) = 0;
};
```

常规用法要求您使用 `BuiltinOpResolver` 并编写以下代码：

```c++
tflite::ops::builtin::BuiltinOpResolver resolver;
```

要添加上面创建的自定义算子，您可以调用 `AddOp`（在将解析器传递给 `InterpreterBuilder` 之前）：

```c++
resolver.AddCustom("Atan", Register_ATAN());
```

如果觉得内置算子集过大，可以基于给定的算子子集（可能仅包含给定模型中的算子）通过代码生成新的 `OpResolver`。这相当于 TensorFlow 的选择性注册（其简单版本可在 `tools` 目录中获得）。

If you want to define your custom operators in Java, you would currently need to build your own custom JNI layer and compile your own AAR [in this jni code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/native/nativeinterpreterwrapper_jni.cc). Similarly, if you wish to define these operators available in Python you can place your registrations in the [Python wrapper code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc).

请注意，可以按照与上文类似的过程支持一组运算（而不是单个算子），只需添加所需数量的 `AddCustom` 算子。另外，`BuiltinOpResolver` 还允许您使用 `AddBuiltin` 重写内置算子的实现。

### 对您的算子进行测试和性能分析

To profile your op with the TensorFlow Lite benchmark tool, you can use the [benchmark model tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#tflite-model-benchmark-tool) for TensorFlow Lite. For testing purposes, you can make your local build of TensorFlow Lite aware of your custom op by adding the appropriate `AddCustom` call (as show above) to [register.cc](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/core/kernels/register.cc)

## 最佳做法

1. 谨慎优化内存分配和取消分配。在 `Prepare` 中分配内存比在 `Invoke` 中分配更高效，并且最好在循环之前而非在每次迭代中分配内存。使用临时张量数据，而不要自己分配内存（请参阅第 2 项）。使用指针/引用而不是无节制地进行复制。

2. 如果某个数据结构在整个运算期间持续存在，建议使用临时张量预分配内存。您可能需要使用 OpData 结构来引用其他函数中的张量索引。请参阅[卷积内核](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/conv.cc)中的示例。示例代码段如下：

    ```
    auto* op_data = reinterpret_cast<OpData*>(node->user_data);
    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(1);
    node->temporaries->data[0] = op_data->temp_tensor_index;
    TfLiteTensor* temp_tensor = &context->tensors[op_data->temp_tensor_index];
    temp_tensor->type =  kTfLiteFloat32;
    temp_tensor->allocation_type = kTfLiteArenaRw;
    ```

3. 如果不想让它浪费太多内存，最好使用静态固定大小的数组（或在 `Resize` 中预分配的 `std::vector`），而不要使用在执行的每次迭代时动态分配的 `std::vector`。

4. 避免实例化尚不存在的标准库容器模板，因为它们会影响二进制文件的大小。例如，如果运算中需要在其他内核中不存在的 `std::map`，可以使用具有直接索引映射的 `std::vector`，同时保持较小的二进制文件大小。请查看其他内核使用的内容以获得深入见解（或询问）。

5. 检查指向由 `malloc` 返回的内存的指针。如果此指针是 `nullptr`，则不应使用该指针执行任何运算。如果在函数内 `malloc` 并出现退出错误，请在退出前释放内存。

6. 使用 `TF_LITE_ENSURE(context, condition)` 检查特定条件。使用 `TF_LITE_ENSURE` 时，您的代码不得将内存挂起（即，应该在分配任何可能泄漏的资源之前使用这些宏）。
