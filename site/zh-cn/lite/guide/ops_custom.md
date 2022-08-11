# 自定义算子

由于 TensorFlow Lite 内置算子库仅支持有限数量的 TensorFlow 算子，所以并非所有模型都可以转换。有关详细信息，请参阅[算子兼容性](ops_compatibility.md)。

为了进行转换，用户可以在 TensorFlow Lite 中提供自己不受支持的 TensorFlow 算子的自定义实现（称为自定义算子）。*而如果要将一系列不受支持（或受支持）的 TensorFlow 算子组合到一个融合的优化自定义算子中，请参阅[算子融合](https://www.tensorflow.org/lite/convert/operation_fusion)*。

使用自定义算子包括四个步骤。

- [创建 TensorFlow 模型。](#create-a-tensorflow-model)确保 Saved Model（或 Graph Def）引用正确命名的 TensorFlow Lite 算子。

- [转换为 TensorFlow Lite 模型。](#convert-to-a-tensorflow-lite-model)确保设置正确的 TensorFlow Lite 转换器属性，以便成功转换模型。

- [创建并注册该算子。](#create-and-register-the-operator)这是为了使 TensorFlow Lite 运行时知道如何将计算图中的算子和参数映射到可执行的 C/C++ 代码。

- [对您的算子进行测试和性能分析。](#test-and-profile-your-operator)如果您只想测试您的自定义算子，最好仅使用您的自定义算子来创建模型，并使用  [benchmark_model](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/benchmark_model.cc) 程序。

我们来通过一个端到端的示例演练一下，运行一个具有自定义算子的模型，该算子为 `tf.sin`（名为 `Sin`，请参阅 #create-a-tensorflow-model），在 TensorFlow 中受支持，但在 TensorFlow Lite 中不受支持。

注：实际上，`tf.sin` 并**非**自定义算子。它是一个 TensorFlow 和 TensorFlow Lite 都支持的常规算子。但在下面的示例中，我们**假设**它是一个自定义算子，以便演示一个简单的工作流。

## 示例：自定义 `Sin` 算子

我们来看一个支持 TensorFlow 算子的示例，该算子是 TensorFlow Lite 所没有的 。假设我们正在使用 `Sin` 算子，并且正在为函数 `y = sin(x + offset)` 构建一个非常简单的模型，其中 `offset` 可训练。

### 创建 TensorFlow 模型

下面的代码片段训练了一个简单的 TensorFlow 模型。这个模型只包含一个名为 `Sin` 的自定义算子，它是一个函数 `y = sin(x + offset)`，其中 `offset` 是可训练的。

```python
import tensorflow as tf

# Define training dataset and variables
x = [-8, 0.5, 2, 2.2, 201]
y = [-0.6569866 ,  0.99749499,  0.14112001, -0.05837414,  0.80641841]
offset = tf.Variable(0.0)

# Define a simple model which just contains a custom operator named `Sin`
@tf.function
def sin(x):
  return tf.sin(x + offset, name="Sin")

# Train model
optimizer = tf.optimizers.Adam(0.01)
def train(x, y):
    with tf.GradientTape() as t:
      predicted_y = sin(x)
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
The predicted offset is: 1.0000001
```

此时，如果尝试使用默认转换器标志生成 TensorFlow Lite 模型，则会收到以下错误消息：

```none
Error:
Some of the operators in the model are not supported by the standard TensorFlow
Lite runtime...... Here is
a list of operators for which you will need custom implementations: Sin.
```

### 转换为 TensorFlow Lite 模型

通过设置转换器属性 `allow_custom_ops`，创建一个具有自定义算子的 TensorFlow Lite 模型，如下所示：

<pre>converter = tf.lite.TFLiteConverter.from_concrete_functions([sin.get_concrete_function(x)], sin)
&lt;b&gt;converter.allow_custom_ops = True&lt;/b&gt;
tflite_model = converter.convert()
</pre>

此时，如果使用默认解释器运行它，则会收到以下错误消息：

```none
Error:
Didn't find custom operator for name 'Sin'
Registration failed.
```

### 创建并注册算子。

所有 TensorFlow Lite 运算符（自定义和内置）都使用由四个函数组成的简单纯 C 接口进行定义：

```c++
typedef struct {
  void* (*init)(TfLiteContext* context, const char* buffer, size_t length);
  void (*free)(TfLiteContext* context, void* buffer);
  TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);
} TfLiteRegistration;
```

有关 <code>TfLiteContext</code> 和 `TfLiteNode` 的详细信息，请参阅 [`common.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h)。前者提供错误报告功能和对全局对象（包括所有张量）的访问。后者允许实现访问其输入和输出。

当解释器加载模型时，它会为计算图中的每个节点调用一次 `init()`。如果运算在计算图中被多次使用，则会多次调用给定的 `init()`。对于自定义运算，将提供配置缓冲区，其中包含将参数名称映射到它们的值的 flexbuffer。内置运算的缓冲区为空，因为解释器已经解析了运算参数。需要状态的内核实现应在此处对其进行初始化，并将所有权转移给调用者。对于每个 `init()` 调用，都会有一个相应的 `free()` 调用，允许实现释放它们可能在 `init()` 中分配的缓冲区。

每当调整输入张量的大小时，解释器都将遍历计算图以通知更改的实现。这使它们有机会调整其内部缓冲区的大小、检查输入形状和类型的有效性，以及重新计算输出形状。这一切都通过 `prepare()` 完成，且实现可以使用 `node->user_data` 访问它们的状态。

最后，每次运行推断时，解释器都会遍历调用 `invoke()` 的计算图，同样，此处的状态也可作为 `node->user_data` 使用。

通过定义上述四个函数和通常如下所示的全局注册函数，自定义算子可以使用与内置算子完全相同的方式实现：

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

请注意，注册不是自动的，而应在某处显式调用 `Register_MY_CUSTOM_OP`。标准 `BuiltinOpResolver`（可从 `:builtin_ops` 目标获得）负责内置算子的注册，而自定义算子必须收集到单独的自定义库中。

### 在 TensorFlow Lite 运行时中定义内核

要在 TensorFlow Lite 中使用算子，我们只需定义两个函数（`Prepare` 和 `Eval`），并构造 `TfLiteRegistration`：

```cpp
TfLiteStatus SinPrepare(TfLiteContext* context, TfLiteNode* node) {
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

TfLiteStatus SinEval(TfLiteContext* context, TfLiteNode* node) {
  using namespace tflite;
  const TfLiteTensor* input = GetInput(context, node,0);
  TfLiteTensor* output = GetOutput(context, node,0);

  float* input_data = input->data.f;
  float* output_data = output->data.f;

  size_t count = 1;
  int num_dims = NumDimensions(input);
  for (int i = 0; i < num_dims; ++i) {
    count *= input->dims->data[i];
  }

  for (size_t i=0; i<count; ++i) {
    output_data[i] = sin(input_data[i]);
  }
  return kTfLiteOk;
}

TfLiteRegistration* Register_SIN() {
  static TfLiteRegistration r = {nullptr, nullptr, SinPrepare, SinEval};
  return &r;
}
```

初始化 `OpResolver` 时，将自定义算子添加到解析器中。这将向 Tensorflow Lite 注册算子，以便 TensorFlow Lite 可以使用新的实现。请注意，`TfLiteRegistration` 中的最后两个参数对应于您为自定义算子定义的 `SinPrepare` 和 `SinEval` 函数。如果使用 `SinInit` 和 `SinFree` 函数来分别初始化在算子中使用的变量并释放空间，则它们将被添加到 `TfLiteRegistration` 的前两个参数中；在此示例中，这些参数被设置为 `nullptr`。

### 在内核库中注册算子

现在我们需要在内核库中注册算子。此操作可通过 `OpResolver` 来完成。在后台，解释器将加载内核库，该库将被分配执行模型中的每个算子。虽然默认库仅包含内置内核，但是可以使用自定义库来替换/增强默认库。

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
resolver.AddCustom("Sin", Register_SIN());
```

如果觉得内置运算集过大，可以基于给定的运算子集（可能只是包含在给定模型中运算）通过代码生成新的 `OpResolver`。这相当于 TensorFlow 的选择性注册（其简单版本可在 `tools` 目录中获得）。

If you want to define your custom operators in Java, you would currently need to build your own custom JNI layer and compile your own AAR [in this jni code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/native/nativeinterpreterwrapper_jni.cc). Similarly, if you wish to define these operators available in Python you can place your registrations in the [Python wrapper code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc).

请注意，可以按照与上文类似的过程支持一组运算（而不是单个算子），只需添加所需数量的 `AddCustom` 算子。另外，`BuiltinOpResolver` 还允许您使用 `AddBuiltin` 重写内置算子的实现。

### 对您的算子进行测试和性能分析

要使用 TensorFlow Lite 基准测试工具来对您的算子进行性能分析，您可以使用 TensorFlow Lite 的[基准模型工具](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#tflite-model-benchmark-tool)。出于测试目的，您可以通过向 [register.cc](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/kernels/register.cc) 添加适当的 `AddCustom` 调用（如上所示），使您本地构建的 TensorFlow Lite 认识您的自定义算子。

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

4. Avoid instantiating standard library container templates that don't already exist, because they affect binary size. For example, if you need a `std::map` in your operation that doesn't exist in other kernels, using a `std::vector` with direct indexing mapping could work while keeping the binary size small. See what other kernels use to gain insight (or ask).

5. 检查指向由 `malloc` 返回的内存的指针。如果此指针是 `nullptr`，则不应使用该指针执行任何运算。如果在函数内 `malloc` 并出现退出错误，请在退出前释放内存。

6. 使用 `TF_LITE_ENSURE(context, condition)` 检查特定条件。使用 `TF_LITE_ENSURE` 时，您的代码不得将内存挂起（即，应该在分配任何可能泄漏的资源之前使用这些宏）。
