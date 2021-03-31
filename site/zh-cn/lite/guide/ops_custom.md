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

### Convert to a TensorFlow Lite Model

通过设置转换器属性 `allow_custom_ops`，创建一个具有自定义算子的 TensorFlow Lite 模型，如下所示：

<pre>converter = tf.lite.TFLiteConverter.from_concrete_functions([sin.get_concrete_function(x)])
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

All TensorFlow Lite operators (both custom and builtin) are defined using a simple pure-C interface that consists of four functions:

```c++
typedef struct {
  void* (*init)(TfLiteContext* context, const char* buffer, size_t length);
  void (*free)(TfLiteContext* context, void* buffer);
  TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);
} TfLiteRegistration;
```

有关 <code>TfLiteContext</code> 和 `TfLiteNode` 的详细信息，请参阅 [`common.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h)。前者提供错误报告功能和对全局对象（包括所有张量）的访问。后者允许实现访问其输入和输出。

When the interpreter loads a model, it calls `init()` once for each node in the graph. A given `init()` will be called more than once if the op is used multiple times in the graph. For custom ops a configuration buffer will be provided, containing a flexbuffer that maps parameter names to their values. The buffer is empty for builtin ops because the interpreter has already parsed the op parameters. Kernel implementations that require state should initialize it here and transfer ownership to the caller. For each `init()` call, there will be a corresponding call to `free()`, allowing implementations to dispose of the buffer they might have allocated in `init()`.

Whenever the input tensors are resized, the interpreter will go through the graph notifying implementations of the change. This gives them the chance to resize their internal buffer, check validity of input shapes and types, and recalculate output shapes. This is all done through `prepare()`, and implementations can access their state using `node->user_data`.

Finally, each time inference runs, the interpreter traverses the graph calling `invoke()`, and here too the state is available as `node->user_data`.

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

When initializing the `OpResolver`, add the custom op into the resolver (see below for an example). This will register the operator with Tensorflow Lite so that TensorFlow Lite can use the new implementation. Note that the last two arguments in `TfLiteRegistration` correspond to the `SinPrepare` and `SinEval` functions you defined for the custom op. If you used `SinInit` and `SinFree` functions to initialize variables used in the op and to free up space, respectively, then they would be added to the first two arguments of `TfLiteRegistration`; those arguments are set to `nullptr` in this example.

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

Regular usage requires that you use the `BuiltinOpResolver` and write:

```c++
tflite::ops::builtin::BuiltinOpResolver resolver;
```

要添加上面创建的自定义算子，您可以调用 `AddOp`（在将解析器传递给 `InterpreterBuilder` 之前）：

```c++
resolver.AddCustom("Sin", Register_SIN());
```

If the set of builtin ops is deemed to be too large, a new `OpResolver` could be code-generated based on a given subset of ops, possibly only the ones contained in a given model. This is the equivalent of TensorFlow's selective registration (and a simple version of it is available in the `tools` directory).

如果想用 Java 定义自定义算子，目前需要您自行构建自定义 JNI 层并[在此 JNI 代码中](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/native/builtin_ops_jni.cc)编译自己的 AAR。同样，如果想定义在 Python 中可用的上述算子，可以将注册放在 [Python 封装容器代码](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc)中。

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
