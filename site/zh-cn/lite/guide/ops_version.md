# TensorFlow Lite 操作(operator)的版本

本文档描述了TensorFlow Lite的操作(operator)版本架构。 操作(operator)的版本使开发人员能够将新功能和参数添加到现有操作中。 此外，它保证以下内容：

- 向后兼容性：新版本的 TensorFlow Lite 实现方式可以处理旧的模型文件。
- 向前兼容性：只要没有使用新功能，旧版本的 TensorFlow Lite 实现方式可以处理由新版 TOCO 生成的新版本的模型文件。
- 向前兼容性检测：如果旧 TensorFlow Lite 实现读取包含不受支持的新版运算的新模型，则应报告错误。

## 示例：向卷积添加膨胀

##示例：将膨胀(Dilation)添加到卷积操作中 本文档的其余部分通过展示如何在卷积操作中添加膨胀系数来解释 TFLite 中操作(operator)的版本。

了解本文档内容并不需要了解卷积核膨胀的知识。需要注意的是：

- 将添加2个新的整数参数：'dilation_width_factor' 和 'dilation_height_factor'。
- 不支持膨胀的旧卷积内核相当于将膨胀系数设置为 1。

### 更改 FlatBuffer 架构(Schema)

要将新参数添加到操作(operator)中，请更改`lite/schema/schema.fbs`中的选项表 。

例如，卷积的选项表如下所示：

```
table DepthwiseConv2DOptions {
  padding:Padding;
  stride_w:int;
  stride_h:int;
  depth_multiplier:int;
  fused_activation_function:ActivationFunctionType;
}
```

添加新参数时，请注意以下两点：

- 添加注释，指明哪个版本支持哪些参数。
- 当新的实现获取新添加的参数的默认值时，它应该与旧实现完全相同。

添加新参数后，参数表如下所示：

```
table Conv2DOptions {
  // 版本1支持的参数：
  padding:Padding;
  stride_w:int;
  stride_h:int;
  fused_activation_function:ActivationFunctionType;

  // 版本2支持的参数：
  dilation_width_factor:int = 1;
  dilation_height_factor:int = 1;
}
```

应为新架构重新生成 `lite/schema/schema_generated.h` 文件。

### 更改C中的结构体和内核实现

在TensorFlow Lite中，内核实现与FlatBuffer定义是分离发。 内核从`lite/builtin_op_data.h`中定义的C的结构体中读取参数。

原始深度卷积参数如下：

```
typedef struct {
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  int depth_multiplier;
  TfLiteFusedActivation activation;
} TfLiteDepthwiseConvParams;
```

与FlatBuffer架构(Schema)一样，通过添加注释，指明从哪个版本开始支持哪些参数。结果如下：

```
typedef struct {
  // Parameters for DepthwiseConv version 1 or above.
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  int depth_multiplier;
  TfLiteFusedActivation activation;
  // Parameters for DepthwiseConv version 2 or above.
  int dilation_width_factor;
  int dilation_height_factor;
} TfLiteDepthwiseConvParams;
```

另外，请更改内核实现从C结构体中读取新添加的参数。 细节在此不再赘述。

### 更改 FlatBuffer 代码以获取新参数

负责读取 FlatBuffer 并生成 C 结构体的逻辑是由 `lite/model.cc` 实现的。

更新该文件以处理新参数，如下所示：

```
TfLiteStatus ParseDepthwiseConv2D(const Operator* op,
                                  ErrorReporter* error_reporter,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteDepthwiseConvParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteDepthwiseConvParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const DepthwiseConv2DOptions* schema_params =
      op->builtin_options_as_DepthwiseConv2DOptions();

  if (schema_params != nullptr) {
    params->padding = ConvertPadding(schema_params->padding());
    params->stride_width = schema_params->stride_w();
    params->stride_height = schema_params->stride_h();
    params->depth_multiplier = schema_params->depth_multiplier();
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());

    params->dilation_width_factor = schema_params->dilation_w_factor();
    params->dilation_height_factor = schema_params->dilation_h_factor();
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}
```

这里不需要检查操作版本。 当新实现读取缺少扩张因子的旧模型文件时，它将使用1作为默认值，并且新内核将与旧内核一致地工作。

### 更改内核注册

MutableOpResolver（在`lite/op_resolver.h`中定义）提供了一些注册操作(operator)内核的函数。默认情况下，最小和最大版本都为1：

```
void AddBuiltin(tflite::BuiltinOperator op, TfLiteRegistration* registration,
                int min_version = 1, int max_version = 1);
void AddCustom(const char* name, TfLiteRegistration* registration,
               int min_version = 1, int max_version = 1);
```

内置的操作在 `lite/kernels/register.cc` 中注册。 在这个例子中，我们实现了一个新的操作内核，它可以处理 `Conv2D` 的版本1和版本2，所以我们需要将下面这行：

```
AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, Register_DEPTHWISE_CONV_2D());
```

修改为：

```
AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, Register_DEPTHWISE_CONV_2D(),
             /* min_version = */ 1,
             /* max_version = */ 2);
```

### 改变 TOCO TFLite 的导出

下一步是让 TFLite 填充执行运算所需的最低版本。在本例中，这意味着：

- 当膨胀系数均为1时，填充 版本=1。
- 除此之外，填充 版本=2。

通过将新版本添加到 `DepthwiseConv2D` 示例，为 `lite/tools/versioning/op_version.cc` 中的算子修改 `GetBuiltinOperatorVersion` 函数：

```
case BuiltinOperator_DEPTHWISE_CONV_2D:
  auto depthwise_conv_params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(op_sig.builtin_data);
  TFLITE_DCHECK(depthwise_conv_params != nullptr);
  if (depthwise_conv_params->dilation_width_factor != 1 ||
       depthwise_conv_params->dilation_height_factor != 1) {
    return 2;
  }
  return 1;
```

### 委托实现

TensorFlow Lite 提供了一个委托 API，可以将操作委派给硬件后端。在 Delegate 的 Prepare 函数中，检查该版本是否支持委派代码中的每个节点。

为此，您需要在 `lite/tools/versioning/runtime_version.cc` 中添加一个新的映射条目。

在本例中，您需要将以下条目添加到 `op_version_map` 中：

```
{{BuiltinOperator_DEPTHWISE_CONV_2D, 2}, %CURRENT_RUNTIME_VERSION%}
```

其中 `%CURRENT_RUNTIME_VERSION%` 对应 [tensorflow/core/public/version.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h) 中定义的当前运行时版本。

### 委托实现

TensorFlow Lite 提供了一个委托 API，可以将运算委托给硬件后端。在委托的 `Prepare` 函数中，检查委托代码中的每个节点是否支持该版本。

```
const int kMaxVersion = 1;
TfLiteNode* node;
TfLiteRegistration* registration = nullptr;
TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(context, node_index, &node, &registration));

if (registration->version > kMaxVersion) {
  // Reject the node if the version isn't supported.
}
```

即使委托仅支持版本 1 运算，这也是必需的，这使委托可以在获得更高版本运算时检测不兼容性。
