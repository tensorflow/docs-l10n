# 实现自定义委托

[目录]

## 什么是 TensorFlow Lite 委托？

TensorFlow Lite [委托](https://www.tensorflow.org/lite/performance/delegates)允许您在另一个执行器上运行模型（部分或全部）。这种机制可以利用各种设备端加速器，例如 GPU 或 Edge TPU（张量处理单元）进行推断。这为开发者提供了一种与默认 TFLite 分离的灵活方法，以加快推断速度。

下面的图表对委托进行了汇总，更多详细信息请见下文。

![TFLite 委托](images/tflite_delegate.png "TFLite Delegates")

## 我应该在什么时候创建自定义委托？

TensorFlow Lite 有各种各样的目标加速器委托，如 GPU、DSP、Edge TPU 和 Android NNAPI 等框架。

对于以下场景，创建您自己的委托非常有用：

- 您希望集成任何现有委托都不支持的新机器学习推断引擎。
- 您有一个自定义的硬件加速器，可以改进已知场景的运行时间。
- 您正在开发可以加速某些模型的 CPU 优化（例如算子融合）。

## 委托的工作原理

考虑如下所示的简单模型计算图，以及对 Conv2D 和 Mean 运算具有更快实现的委托 “MyDelegate”。

![原始计算图](../images/performance/tflite_delegate_graph_1.png "Original Graph")

应用此 “MyDelegate” 后，原始的 TensorFlow Lite 计算图将进行如下更新：

![具有委托的计算图](../images/performance/tflite_delegate_graph_2.png "Graph with delegate")

上面的计算图是 TensorFlow Lite 按照以下两条规则拆分原始计算图后获得的：

- 可以由委托处理的特定运算被放入分区中，同时仍然满足运算之间的原始计算工作流依赖关系。
- 每个要委托的分区只有未由委托处理的输入和输出节点。

由委托处理的每个分区都被替换为原始计算图中的委托节点（也可以称为委托内核），这些节点会在调用时对分区求值。

根据模型的不同，最终的计算图可能以一个或多个节点结束，后者意味着委托不支持某些运算。通常，您不希望由委托处理多个分区，因为每次从委托切换到主计算图时，都会由于内存复制（例如，从 GPU 到 CPU）而将结果从委托的子计算图传递到主计算图，从而产生开销。此类开销可能会抵消性能收益，尤其是在有大量内存副本的情况下。

## 实现您自己的自定义委托

添加委托的首选方法是使用 [SimpleDelegate API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/simple_delegate.h)。

要创建新的委托，您需要实现 2 个接口，并为这些接口方法提供您自己的实现。

### 1 - `SimpleDelegateInterface`

此类表示委托的功能、支持哪些运算，以及用于创建封装委托计算图的内核的工厂类。有关更多详细信息，请参阅此 [C++ 头文件](https://github.com/tensorflow/tensorflow/blob/8a643858ce174b8bd1b4bb8fa4bfaa62f7e8c45f/tensorflow/lite/delegates/utils/simple_delegate.h#L71)中定义的接口。代码中的注释对每个 API 进行了详细说明。

### 2 - `SimpleDelegateKernelInterface`

此类封装了用于初始化/准备/和运行委托分区的逻辑。

它包含以下内容：（请参阅[定义](https://github.com/tensorflow/tensorflow/blob/8a643858ce174b8bd1b4bb8fa4bfaa62f7e8c45f/tensorflow/lite/delegates/utils/simple_delegate.h#L43)）

- Init(...)：它将被调用一次，以执行任何一次性初始化。
- Prepare(...)：为该节点的每个不同实例调用，适用于有多个委托分区的情况。通常，您希望在这里进行内存分配，因为每次调整张量的大小时都会调用它。
- Invoke(...)：将被调用以进行推断。

### 示例

在本例中，您将创建一个非常简单的委托，该委托仅支持两种类型的操作 (ADD) 和 (SUB)，且仅使用 float32 张量。

```
// MyDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class MyDelegate : public SimpleDelegateInterface {
 public:
  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    // Only supports Add and Sub ops.
    if (kTfLiteBuiltinAdd != registration->builtin_code &&
        kTfLiteBuiltinSub != registration->builtin_code)
      return false;
    // This delegate only supports float32 types.
    for (int i = 0; i < node->inputs->size; ++i) {
      auto& tensor = context->tensors[node->inputs->data[i]];
      if (tensor.type != kTfLiteFloat32) return false;
    }
    return true;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override {
    static constexpr char kName[] = "MyDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<MyDelegateKernel>();
  }
};
```

接下来，通过继承 `SimpleDelegateKernelInterface` 来创建您自己的委托内核

```
// My delegate kernel.
class MyDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    // Save index to all nodes which are part of this delegate.
    inputs_.resize(params->nodes_to_replace->size);
    outputs_.resize(params->nodes_to_replace->size);
    builtin_code_.resize(params->nodes_to_replace->size);
    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
      const int node_index = params->nodes_to_replace->data[i];
      // Get this node information.
      TfLiteNode* delegated_node = nullptr;
      TfLiteRegistration* delegated_node_registration = nullptr;
      TF_LITE_ENSURE_EQ(
          context,
          context->GetNodeAndRegistration(context, node_index, &delegated_node,
                                          &delegated_node_registration),
          kTfLiteOk);
      inputs_[i].push_back(delegated_node->inputs->data[0]);
      inputs_[i].push_back(delegated_node->inputs->data[1]);
      outputs_[i].push_back(delegated_node->outputs->data[0]);
      builtin_code_[i] = delegated_node_registration->builtin_code;
    }
    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    // Evaluate the delegated graph.
    // Here we loop over all the delegated nodes.
    // We know that all the nodes are either ADD or SUB operations and the
    // number of nodes equals ''inputs_.size()'' and inputs[i] is a list of
    // tensor indices for inputs to node ''i'', while outputs_[i] is the list of
    // outputs for node
    // ''i''. Note, that it is intentional we have simple implementation as this
    // is for demonstration.

    for (int i = 0; i < inputs_.size(); ++i) {
      // Get the node input tensors.
      // Add/Sub operation accepts 2 inputs.
      auto& input_tensor_1 = context->tensors[inputs_[i][0]];
      auto& input_tensor_2 = context->tensors[inputs_[i][1]];
      auto& output_tensor = context->tensors[outputs_[i][0]];
      TF_LITE_ENSURE_EQ(
          context,
          ComputeResult(context, builtin_code_[i], &input_tensor_1,
                        &input_tensor_2, &output_tensor),
          kTfLiteOk);
    }
    return kTfLiteOk;
  }

 private:
  // Computes the result of addition of 'input_tensor_1' and 'input_tensor_2'
  // and store the result in 'output_tensor'.
  TfLiteStatus ComputeResult(TfLiteContext* context, int builtin_code,
                             const TfLiteTensor* input_tensor_1,
                             const TfLiteTensor* input_tensor_2,
                             TfLiteTensor* output_tensor) {
    if (NumElements(input_tensor_1) != NumElements(input_tensor_2) ||
        NumElements(input_tensor_1) != NumElements(output_tensor)) {
      return kTfLiteDelegateError;
    }
    // This code assumes no activation, and no broadcasting needed (both inputs
    // have the same size).
    auto* input_1 = GetTensorData<float>(input_tensor_1);
    auto* input_2 = GetTensorData<float>(input_tensor_2);
    auto* output = GetTensorData<float>(output_tensor);
    for (int i = 0; i < NumElements(input_tensor_1); ++i) {
      if (builtin_code == kTfLiteBuiltinAdd)
        output[i] = input_1[i] + input_2[i];
      else
        output[i] = input_1[i] - input_2[i];
    }
    return kTfLiteOk;
  }

  // Holds the indices of the input/output tensors.
  // inputs_[i] is list of all input tensors to node at index 'i'.
  // outputs_[i] is list of all output tensors to node at index 'i'.
  std::vector<std::vector<int>> inputs_, outputs_;
  // Holds the builtin code of the ops.
  // builtin_code_[i] is the type of node at index 'i'
  std::vector<int> builtin_code_;
};


```

## 对新委托进行基准测试和评估

TFLite 有一组工具，可以用来快速针对 TFLite 模型进行测试。

- [模型基准测试工具](https://github.com/tensorflow/tensorflow/tree/f9ef3a8a0b64ad6393785f3259e9a24af09c84ad/tensorflow/lite/tools/benchmark)：该工具会接受 TFLite 模型，生成随机输入，然后以指定的运行次数重复运行该模型。它会在最后打印聚合的延迟统计数据。
- [推断比较工具](https://github.com/tensorflow/tensorflow/tree/f9ef3a8a0b64ad6393785f3259e9a24af09c84ad/tensorflow/lite/tools/evaluation/tasks/inference_diff)：对于给定的模型，该工具会生成随机的高斯数据，并通过两个不同的 TFLite 解释器传递该数据，一个运行单线程 CPU 内核，另一个使用用户定义的规范。它会在每个元素的基础上测量每个解释器的输出张量之间的绝对差异。此工具还可以帮助解决调试准确率问题。
- 还有特定于任务的评估工具，用于图像分类和目标检测。可在[此处](https://www.tensorflow.org/lite/performance/delegates#tools_for_evaluation)找到这些工具

此外，TFLite 包含一大组内核和运算单元测试，可以重用这些测试来测试覆盖范围更广的新委托，并确保常规的 TFLite 执行路径不会中断。

若要为新委托重用 TFLite 测试和工具，可以使用以下两个选项之一：

- 利用[委托注册器](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates)机制。
- 利用[外部委托](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/external)机制。

### 选择最佳方式

这两种方式都需要进行一些更改，如下所述。然而，第一种方式会静态地将委托联系在一起，并需要重新构建测试、基准和评估工具。而第二种方式会将委托作为共享库，并要求您公开共享库中的创建/删除方法。

因此，外部委托机制将与 TFLite 的[预构建的 TensorFlow Lite 工具二进制文件](#download-links-for-nightly-pre-built-tflite-tooling-binaries)一起工作。但它不那么明确，在自动化集成测试中设置可能会更复杂。为了清楚起见，请使用委托注册器方式。

### 选项 1：利用委托注册器

[委托注册器](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates)会保存委托提供程序的列表，每个委托提供程序都会提供一种基于命令行标志创建 TFLite 委托的简单方式，因此便于工具使用。要将新的委托插入到上面提到的所有 TensorFlow Lite 工具中，首先要创建一个新的委托提供程序，就像这[一个](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/hexagon_delegate_provider.cc)，然后只对 BUILD 规则进行一些更改。此集成过程的完整示例如下所示（代码可在[此处](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/utils/dummy_delegate)找到）。

假设您有一个实现 SimpleDelegate API 的委托，以及创建/删除这个“虚拟”委托的外部 “C” API，如下所示：

```
// Returns default options for DummyDelegate.
DummyDelegateOptions TfLiteDummyDelegateOptionsDefault();

// Creates a new delegate instance that need to be destroyed with
// `TfLiteDummyDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteDummyDelegateCreate(const DummyDelegateOptions* options);

// Destroys a delegate created with `TfLiteDummyDelegateCreate` call.
void TfLiteDummyDelegateDelete(TfLiteDelegate* delegate);
```

要将 “DummyDelegate” 与基准测试工具和推断工具集成，需要定义一个 DelegateProvider，如下所示：

```
class DummyDelegateProvider : public DelegateProvider {
 public:
  DummyDelegateProvider() {
    default_params_.AddParam("use_dummy_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;

  std::string GetName() const final { return "DummyDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(DummyDelegateProvider);

std::vector<Flag> DummyDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_dummy_delegate", params,
                                              "use the dummy delegate.")};
  return flags;
}

void DummyDelegateProvider::LogParams(const ToolParams& params) const {
  TFLITE_LOG(INFO) << "Use dummy test delegate : ["
                   << params.Get<bool>("use_dummy_delegate") << "]";
}

TfLiteDelegatePtr DummyDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_dummy_delegate")) {
    auto default_options = TfLiteDummyDelegateOptionsDefault();
    return TfLiteDummyDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

```

BUILD 规则定义很重要，因为您需要确保库始终处于链接状态，而不会被优化器删除。

```
#### The following are for using the dummy test delegate in TFLite tooling ####
cc_library(
    name = "dummy_delegate_provider",
    srcs = ["dummy_delegate_provider.cc"],
    copts = tflite_copts(),
    deps = [
        ":dummy_delegate",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
    ],
    alwayslink = 1, # This is required so the optimizer doesn't optimize the library away.
)
```

现在，在您的 BUILD 文件中添加以下两个封装器规则，以创建基准测试工具和推断工具以及其他评估工具的版本，这些工具可以与您自己的委托一起运行。

```
cc_binary(
    name = "benchmark_model_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tensorflow/lite/tools/benchmark:benchmark_model_main",
    ],
)

cc_binary(
    name = "inference_diff_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/inference_diff:run_eval_lib",
    ],
)

cc_binary(
    name = "imagenet_classification_eval_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification:run_eval_lib",
    ],
)

cc_binary(
    name = "coco_object_detection_eval_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval_lib",
    ],
)
```

您还可以将此委托提供程序插入到 TFLite 内核测试中，如[此处](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/dummy_delegate/README.md#kernel-tests)所述。

### 选项 2：利用外部委托

在此替代方案中，您首先创建外部委托适配器 [external_delegate_adaptor.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/dummy_delegate/external_delegate_adaptor.cc)，如下所示。请注意，与选项 1 相比，由于[前面提到的原因](#comparison-between-the-two-options)，这种方式的推荐指数略低。

```
TfLiteDelegate* CreateDummyDelegateFromOptions(char** options_keys,
                                               char** options_values,
                                               size_t num_options) {
  DummyDelegateOptions options = TfLiteDummyDelegateOptionsDefault();

  // Parse key-values options to DummyDelegateOptions.
  // You can achieve this by mimicking them as command-line flags.
  std::unique_ptr<const char*> argv =
      std::unique_ptr<const char*>(new const char*[num_options + 1]);
  constexpr char kDummyDelegateParsing[] = "dummy_delegate_parsing";
  argv.get()[0] = kDummyDelegateParsing;

  std::vector<std::string> option_args;
  option_args.reserve(num_options);
  for (int i = 0; i < num_options; ++i) {
    option_args.emplace_back("--");
    option_args.rbegin()->append(options_keys[i]);
    option_args.rbegin()->push_back('=');
    option_args.rbegin()->append(options_values[i]);
    argv.get()[i + 1] = option_args.rbegin()->c_str();
  }

  // Define command-line flags.
  // ...
  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(...),
      ...,
      tflite::Flag::CreateFlag(...),
  };

  int argc = num_options + 1;
  if (!tflite::Flags::Parse(&argc, argv.get(), flag_list)) {
    return nullptr;
  }

  return TfLiteDummyDelegateCreate(&options);
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Defines two symbols that need to be exported to use the TFLite external
// delegate. See tensorflow/lite/delegates/external for details.
TFL_CAPI_EXPORT TfLiteDelegate* tflite_plugin_create_delegate(
    char** options_keys, char** options_values, size_t num_options,
    void (*report_error)(const char*)) {
  return tflite::tools::CreateDummyDelegateFromOptions(
      options_keys, options_values, num_options);
}

TFL_CAPI_EXPORT void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate) {
  TfLiteDummyDelegateDelete(delegate);
}

#ifdef __cplusplus
}
#endif  // __cplusplus
```

接下来，创建相应的 BUILD 目标来构建动态库，如下所示：

```
cc_binary(
    name = "dummy_external_delegate.so",
    srcs = [
        "external_delegate_adaptor.cc",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":dummy_delegate",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
    ],
)
```

创建此外部委托 .so 文件后，只要二进制文件链接到支持命令行标志的 [external_delegate_provider](https://github.com/tensorflow/tensorflow/blob/8c6f2d55762f3fc94f98fdd8b3c5d59ee1276dba/tensorflow/lite/tools/delegates/BUILD#L145-L159) 库，您就可以构建二进制文件或使用预构建的文件与新委托一起运行，如[此处](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates#external-delegate-provider)所述。注：此外部委托提供程序已链接到现有的测试和工具二进制文件。

有关如何通过此外部委托方式对虚拟委托进行基准测试的说明，请参阅[此处](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/dummy_delegate/README.md#option-2-utilize-tensorflow-lite-external-delegate)的说明。您可以对前面提到的测试和评估工具使用类似的命令。

值得注意的是，*外部委托*是 TensorFlow Lite Python 绑定中的*委托*的对应 C++ 实现，如[处](https://github.com/tensorflow/tensorflow/blob/7145fc0e49be01ef6943f4df386ce38567e37797/tensorflow/lite/python/interpreter.py#L42)所示。因此，这里创建的动态外部委托适配器库可以直接与 TensorFlow Lite Python API 一起使用。

## 资源

### 下载 Nightly 预构建的 TFLite 工具二进制文件的链接

<table>
  <tr>
   <td>操作系统</td>
   <td>ARCH</td>
   <td>BINARY_NAME</td>
  </tr>
  <tr>
   <td rowspan="3">Linux</td>
   <td>x86_64</td>
   <td>
<ul>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model">benchmark_model</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_eval_inference_diff">inference_diff</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_eval_imagenet_image_classification">imagenet_image_classification_eval</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_eval_coco_object_detection">coco_object_detection_eval</a></li>
</ul>
   </td>
  </tr>
  <tr>
   <td>arm</td>
   <td>
<ul>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model">benchmark_model</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_eval_inference_diff">inference_diff</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_eval_imagenet_image_classification">imagenet_image_classification_eval</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_eval_coco_object_detection">coco_object_detection_eval</a></li>
</ul>
   </td>
  </tr>
  <tr>
   <td>aarch64</td>
   <td>
<ul>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model">benchmark_model</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_eval_inference_diff">inference_diff</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_eval_imagenet_image_classification">imagenet_image_classification_eval</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_eval_coco_object_detection">coco_object_detection_eval</a></li>
</ul>
   </td>
  </tr>
  <tr>
   <td rowspan="2">Android</td>
   <td>arm</td>
   <td>
<ul>
<li><a href="http://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model">benchmark_model</a></li>
<li>
<strong><p data-md-type="paragraph"><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model.apk">benchmark_model.apk</a></p></strong>
</li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_eval_inference_diff">inference_diff</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_eval_imagenet_image_classification">imagenet_image_classification_eval</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_eval_coco_object_detection">coco_object_detection_eval</a></li>
</ul>
   </td>
  </tr>
  <tr>
   <td>aarch64</td>
   <td>
<ul>
<li><a href="http://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model">benchmark_model</a></li>
<li>
<strong><p data-md-type="paragraph"><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model.apk">benchmark_model.apk</a></p></strong>
</li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_inference_diff">inference_diff</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_imagenet_image_classification">imagenet_image_classification_eval</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_coco_object_detection">coco_object_detection_eval</a></li>
</ul>
   </td>
  </tr>
</table>
