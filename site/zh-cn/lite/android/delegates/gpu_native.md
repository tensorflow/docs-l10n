# 使用 C/C++ API 进行 GPU 加速委托

使用图形处理单元 (GPU) 运行机器学习 (ML) 模型可以显著提高支持机器学习的应用的性能和用户体验。在 Android 设备上，您可以使用[*委托*](../../performance/delegates)和以下 API 之一启用模型的 GPU 加速执行：

- Interpreter API - [指南](./gpu)
- Task Library API - [指南](./gpu_task)
- 原生 (C/C++) API - 本指南

本指南涵盖 C API、C++ API 的 GPU 委托的高级使用以及量化模型的使用。有关将 GPU 委托用于 TensorFlow Lite 的更多信息，包括最佳做法和高级技术，请参阅 [GPU 委托](../../performance/gpu)页面。

## 启用 GPU 加速

通过使用 `TfLiteGpuDelegateV2Create()` 创建委托并使用 `TfLiteGpuDelegateV2Delete()` 销毁委托，在 C 或 C++ 中使用 Android 的 TensorFlow Lite GPU 委托，如以下示例代码所示：

```c++
// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr<Interpreter> interpreter;
InterpreterBuilder(*model, op_resolver)(&interpreter);

// NEW: Prepare GPU delegate.
auto* delegate = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
if (interpreter->Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));

// NEW: Clean up.
TfLiteGpuDelegateV2Delete(delegate);
```

查看 `TfLiteGpuDelegateOptionsV2` 对象代码以使用自定义选项构建一个委托实例。您可以使用 `TfLiteGpuDelegateOptionsV2Default()` 初始化默认选项，然后根据需要对其进行修改。

C 或 C++ 中 Android 的 TensorFlow Lite GPU 委托使用 [Bazel](https://bazel.io) 构建系统。可以使用以下命令构建委托：

```sh
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:delegate                           # for static library
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so  # for dynamic library
```

调用 `Interpreter::ModifyGraphWithDelegate()` 或 `Interpreter::Invoke()` 时，调用者在当前线程中必须具有 `EGLContext`，并且 `Interpreter::Invoke()` 必须从相同的 `EGLContext` 调用。如果 `EGLContext` 不存在，则委托将在内部创建一个，但您随后必须确保该 `Interpreter::Invoke()` 始终从调用 `Interpreter::ModifyGraphWithDelegate()` 的同一个线程调用。

## 量化模型 {:#quantized-models}

Android GPU 委托库默认支持量化模型。您无需更改任何代码即可将量化模型与 GPU 委托一起使用。以下部分说明了如何停用量化支持以用于测试或实验目的。

#### 停用量化模型支持

以下代码显示了如何***停用***对量化模型的支持。

<div>
  <devsite-selector>
    <section>
      <h3>C++</h3>
      <p></p>
<pre class="prettyprint lang-c++">TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;

auto* delegate = TfLiteGpuDelegateV2Create(options);
if (interpreter-&gt;ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;
      </pre>
    </section>
  </devsite-selector>
</div>

有关使用 GPU 加速运行量化模型的更多信息，请参阅 [GPU 委托](../../performance/gpu#quantized-models)概述。
