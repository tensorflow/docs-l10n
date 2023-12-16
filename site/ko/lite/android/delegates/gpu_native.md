# C/C++ API를 사용한 GPU 가속 대리자

그래픽 처리 장치(GPU)를 사용하여 머신러닝(ML) 모델을 실행하면 ML 지원 애플리케이션의 성능과 사용자 경험을 크게 향상할 수 있습니다. Android 기기에서는 [*대리자*](../../performance/delegates) 및 다음 API 중 하나를 사용하여 모델의 GPU 가속 실행 사용을 활성화할 수 있습니다.

- 인터프리터 API - [가이드](./gpu)
- 작업 라이브러리 API - [가이드](./gpu_task)
- 네이티브 (C/C++) API - 이 가이드

이 가이드는 C API, C++ API 및 양자화 모델 사용을 위한 GPU 대리자의 고급 사용법을 다룹니다. 모범 사례와 고급 기법을 포함한 TensorFlow Lite용 GPU 대리자 사용에 대한 자세한 내용은 [GPU 대리자](../../performance/gpu) 페이지를 참조하세요.

## GPU 가속 사용하기

다음 예시 코드에 표시된 대로 `TfLiteGpuDelegateV2Create()`로 대리자를 생성하고 `TfLiteGpuDelegateV2Delete()`로 제거하여 C or C++에서 Android용 TensorFlow Lite GPU 대리자를 사용합니다.

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

`TfLiteGpuDelegateOptionsV2` 객체 코드를 리뷰하여 사용자 정의 옵션으로 대리자 인스턴스를 빌드합니다. `TfLiteGpuDelegateOptionsV2Default()`로 기본 옵션을 초기화하고 필요한 대로 수정할 수 있습니다.

C 또는 C++의 Android용 TensorFlow Lite GPU 대리자는 [Bazel](https://bazel.io) 빌드 시스템을 사용합니다. 다음 명령을 사용하여 대리자를 빌드할 수 있습니다.

```sh
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:delegate                           # for static library
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so  # for dynamic library
```

`Interpreter::ModifyGraphWithDelegate()` 또는 `Interpreter::Invoke()`를 호출하면 호출 함수는 현재 스레드에 `EGLContext`가 있어야 하며 `Interpreter::Invoke()`는 동일한 `EGLContext`에서 호출되어야 합니다. `EGLContext`가 존재하지 않는다면 대리자는 내부에 하나를 생성하지만 `Interpreter::Invoke()`가 `Interpreter::ModifyGraphWithDelegate()`가 호출된 동일한 스레드에서 항상 호출되었는지 확인해야 합니다.

## 양자화 모델 {:#quantized-models}

Android GPU 대리자 라이브러리는 기본으로 양자화 모델을 지원합니다. GPU 대리자를 통해 양자화 모델을 사용해 모드를 변경할 필요가 없습니다. 다음 섹션은 테스트 또는 실험 목적으로 양자화 지원을 비활성화하는 방법을 설명합니다.

#### Disable quantized model support

다음 코드는 양자화 모델에 대한 지원을 ***비활성화***하는 방법을 보여줍니다.

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

GPU 가속화로 양자화 모델을 실행하는 데 대한 더 자세한 정보는 [GPU 대리자](../../performance/gpu#quantized-models) 개요를 참조하세요.
