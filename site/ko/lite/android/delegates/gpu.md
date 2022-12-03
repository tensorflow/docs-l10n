# Android용 GPU 가속화 대리자

그래픽 처리 장치(GPU)를 사용하여 머신러닝(ML) 모델을 실행하면 모델의 성능과 ML 지원 애플리케이션의 사용자 경험을 극적으로 향상할 수 있습니다. Android 기기에서, [*대리자*](../../performance/delegates)를 사용해 모델에 대한 GPU 가속화 실행 사용을 활성화할 수 있습니다. 대리자는 TensorFlow Lite의 하드웨어 드라이버 역할을 하여 GPU 프로세서에서 모델의 코드를 실행할 수 있습니다.

이 페이지는 Android 앱에서 TensorFlow Lite 모델용 GPU 가속화를 활성화하는 방법을 설명합니다. 모범 사례와 고급 기술을 포함한 TensorFlow Lite용 GPU 대리자를 사용하는 데 대한 더 자세한 정보는 [GPU 대리자](../../performance/gpu) 페이지를 참조하세요.

## Task Library API가 있는 GPU 사용

TensorFlow Lite [Task Libraries](../../inference_with_metadata/task_library/overview)는 머신러닝 애플리케이션을 빌드하기 위한 작업별 API 세트를 제공합니다. 이 섹션은 이러한 API를 통해 GPU 가속 장치 대리자 사용 방법을 설명합니다.

### 프로젝트 종속성 추가

다음 코드 예시에 표시된 대로 `tensorflow-lite-gpu-delegate-plugin` 패키지를 포함하도록 개발 프로젝트`build.gradle` 파일을 업데이트하는 다음 종속성을 추가하여 TensorFlow Lite Task Libraries를 통해 GPU 대리자 API에 대한 접속을 활성화하세요.

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

### GPU 가속화 사용

[`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder) 클래스가 있는 Task API 모델 클래스용 GPU 대리자 옵션을 활성화하세요. 예를 들어, 다음 코드 예시에 표시된 대로 `ObjectDetector`에 GPU를 설정할 수 있습니다.

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">    import org.tensorflow.lite.task.core.BaseOptions
    import org.tensorflow.lite.task.gms.vision.detector.ObjectDetector

    val baseOptions = BaseOptions.builder().useGpu().build()

    val options =
        ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setMaxResults(1)
            .build()

    val objectDetector = ObjectDetector.createFromFileAndOptions(
      context, model, options)

      </pre>
    </section>
    <section>
      <h3> Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    import org.tensorflow.lite.task.core.BaseOptions
    import org.tensorflow.lite.task.gms.vision.detector.ObjectDetector

    BaseOptions baseOptions = BaseOptions.builder().useGpu().build();

    ObjectDetectorOptions options =
        ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setMaxResults(1)
            .build();

    val objectDetector = ObjectDetector.createFromFileAndOptions(
      context, model, options);
      </pre>
    </section>
  </devsite-selector>
</div>

## Interpreter API가 있는 GPU 사용

TensorFlow Lite [Interpreter API](../../api_docs/java/org/tensorflow/lite/InterpreterApi)는 머신러닝 애플리케이션을 빌드하는 목적을 가진 일반 API 세트를 제공합니다. 이 섹션은 이러한 API를 통해 GPU 가속 장치 대리자 사용 방법을 설명합니다.

### 프로젝트 종속성 추가

다음 코드 예시에 표시된 대로 `org.tensorflow:tensorflow-lite-gpu` 패키지를 포함하도록 개발 프로젝트`build.gradle` 파일을 업데이트하는 다음 종속성을 추가하여 GPU 대리자 API에 대한 접속을 활성화하세요.

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite'
    implementation 'org.tensorflow:tensorflow-lite-gpu'
}
```

### GPU 가속화 사용

`TfLiteDelegate`로 GPU에서 TensorFlow Lite를 실행합니다. Java에서는 `Interpreter.Options`를 통해 `GpuDelegate`를 지정할 수 있습니다.

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">    import org.tensorflow.lite.Interpreter
    import org.tensorflow.lite.gpu.CompatibilityList
    import org.tensorflow.lite.gpu.GpuDelegate

    val compatList = CompatibilityList()

    val options = Interpreter.Options().apply{
        if(compatList.isDelegateSupportedOnThisDevice){
            // if the device has a supported GPU, add the GPU delegate
            val delegateOptions = compatList.bestOptionsForThisDevice
            this.addDelegate(GpuDelegate(delegateOptions))
        } else {
            // if the GPU is not supported, run on 4 threads
            this.setNumThreads(4)
        }
    }

    val interpreter = Interpreter(model, options)

    // Run inference
    writeToInput(input)
    interpreter.run(input, output)
    readFromOutput(output)
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    import org.tensorflow.lite.Interpreter;
    import org.tensorflow.lite.gpu.CompatibilityList;
    import org.tensorflow.lite.gpu.GpuDelegate;

    // Initialize interpreter with GPU delegate
    Interpreter.Options options = new Interpreter.Options();
    CompatibilityList compatList = CompatibilityList();

    if(compatList.isDelegateSupportedOnThisDevice()){
        // if the device has a supported GPU, add the GPU delegate
        GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
        GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
        options.addDelegate(gpuDelegate);
    } else {
        // if the GPU is not supported, run on 4 threads
        options.setNumThreads(4);
    }

    Interpreter interpreter = new Interpreter(model, options);

    // Run inference
    writeToInput(input);
    interpreter.run(input, output);
    readFromOutput(output);
      </pre>
    </section>
  </devsite-selector>
</div>

참고: GPU 대리자는 실행되는 것과 같은 스레드에서 생성되어야 합니다. 그렇지 않으면 오류(`TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized`)가 발생할 수 있습니다.

GPU 대리자는 또한 Android Studio에서 연결된 ML 모델과 사용할 수 있습니다. 더 자세한 정보는 [메타데이터를 사용하는 일반 모델 인터페이스](../../inference_with_metadata/codegen#acceleration)를 참조하세요.

## 고급 GPU 지원

이 섹션은 C API, C++ API 및 양자화된 모델의 사용을 포함한 Android용 GPU 대리자 고급 사용에 대해 다룹니다.

### Android용 C/C++ API

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

C 또는 C++의 Android용 TensorFlow Lite GPU 대리자는 [Bazel](https://bazel.io) 빌드 시스템을 사용합니다. 다음 명령을 사용하여 대리자를 빌드할 수 있씁니다.

```sh
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:delegate                           # for static library
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so  # for dynamic library
```

`Interpreter::ModifyGraphWithDelegate()` 또는 `Interpreter::Invoke()`를 호출하면 호출자는 현재 스레드에 `EGLContext`가 있어야 하며 `Interpreter::Invoke()`는 동일한 `EGLContext`에서 호출되어야 합니다. `EGLContext`가 존재하지 않는다면 대리자는 내부에 하나를 생성하지만 `Interpreter::Invoke()`가 `Interpreter::ModifyGraphWithDelegate()`가 호출된 동일한 스레드에서 항상 호출되었는지 확인해야 합니다.

### 양자화 모델 {:#quantized-models}

Android GPU 대리자 라이브러리는 기본으로 양자화 모델을 지원합니다. GPU 대리자를 통해 양자화 모델을 사용해 모드를 변경할 필요가 없습니다. 다음 섹션은 테스트 또는 실험 목적으로 양자화 지원을 비활성화하는 방법을 설명합니다.

#### 양자화 모델 지원 비활성화

다음 코드는 양자화 모델에 대한 지원을 ***비활성화***하는 방법을 보여줍니다.

<div>
  <devsite-selector>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">GpuDelegate delegate = new GpuDelegate(new GpuDelegate.Options().setQuantizedModelsAllowed(false));

Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
      </pre>
    </section>
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
