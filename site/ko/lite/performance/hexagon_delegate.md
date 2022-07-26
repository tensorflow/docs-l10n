# TensorFlow Lite Hexagon 대리자

이 설명서에서는 Java 및/또는 C API로 애플리케이션에서 TensorFlow Lite Hexagon 대리자를 사용하는 방법을 설명합니다. 대리자는 Qualcomm Hexagon 라이브러리를 활용하여 DSP에서 양자화된 커널을 실행합니다. 이 대리자는 특히 NNAPI DSP 가속을 사용할 수 없는 기기(예: 구형 기기 또는 아직 DSP NNAPI 드라이버가 없는 기기)의 경우 NNAPI 기능을 *보완*하기 위한 것입니다.

참고: 이 대리자는 실험(베타) 단계입니다.

**지원되는 기기:**

현재, 다음과 같은 Hexagon 아키텍처가 지원됩니다(여기에 국한되지는 않음).

- Hexagon 680
    - SoC 예: Snapdragon 821, 820, 660
- Hexagon 682
    - SoC 예: Snapdragon 835
- Hexagon 685
    - SoC 예: Snapdragon 845, Snapdragon 710, QCS605, QCS603
- Hexagon 690
    - SoC 예: Snapdragon 855, QCS610, QCS410, RB5

**지원되는 모델**

Hexagon 대리자는 [사후 훈련 정수 양자화](https://www.tensorflow.org/lite/performance/quantization_spec)를 사용하여 생성된 모델을 포함하여 [8bit 대칭 양자화 사양](https://www.tensorflow.org/lite/performance/post_training_integer_quant)을 준수하는 모든 모델을 지원합니다. 기존 [양자화 인식 훈련](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize) 경로로 훈련된 UInt8 모델도 지원됩니다(예: 호스팅 모델 페이지의 [양자화 버전](https://www.tensorflow.org/lite/guide/hosted_models#quantized_models)).

## Hexagon 대리자 Java API

```java
public class HexagonDelegate implements Delegate, Closeable {

  /*
   * Creates a new HexagonDelegate object given the current 'context'.
   * Throws UnsupportedOperationException if Hexagon DSP delegation is not
   * available on this device.
   */
  public HexagonDelegate(Context context) throws UnsupportedOperationException


  /**
   * Frees TFLite resources in C runtime.
   *
   * User is expected to call this method explicitly.
   */
  @Override
  public void close();
}
```

### 사용 예제

#### 1단계: 야간 Hexagon 대리자 AAR을 사용하도록 app/build.gradle를 편집합니다.

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
  implementation 'org.tensorflow:tensorflow-lite-hexagon:0.0.0-nightly-SNAPSHOT'
}
```

#### 2단계: Android 앱에 Hexagon 라이브러리를 추가합니다.

- hexagon_nn_skel.run을 다운로드하고 실행합니다. 3개의 서로 다른 공유 라이브러리 “libhexagon_nn_skel.so”, “libhexagon_nn_skel_v65.so”, “libhexagon_nn_skel_v66.so”를 제공해야 합니다.
    - [v1.10.3](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_1_10_3_1.run)
    - [v1.14](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.14.run)
    - [v1.17](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.17.0.0.run)
    - [v1.20](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.0.run)
    - [v1.20.0.1](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.1.run)

참고: 라이선스 계약에 동의해야 합니다.

참고: 2021년 2월 23일부터 v1.20.0.1을 사용해야 합니다.

참고: 호환되는 버전의 인터페이스 라이브러리와 함께 hexagon_nn 라이브러리를 사용해야 합니다. 인터페이스 라이브러리는 AAR의 일부이며 bazel로 [config](https://github.com/tensorflow/tensorflow/blob/master/third_party/hexagon/workspace.bzl)를 통해 가져옵니다. 여기서 bazel 구성의 버전을 사용해야 합니다.

- 다른 공유 라이브러리와 함께 앱에 3개를 모두 포함합니다. [앱에 공유 라이브러리를 추가하는 방법](#how-to-add-shared-library-to-your-app)을 참조하세요. 대리자는 기기에 따라 성능이 가장 좋은 라이브러리를 자동으로 선택합니다.

참고: 앱이 32bit 및 64bit ARM 기기용으로 빌드되는 경우 Hexagon 공유 라이브러리를 32bit 및 64bit lib 폴더 모두에 추가해야 합니다.

#### 3단계: 대리자를 생성하고 TensorFlow Lite 인터프리터를 초기화합니다.

```java
import org.tensorflow.lite.HexagonDelegate;

// Create the Delegate instance.
try {
  hexagonDelegate = new HexagonDelegate(activity);
  tfliteOptions.addDelegate(hexagonDelegate);
} catch (UnsupportedOperationException e) {
  // Hexagon delegate is not supported on this device.
}

tfliteInterpreter = new Interpreter(tfliteModel, tfliteOptions);

// Dispose after finished with inference.
tfliteInterpreter.close();
if (hexagonDelegate != null) {
  hexagonDelegate.close();
}
```

## Hexagon 대리자 C API

```c
struct TfLiteHexagonDelegateOptions {
  // This corresponds to the debug level in the Hexagon SDK. 0 (default)
  // means no debug.
  int debug_level;
  // This corresponds to powersave_level in the Hexagon SDK.
  // where 0 (default) means high performance which means more power
  // consumption.
  int powersave_level;
  // If set to true, performance information about the graph will be dumped
  // to Standard output, this includes cpu cycles.
  // WARNING: Experimental and subject to change anytime.
  bool print_graph_profile;
  // If set to true, graph structure will be dumped to Standard output.
  // This is usually beneficial to see what actual nodes executed on
  // the DSP. Combining with 'debug_level' more information will be printed.
  // WARNING: Experimental and subject to change anytime.
  bool print_graph_debug;
};

// Return a delegate that uses Hexagon SDK for ops execution.
// Must outlive the interpreter.
TfLiteDelegate*
TfLiteHexagonDelegateCreate(const TfLiteHexagonDelegateOptions* options);

// Do any needed cleanup and delete 'delegate'.
void TfLiteHexagonDelegateDelete(TfLiteDelegate* delegate);

// Initializes the DSP connection.
// This should be called before doing any usage of the delegate.
// "lib_directory_path": Path to the directory which holds the
// shared libraries for the Hexagon NN libraries on the device.
void TfLiteHexagonInitWithPath(const char* lib_directory_path);

// Same as above method but doesn't accept the path params.
// Assumes the environment setup is already done. Only initialize Hexagon.
Void TfLiteHexagonInit();

// Clean up and switch off the DSP connection.
// This should be called after all processing is done and delegate is deleted.
Void TfLiteHexagonTearDown();
```

### 사용 예제

#### 1단계: 야간 Hexagon 대리자 AAR을 사용하도록 app/build.gradle를 편집합니다.

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
  implementation 'org.tensorflow:tensorflow-lite-hexagon:0.0.0-nightly-SNAPSHOT'
}
```

#### 2단계: Android 앱에 Hexagon 라이브러리를 추가합니다.

- hexagon_nn_skel.run을 다운로드하고 실행합니다. 3개의 서로 다른 공유 라이브러리 “libhexagon_nn_skel.so”, “libhexagon_nn_skel_v65.so”, “libhexagon_nn_skel_v66.so”를 제공해야 합니다.
    - [v1.10.3](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_1_10_3_1.run)
    - [v1.14](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.14.run)
    - [v1.17](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.17.0.0.run)
    - [v1.20](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.0.run)
    - [v1.20.0.1](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.1.run)

참고: 라이선스 계약에 동의해야 합니다.

참고: 2021년 2월 23일부터 v1.20.0.1을 사용해야 합니다.

참고: 호환되는 버전의 인터페이스 라이브러리와 함께 hexagon_nn 라이브러리를 사용해야 합니다. 인터페이스 라이브러리는 AAR의 일부이며 bazel로 [config](https://github.com/tensorflow/tensorflow/blob/master/third_party/hexagon/workspace.bzl)를 통해 가져옵니다. 여기서 bazel 구성의 버전을 사용해야 합니다.

- 다른 공유 라이브러리와 함께 앱에 3개를 모두 포함합니다. [앱에 공유 라이브러리를 추가하는 방법](#how-to-add-shared-library-to-your-app)을 참조하세요. 대리자는 기기에 따라 성능이 가장 좋은 라이브러리를 자동으로 선택합니다.

참고: 앱이 32bit 및 64bit ARM 기기용으로 빌드되는 경우 Hexagon 공유 라이브러리를 32bit 및 64bit lib 폴더 모두에 추가해야 합니다.

#### 3단계: C 헤더를 포함합니다.

- 헤더 파일 "hexagon_delegate.h"는 [GitHub](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/hexagon/hexagon_delegate.h)에서 다운로드하거나 Hexagon delegate AAR에서 추출할 수 있습니다.

#### 4단계: 대리자를 생성하고 TensorFlow Lite 인터프리터를 초기화합니다.

- 코드에서 네이티브 Hexagon 라이브러리가 로드되었는지 확인합니다. 활동 또는 Java 진입점에서 `System.loadLibrary("tensorflowlite_hexagon_jni");`<br>을 호출하여 수행할 수 있습니다.

- 대리자를 만듭니다. 예는 다음과 같습니다.

```c
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"

// Assuming shared libraries are under "/data/local/tmp/"
// If files are packaged with native lib in android App then it
// will typically be equivalent to the path provided by
// "getContext().getApplicationInfo().nativeLibraryDir"
const char[] library_directory_path = "/data/local/tmp/";
TfLiteHexagonInitWithPath(library_directory_path);  // Needed once at startup.
::tflite::TfLiteHexagonDelegateOptions params = {0};
// 'delegate_ptr' Need to outlive the interpreter. For example,
// If use case will need to resize input or anything that can trigger
// re-applying delegates then 'delegate_ptr' need to outlive the interpreter.
auto* delegate_ptr = ::tflite::TfLiteHexagonDelegateCreate(&params);
Interpreter::TfLiteDelegatePtr delegate(delegate_ptr,
  [](TfLiteDelegate* delegate) {
    ::tflite::TfLiteHexagonDelegateDelete(delegate);
  });
interpreter->ModifyGraphWithDelegate(delegate.get());
// After usage of delegate.
TfLiteHexagonTearDown();  // Needed once at end of app/DSP usage.
```

## 앱에 공유 라이브러리 추가하기

- 'app/src/main/jniLibs' 폴더를 만들고 각 대상 아키텍처에 대한 디렉터리를 만듭니다. 예를 들면 다음과 같습니다.
    - ARM 64bit: `app/src/main/jniLibs/arm64-v8a`
    - ARM 32bit: `app/src/main/jniLibs/armeabi-v7a`
- 아키텍처와 일치하는 디렉토리에 .so를 넣습니다.

참고: 애플리케이션 게시에 App Bundle을 사용하는 경우 gradle.properties 파일에서 android.bundle.enableUncompressedNativeLibs=false를 설정할 수 있습니다.

## 피드백

문제가 있는 경우, 사용된 전화 모델 및 보드(<code>adb shell getprop ro.product.device</code> 및 `adb shell getprop ro.board.platform`)를 포함하여 필요한 모든 재현 세부 정보와 함께 <a>GitHub</a> 문제를 만드세요.

## 자주하는 질문

- 대리자는 어떤 연산을 지원하나요?
    - 현재 [지원되는 연산 및 제약 조건](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/hexagon/README.md) 목록을 참조하세요.
- 대리자를 활성화할 때 모델이 DSP를 사용하고 있음을 어떻게 알 수 있습니까?
    - 대리자를 활성화하면 두 개의 로그 메시지가 출력됩니다. 하나는 대리자가 생성되었는지 여부를 나타내고 다른 하나는 대리자를 사용하여 실행 중인 노드 수를 나타냅니다. <br> `Created TensorFlow Lite delegate for Hexagon.` <br> `Hexagon delegate: X nodes delegated out of Y nodes.`
- 대리자를 실행하려면 모델의 모든 연산이 지원되어야 합니까?
    - 아니요, 모델은 지원되는 연산에 따라 하위 그래프로 분할됩니다. 지원되지 않는 모든 연산은 CPU에서 실행됩니다.
- 소스에서 Hexagon 대리자 AAR을 어떻게 만들 수 있습니까?
    - `bazel build -c opt --config=android_arm64 tensorflow/lite/delegates/hexagon/java:tensorflow-lite-hexagon`을 사용합니다.
- Android 기기에 지원되는 SoC가 있는데 Hexagon 대리자가 초기화되지 않는 이유는 무엇입니까?
    - 기기에 실제로 지원되는 SoC가 있는지 확인하세요. `adb shell cat /proc/cpuinfo | grep Hardware`를 실행하여 'Hardware : Qualcomm Technologies, Inc MSMXXXX'와 같은 내용을 반환하는지 확인합니다.
    - 일부 휴대폰 제조업체는 같은 휴대폰 모델에 대해 서로 다른 SoC를 사용합니다. 따라서 Hexagon 대리자는 같은 전화 모델이라도 모든 기기가 아닌 일부에서만 동작할 수 있습니다.
    - 일부 휴대폰 제조업체는 시스템이 아닌 Android 앱에서 Hexagon DSP 사용을 의도적으로 제한하여 Hexagon 대리자가 동작하지 못하게 합니다.
- 내 전화기에 DSP 액세스가 잠겼습니다. 전화를 루팅했지만 여전히 대리자를 실행할 수 없습니다. 어떻게 해야 합니까?
    - `adb shell setenforce 0`을 실행하여 SELinux 적용을 비활성화해야 합니다.
