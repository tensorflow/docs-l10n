# TensorFlow 연산자 선택

Since the TensorFlow Lite builtin operator library only supports a limited number of TensorFlow operators, not every model is convertible. For details, refer to [operator compatibility](ops_compatibility.md).

변환을 위해 사용자는 TensorFlow Lite 모델에서 [특정 TensorFlow ops](op_select_allowlist.md)를 사용하도록 설정할 수 있습니다. 그러나 TensorFlow ops로 TensorFlow Lite 모델을 실행하려면 핵심 TensorFlow 런타임을 가져와야 하므로 TensorFlow Lite 인터프리터 바이너리 크기가 늘어납니다. Android의 경우 필요한 Tensorflow ops만 선택적으로 빌드하여 이를 방지할 수 있습니다. 자세한 내용은 [바이너리 크기 줄이기](../guide/reduce_binary_size.md)를 참조하세요.

이 문서는 선택한 플랫폼에서 TensorFlow ops이 포함된 TensorFlow Lite 모델을 [변환](#convert_a_model)하고 [실행](#run_inference)하는 방법을 설명합니다. 또한 [성능 및 크기 메트릭](#metrics) 및 [알려진 제한 사항](#known_limitations)에 대해서도 설명합니다.

## 모델 변환하기

다음 예제에서는 선택한 TensorFlow ops를 사용하여 TensorFlow Lite 모델을 생성하는 방법을 보여줍니다.

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

## 추론 실행하기

특정 TensorFlow ops를 지원하여 변환된 TensorFlow Lite 모델을 사용하는 경우, 클라이언트는 TensorFlow ops의 필수 라이브러리가 포함된 TensorFlow Lite 런타임도 사용해야 합니다.

### Android AAR

To reduce the binary size, please build your own custom AAR files as guided in the [next section](#building-the-android-aar). If the binary size is not a considerable concern, we recommend using the prebuilt [AAR with TensorFlow ops hosted at MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-select-tf-ops).

다음과 같이 표준 TensorFlow Lite AAR과 함께 이 AAR을 추가하여 `build.gradle` 종속성에서 이 내용을 지정할 수 있습니다.

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
    // This dependency adds the necessary TF op support.
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:0.0.0-nightly-SNAPSHOT'
}
```

야간 스냅샷을 사용하려면 [Sonatype 스냅샷 저장소](./build_android#use_nightly_snapshots)를 추가했는지 확인하세요.

종속성을 추가한 경우, 그래프의 TensorFlow ops를 처리하는 데 필요한 대리자가 이러한 연산자가 필요한 그래프에 대해 자동으로 설치됩니다.

*참고*: TensorFlow ops 종속성은 상대적으로 크기 때문에 `abiFilters`를 설정하여 `.gradle` 파일에서 불필요한 x86 ABI를 제거하는 것이 좋습니다.

```build
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

#### Android AAR 빌드하기

바이너리 크기를 줄이거나 다른 높은 수준의 경우, 라이브러리를 수동으로 빌드할 수도 있습니다. <a href="android.md">제대로 작동하는 TensorFlow Lite 빌드 환경</a>을 가정하고 다음과 같이 특정 TensorFlow ops로 Android AAR을 빌드합니다.

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

그러면 TensorFlow Lite 내장 및 사용자 정의 ops에 대한 AAR 파일 `bazel-bin/tmp/tensorflow-lite.aar`이 생성됩니다. 그리고 TensorFlow ops에 대한 AAR 파일 `bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar`을 생성합니다. 제대로 작동하는 빌드 환경이 없는 경우 [docker를 사용하여 위의 파일을 빌드](../guide/reduce_binary_size.md#selectively_build_tensorflow_lite_with_docker)할 수도 있습니다.

여기에서 AAR 파일을 프로젝트로 직접 가져오거나 사용자 정의 AAR 파일을 로컬 Maven 리포지토리에 게시할 수 있습니다.

```sh
mvn install:install-file \
  -Dfile=bazel-bin/tmp/tensorflow-lite.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite -Dversion=0.1.100 -Dpackaging=aar
mvn install:install-file \
  -Dfile=bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite-select-tf-ops -Dversion=0.1.100 -Dpackaging=aar
```

마지막으로, 앱의 `build.gradle`에서 `mavenLocal()` 종속성이 있는지 확인하고, 표준 TensorFlow Lite 종속성을 특정 TensorFlow ops를 지원하는 종속성으로 바꿉니다.

```build
allprojects {
    repositories {
        mavenCentral()
        maven {  // Only for snapshot artifacts
            name 'ossrh-snapshot'
            url 'https://oss.sonatype.org/content/repositories/snapshots'
        }
        mavenLocal()
    }
}

dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.1.100'
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:0.1.100'
}
```

### iOS

#### CocoaPods 사용하기

`TensorFlowLiteSwift` 또는 `TensorFlowLiteObjC` CocoaPods와 함께 사용할 수 있는 `armv7` 및 `arm64`에 대해 야간에 사전 빌드되는 특정 TF ops CocoaPods가 제공됩니다.

*참고*: `x86_64` 시뮬레이터에서 선택한 TF ops를 사용해야 하는 경우, 선택한 ops 프레임워크를 직접 빌드할 수 있습니다. 자세한 내용은 [Bazel + Xcode 사용](#using_bazel_xcode) 섹션을 참조하세요.

```ruby
# In your Podfile target:
  pod 'TensorFlowLiteSwift'   # or 'TensorFlowLiteObjC'
  pod 'TensorFlowLiteSelectTfOps', '~> 0.0.1-nightly'
```

`pod install`을 실행한 후에는 추가 링커 플래그를 제공하여 특정 TF ops 프레임워크를 프로젝트에 강제로 로드해야 합니다. Xcode 프로젝트에서 `Build Settings` -&gt; `Other Linker Flags`로 이동하여 다음을 추가합니다.

```text
-force_load $(SRCROOT)/Pods/TensorFlowLiteSelectTfOps/Frameworks/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps
```

그러면 iOS 앱에서 `SELECT_TF_OPS`로 변환된 모든 모델을 실행할 수 있습니다. 예를 들어, [이미지 분류 iOS 앱](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios)을 수정하여 특정 TF ops 기능을 테스트할 수 있습니다.

- 모델 파일을 `SELECT_TF_OPS`가 활성화된 상태에서 변환된 파일로 바꿉니다.
- 지침에 따라 `TensorFlowLiteSelectTfOps` 종속성을 `Podfile`에 추가합니다.
- 위와 같이 추가 링커 플래그를 추가합니다.
- 예제 앱을 실행하고 모델이 올바르게 동작하는지 확인합니다.

#### Bazel + Xcode 사용하기

iOS를 위한 특정 TensorFlow ops를 포함한 TensorFlow Lite는 Bazel을 사용하여 빌드할 수 있습니다. 먼저, [iOS 빌드 지침](build_ios.md)에 따라 Bazel 작업 공간과 `.bazelrc` 파일을 올바르게 구성합니다.

iOS 지원이 활성화된 상태로 작업 공간을 구성한 후에는 다음 명령을 사용하여 일반 `TensorFlowLiteC.framework`에 추가할 수 있는 특정 TF ops 애드온 프레임워크를 빌드할 수 있습니다. 특정 TF ops 프레임워크는 `i386` 아키텍처용으로 빌드할 수 없으므로, `i386`이 아닌 대상 아키텍처의 목록을 명시적으로 제공할 필요가 있습니다.

```sh
bazel build -c opt --config=ios --ios_multi_cpus=armv7,arm64,x86_64 \
  //tensorflow/lite/ios:TensorFlowLiteSelectTfOps_framework
```

그러면 `bazel-bin/tensorflow/lite/ios/` 디렉터리 아래에 프레임워크가 생성됩니다. iOS 빌드 가이드의 [Xcode 프로젝트 설정](./build_ios.md#modify_xcode_project_settings_directly) 섹션에 설명된 유사한 단계에 따라 이 새 프레임워크를 Xcode 프로젝트에 추가할 수 있습니다.

앱 프로젝트에 프레임워크를 추가한 후, 앱 프로젝트에 추가 링커 플래그를 지정하여 특정 TF ops 프레임워크를 강제로 로드해야 합니다. Xcode 프로젝트에서 `Build Settings` -&gt; `Other Linker Flags`로 이동하여 다음을 추가합니다.

```text
-force_load <path/to/your/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps>
```

### C/C++

If you're using Bazel or [CMake](https://www.tensorflow.org/lite/guide/build_cmake) to build TensorFlow Lite interpreter, you can enable Flex delegate by linking a TensorFlow Lite Flex delegate shared library. You can build it with Bazel as the following command.

```
bazel build -c opt --config=monolithic tensorflow/lite/delegates/flex:tensorflowlite_flex
```

This command generates the following shared library in `bazel-bin/tensorflow/lite/delegates/flex`.

Platform | Library name
--- | ---
Linux | libtensorflowlite_flex.so
macOS | libtensorflowlite_flex.dylib
Windows | tensorflowlite_flex.dll

Note that the necessary `TfLiteDelegate` will be installed automatically when creating the interpreter at runtime as long as the shared library is linked. It is not necessary to explicitly install the delegate instance as is typically required with other delegate types.

**Note:** This feature is available since version 2.7.

### Python

특정 TensorFlow ops가 포함된 TensorFlow Lite가 [TensorFlow pip 패키지](https://www.tensorflow.org/install/pip)와 함께 자동으로 설치됩니다. [TensorFlow Lite Interpreter pip 패키지](https://www.tensorflow.org/lite/guide/python#install_just_the_tensorflow_lite_interpreter)만 설치하도록 선택할 수도 있습니다.

참고: 특정 TensorFlow ops가 포함된 TensorFlow Lite는 Linux의 경우 2.3, 기타 환경의 경우 2.4 이후부터 TensorFlow pip 패키지 버전에서 사용할 수 있습니다.

## 메트릭

### 성능

내장 및 특정 TensorFlow ops를 혼합하여 사용하는 경우, 동일한 TensorFlow Lite 최적화 및 최적화된 내장 커널을 모두 사용할 수 있으며 변환된 모델과 함께 사용할 수 있습니다.

다음 표는 Pixel 2의 MobileNet에서 추론을 실행하는 데 걸린 평균 시간을 설명합니다. 다음은 100회 실행한 평균 시간입니다. 이러한 연산 작업은 `--config=android_arm64 -c opt` 플래그를 사용하여 Android용으로 빌드되었습니다.

빌드 | 시간(밀리 초)
--- | ---
내장 ops만(`TFLITE_BUILTIN`) | 260.7
TF ops만 사용( `SELECT_TF_OPS`) | 264.5

### Binary size

다음 표는 각 빌드에 대한 TensorFlow Lite의 바이너리 크기를 설명합니다. 대상 ops는 `--config=android_arm -c opt`를 사용하여 Android용으로 빌드되었습니다.

빌드 | C++ 바이너리 크기 | Android APK Size
--- | --- | ---
내장 ops만 | 796 KB | 561 KB
내장 ops + TF ops | 23.0 MB | 8.0 MB
내장 ops + TF ops (1) | 4.1 MB | 1.8 MB

(1) 이러한 라이브러리는 8개의 TFLite 내장 ops 및 3개의 Tensorflow ops가 있는 [i3d-kinetics-400 모델](https://tfhub.dev/deepmind/i3d-kinetics-400/1)에 맞게 선택적으로 빌드됩니다. 자세한 내용은 [TensorFlow Lite 바이너리 크기 줄이기](../guide/reduce_binary_size.md) 섹션을 참조하세요.

## 알려진 제한 사항

- 지원되지 않는 유형: 특정 TensorFlow ops는 일반적으로 TensorFlow에서 사용할 수 있는 전체 입력/출력 유형을 지원하지 않을 수 있습니다.

## 업데이트

- 버전 2.6
    - GraphDef 속성 기반 연산자 및 HashTable 리소스 초기화에 대한 지원이 향상되었습니다.
- 버전 2.5
    - [훈련 후 양자화](../performance/post_training_quantization.md)라고 하는 최적화를 적용할 수 있습니다.
- 버전 2.4
    - 하드웨어 가속 대리자와의 호환성이 향상되었습니다.
