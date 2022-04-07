# Android 빠른 시작

Android에서 TensorFlow Lite를 시작하려면 다음 예제를 살펴볼 것을 권장합니다.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android">Android 이미지 분류의 예</a>

소스 코드에 대한 설명은 [TensorFlow Lite Android 이미지 분류](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/EXPLORE_THE_CODE.md)를 읽어보세요.

이 예제 앱은 [이미지 분류](https://www.tensorflow.org/lite/models/image_classification/overview)를 사용하여 기기의 후면 카메라에서 보여지는 내용을 지속적으로 분류합니다. 이 앱은 기기 또는 에뮬레이터에서 실행할 수 있습니다.

추론은 TensorFlow Lite Java API 및 [TensorFlow Lite Android 지원 라이브러리](../inference_with_metadata/lite_support.md)를 사용하여 수행됩니다. 데모 앱은 프레임을 실시간으로 분류하여 가장 가능성이 높은 분류를 표시합니다. 그러면 사용자는 부동 소수점 또는 [양자화](https://www.tensorflow.org/lite/performance/post_training_quantization) 모델 중에서 선택하고, 스레드 수를 선택하고, CPU, GPU 및 [NNAPI](https://developer.android.com/ndk/guides/neuralnetworks) 중 어디서 실행할지 결정할 수 있습니다.

참고: 다양한 사용 사례에서 TensorFlow Lite의 사용을 시연하는 추가 Android 애플리케이션을 [예제](https://www.tensorflow.org/lite/examples)에서 확인할 수 있습니다.

## Android Studio에서 빌드하기

Android Studio에서 예제를 빌드하려면 [README.md](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md)의 안내를 따릅니다.

## 고유한 Android 앱 만들기

고유한 Android 코드 작성을 빠르게 시작하려면 [Android 이미지 분류 예](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android)를 출발점으로 이용하는 것이 좋습니다.

이어지는 섹션에서 Android에서 TensorFlow Lite를 사용할 때 유용한 몇 가지 정보를 제공합니다.

### Android Studio ML 모델 바인딩 사용

참고: [Android Studio 4.1](https://developer.android.com/studio) 이상 필요

TensorFlow Lite(TFLite) 모델을 가져오려면 다음과 같이 합니다.

1. TFLite 모델을 사용하려는 모듈을 마우스 오른쪽 버튼으로 클릭하거나 `파일`, `새로 만들기` &gt; `기타` &gt; `TensorFlow Lite 모델` ![Right-click menus to access the TensorFlow Lite import functionality](../images/android/right_click_menu.png)을 클릭합니다.

2. TFLite 파일의 위치를 선택합니다. 도구가 ML 모델을 바인딩하고 모든 종속성은 Android 모듈의 `build.gradle` 파일에 자동으로 삽입하는 등 사용자를 대신하여 모듈의 종속성을 구성합니다.

    선택 사항: [GPU 가속](../performance/gpu)을 사용하려는 경우 TensorFlow GPU를 가져오기 위한 두 번째 확인란을 선택합니다. ![Import dialog for TFLite model](../images/android/import_dialog.png)

3. `Finish`를 클릭합니다.

4. 가져오기가 성공하면 다음 화면이 나타납니다. 모델 사용을 시작하려면 Kotlin 또는 Java를 선택하고 `Sample Code` 섹션 아래에 코드를 복사하여 붙여 넣습니다. Android Studio의 `ml` 디렉터리 아래에 있는 TFLite 모델을 두 번 클릭하여 이 화면으로 돌아갈 수 있습니다. ![Model details page in Android Studio](../images/android/model_details.png)

### TensorFlow Lite Task 라이브러리 사용

TensorFlow Lite Task 라이브러리에는 앱 개발자가 TFLite로 ML 경험을 만들 수 있는 강력하고 사용하기 쉬운 작업별 라이브러리 세트가 포함되어 있습니다. 이미지 분류, 질문 및 답변 등과 같은 주요 머신 러닝 작업에 최적화된 기본 제공 모델 인터페이스가 제공됩니다. 모델 인터페이스는 각 작업에 맞게 특별히 설계되어 최상의 성능과 유용성을 제공합니다. Task 라이브러리는 크로스 플랫폼에서 작동하며 Java, C++ 및 Swift(곧 제공 예정)에서 지원됩니다.

Android 앱에서 Task 라이브러리를 사용하려면 각각 [Task Vision 라이브러리](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-vision) 및 [Task Text 라이브러리](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-text)에 대해 MavenCentral에서 호스팅되는 AAR을 사용하는 것이 좋습니다.

다음과 같이 `build.gradle` 종속성에서 이 요소를 지정할 수 있습니다.

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-task-vision:0.3.0'
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.3.0'
    implementation 'org.tensorflow:tensorflow-lite-task-audio:0.3.0'
}
```

야간 스냅샷을 사용하려면 [Sonatype 스냅샷 저장소](./build_android#use_nightly_snapshots)를 추가했는지 확인하세요.

자세한 내용은 [TensorFlow Lite 작업 라이브러리 개요](../inference_with_metadata/task_library/overview.md)의 소개를 참조하세요.

### TensorFlow Lite Android 지원 라이브러리 사용하기

TensorFlow Lite Android 지원 라이브러리를 사용하면 모델을 애플리케이션에 통합하기가 쉬워집니다. 이 라이브러리는 원시 입력 데이터를 모델에 필요한 형식으로 변환하고 모델의 출력을 해석하여 필요한 상용구 코드의 양을 줄이는 고급 API를 제공합니다.

이미지 및 배열을 포함하여 입력 및 출력에 대해 공통 데이터 형식이 지원됩니다. 또한 이미지 크기 조정 및 자르기와 같은 작업을 수행하는 전처리 및 후 처리 기능도 제공됩니다.

Android 앱에서 지원 라이브러리를 사용하려면 [MavenCentral에서 호스팅되는 TensorFlow Lite 지원 라이브러리 AAR](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-support)의 사용을 권장합니다.

다음과 같이 `build.gradle` 종속성에서 이 요소를 지정할 수 있습니다.

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-support:0.3.0'
}
```

야간 스냅샷을 사용하려면 [Sonatype 스냅샷 저장소](./build_android#use_nightly_snapshots)를 추가했는지 확인하세요.

시작하려면 [TensorFlow Lite Android 지원 라이브러리](../inference_with_metadata/lite_support.md)의 지침을 따르세요.

### MavenCentral에서 TensorFlow Lite AAR 사용하기

Android 앱에서 TensorFlow Lite를 사용하려면 [MavenCentral에서 호스팅되는 TensorFlow Lite AAR](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite)을 사용하는 것이 좋습니다.

다음과 같이 `build.gradle` 종속성에서 이 요소를 지정할 수 있습니다.

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
}
```

야간 스냅샷을 사용하려면 [Sonatype 스냅샷 저장소](./build_android#use_nightly_snapshots)를 추가했는지 확인하세요.

이 AAR에는 모든 [Android ABI](https://developer.android.com/ndk/guides/abis)에 대한 바이너리가 포함되어 있습니다. 지원해야 하는 ABI만 포함하여 애플리케이션의 바이너리 크기를 줄일 수 있습니다.

대부분의 개발자는 `x86`, `x86_64` 및 `arm32` ABI를 생략하는 것이 좋습니다. 이를 위해 대부분의 최신 Android 기기에 적용되는 `armeabi-v7a` 및 `arm64-v8a`를 포함한 다음 Gradle 구성을 이용할 수 있습니다.

```build
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

`abiFilters`에 대해 자세히 알아보려면 Android Gradle 설명서에서 [`NdkOptions`](https://google.github.io/android-gradle-dsl/current/com.android.build.gradle.internal.dsl.NdkOptions.html)를 참조하세요.

## C++를 사용하여 Android 앱 빌드하기

NDK로 앱을 빌드하는 경우, C++를 통해 TFLite를 사용하는 두 가지 방법이 있습니다.

### TFLite C API 사용하기

이 방법이 *권장*됩니다. [MavenCentral에서 호스팅되는 TensorFlow Lite AAR](https://search.maven.org/artifact/org.tensorflow/tensorflow/tensorflow-lite)을 다운로드하고 이름을 `tensorflow-lite-*.zip`으로 바꾼 다음 압축을 풉니다. NDK 프로젝트에서 `headers/tensorflow/lite/` 및 `headers/tensorflow/lite/c/` 폴더에 네 개의 헤더 파일을 포함하고 `jni/` 폴더에 관련 `libtensorflowlite_jni.so` 동적 라이브러리를 포함해야 합니다.

`c_api.h` 헤더 파일에는 TFLite C API 사용을 위한 기본 설명서가 포함되어 있습니다.

### TFLite C++ API 사용하기

C++ API를 통해 TFLite를 사용하려는 경우, C++ 공유 라이브러리를 빌드할 수 있습니다.

32bit armeabi-v7a:

```sh
bazel build -c opt --config=android_arm //tensorflow/lite:libtensorflowlite.so
```

64bit arm64-v8a:

```sh
bazel build -c opt --config=android_arm64 //tensorflow/lite:libtensorflowlite.so
```

현재, 필요한 모든 헤더 파일을 추출하는 간단한 방법은 없으므로 모든 헤더 파일을 TensorFlow 리포지토리의 `tensorflow/lite/`에 포함해야 합니다. 또한 [FlatBuffers](https://github.com/google/flatbuffers) 및 [Abseil](https://github.com/abseil/abseil-cpp)의 헤더 파일도 필요합니다.

## TFLite의 최소 SDK 버전

라이브러리 | `minSdkVersion` | 장치 요구 사항
--- | --- | ---
tensorflow-lite | 19 | NNAPI 사용 필요
:                             :                 : API 27+                : |  |
tensorflow-lite-gpu | 19 | GLES 3.1 또는 OpenCL
:                             :                 : (일반적으로 유일        : |  |
:                             :                 : API 21+에서 사용 가능   : |  |
tensorflow-lite-hexagon | 19 | -
tensorflow-lite-support | 19 | -
tensorflow-lite-task-vision | 21 | android.graphics.Color
:                             :                 : 관련 API 필요   : |  |
:                             :                 : API 26+                : |  |
tensorflow-lite-task-text | 21 | -
tensorflow-lite-task-audio | 23 | -
tensorflow-lite-metadata | 19 | -
