# Android용 개발 도구

TensorFlow Lite는 모델을 Android 앱에 통합하기 위한 다양한 도구를 제공합니다. 이 페이지에서는 Kotlin, Java, C++로 앱을 빌드하는 데 사용하는 개발 도구와 Android Studio에서 TensorFlow Lite 개발 지원에 대해 설명합니다.

요점: 일반적으로, TensorFlow Lite를 Android 앱에 통합하기 위해 [TensorFlow Lite 작업 라이브러리](#task_library)를 사용해야 합니다(이 라이브러리가 원하는 사용 목적을 지원하지 않는 경우가 아니라면). 작업 라이브러리에서 지원하지 않는 경우에는 [TensorFlow Lite 라이브러리](#lite_lib)와 [지원 라이브러리](#support_lib)를 사용하세요.

Android 코드 작성을 빠르게 시작하려면 [Android 빠른 시작](../android/quickstart)을 참조하세요.

## Kotlin 및 Java로 빌드하기 위한 도구

다음 섹션에서는 Kotlin 및 Java 언어를 사용하는 TensorFlow Lite용 개발 도구에 대해 설명합니다.

### TensorFlow Lite 작업 라이브러리 {:#task_library}

TensorFlow Lite 작업 라이브러리에는 앱 개발자가 TensorFlow Lite로 빌드할 수 있는 강력하고 사용하기 쉬운 작업별 라이브러리 세트가 포함되어 있습니다. 이러한 라이브러리는 이미지 분류, 질문 및 답변 등과 같은 인기 있는 머신 러닝 작업에 최적화된 기본 모델 인터페이스를 제공합니다. 모델 인터페이스는 최상의 성능과 사용성을 달성하도록 각 작업에 대해 특별히 설계되었습니다. 작업 라이브러리는 크로스 플랫폼으로 작동하며 Java 및 C++에서 지원됩니다.

Android 앱에서 태스크 라이브러리를 사용하려면 각각 [Task Vision 라이브러리](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-vision), [Task Text 라이브러리](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-text) 및 [Task Audio 라이브러리](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-audio)용 MavenCentral의 AAR을 사용하세요.

다음과 같이 `build.gradle` 종속성에서 이를 지정할 수 있습니다.

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-task-vision:0.3.0'
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.3.0'
    implementation 'org.tensorflow:tensorflow-lite-task-audio:0.3.0'
}
```

야간 스냅샷을 사용하는 경우 [Sonatype 스냅샷 리포지토리](./lite_build#use_nightly_snapshots)를 프로젝트에 추가해야 합니다.

자세한 내용은 [TensorFlow Lite 작업 라이브러리 개요](../inference_with_metadata/task_library/overview.md) 개요를 참조하세요.

### TensorFlow Lite 라이브러리 {:#lite_lib}

[MavenCentral에서 호스팅되는 AAR](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite)을 개발 프로젝트에 추가하여 Android 앱에서 TensorFlow Lite 라이브러리를 사용합니다.

다음과 같이 `build.gradle` 종속성에서 이를 지정할 수 있습니다.

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
}
```

야간 스냅샷을 사용하는 경우 [Sonatype 스냅샷 리포지토리](./lite_build#use_nightly_snapshots)를 프로젝트에 추가해야 합니다.

이 AAR에는 모든 [Android ABI](https://developer.android.com/ndk/guides/abis)에 대한 바이너리가 포함되어 있습니다. 지원해야 하는 ABI만 포함하여 애플리케이션의 바이너리 크기를 줄일 수 있습니다.

특정 하드웨어를 대상으로 하지 않는 한 대부분의 경우 `x86`, `x86_64` 및 `arm32` ABI를 생략해야 합니다. 다음 Gradle 구성으로 이를 구성할 수 있습니다. 특히 여기에는 `armeabi-v7a` 및 `arm64-v8a`만 포함되며, 이것으로 대부분의 최신 Android 기기가 지원됩니다.

```build
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

`abiFilters`에 대해 자세히 알아보려면 Android NDK 설명서에서 [Android ABI](https://developer.android.com/ndk/guides/abis)를 참조하세요.

### TensorFlow Lite 지원 라이브러리 {:#support_lib}

TensorFlow Lite Android 지원 라이브러리를 사용하면 모델을 애플리케이션에 통합하기가 쉬워집니다. 이 라이브러리는 원시 입력 데이터를 모델에 필요한 형식으로 변환하고 모델의 출력을 해석하여 필요한 상용구 코드의 양을 줄이는 고급 API를 제공합니다.

이미지 및 배열을 포함하여 입력 및 출력에 대해 공통 데이터 형식이 지원됩니다. 또한 이미지 크기 조정 및 자르기와 같은 작업을 수행하는 전처리 및 후처리 기능도 제공됩니다.

[MavenCentral에서 호스팅되는 TensorFlow Lite 지원 라이브러리 AAR](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-support)을 포함하여 Android 앱에서 지원 라이브러리를 사용하세요.

다음과 같이 `build.gradle` 종속성에서 이를 지정할 수 있습니다.

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-support:0.3.0'
}
```

야간 스냅샷을 사용하는 경우 [Sonatype 스냅샷 리포지토리](./lite_build#use_nightly_snapshots)를 프로젝트에 추가해야 합니다.

시작하는 방법에 대한 지침은 [TensorFlow Lite Android 지원 라이브러리](../inference_with_metadata/lite_support.md)를 참조하세요.

### 라이브러리용 최소 Android SDK 버전

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

### Android Studio 사용하기

위에서 설명한 개발 라이브러리 외에도 Android Studio는 아래에 설명된 대로 TensorFlow Lite 모델 통합을 지원합니다.

#### Android Studio ML 모델 바인딩

Android Studio 4.1 이상의 ML 모델 바인딩 기능을 사용하면 `.tflite` 모델 파일을 기존 Android 앱으로 가져오고 인터페이스 클래스를 생성하여 코드를 모델과 더 쉽게 통합할 수 있습니다.

TensorFlow Lite(TFLite) 모델을 가져오려면 다음과 같이 합니다.

1. TFLite 모델을 사용하려는 모듈을 마우스 오른쪽 버튼으로 클릭하거나 **File(파일) &gt; New(새로 만들기) &gt; Other(기타) &gt; TensorFlow Lite 모델**을 클릭합니다.

2. TensorFlow Lite 파일의 위치를 선택합니다. 도구는 ML 모델 바인딩을 사용하여 모듈의 종속성을 구성하고 필요한 모든 종속성을 Android 모듈의 `build.gradle` 파일에 자동으로 추가합니다.

    참고: [GPU 가속](../performance/gpu)을 사용하려는 경우 TensorFlow GPU를 가져오기 위한 두 번째 확인란을 선택합니다.

3. `Finish`를 클릭하여 가져오기 프로세스를 시작합니다. 가져오기가 완료되면 도구는 입력 및 출력 텐서를 포함하여 모델을 설명하는 화면을 표시합니다.

4. 모델 사용을 시작하려면 Kotlin 또는 Java를 선택하고 **샘플 코드** 섹션에 코드를 복사하여 붙여넣습니다.

Android Studio의 `ml` 디렉터리 아래에 있는 TensorFlow Lite 모델을 더블 클릭하여 모델 정보 화면으로 돌아갈 수 있습니다. Android Studio의 Modle Binding 기능 사용에 대한 자세한 내용은 Android Studio [릴리스 정보](https://developer.android.com/studio/releases#4.1-tensor-flow-lite-models)를 참조하세요. Android Studio에서 모델 바인딩을 사용하는 방법에 대한 개요는 코드 예제 [지침](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md)을 참조하세요.

## C 및 C++로 빌드하기 위한 도구

TensorFlow Lite용 C 및 C++ 라이브러리는 주로 Android NDK(Native Development Kit)를 사용하여 앱을 빌드하는 개발자를 대상으로 합니다. NDK로 앱을 빌드하는 경우 C++를 통해 TFLite를 사용하는 두 가지 방법이 있습니다.

### TFLite C API

이 API 사용이 NDK를 사용하는 개발자에게 *권장*되는 접근 방식입니다. [MavenCentral에서 호스팅되는 TensorFlow Lite AAR](https://search.maven.org/artifact/org.tensorflow/tensorflow/tensorflow-lite) 파일을 다운로드하고 이름을 `tensorflow-lite-*.zip`으로 변경한 다음 압축을 풉니다. `headers/tensorflow/lite/` 및 `headers/tensorflow/lite/c/` 폴더에 4개의 헤더 파일을 포함하고 NDK 프로젝트의 `jni/` 폴더에 관련 `libtensorflowlite_jni.so` 동적 라이브러리를 포함해야 합니다.

`c_api.h` 헤더 파일에는 TFLite C API 사용을 위한 기본 설명서가 포함되어 있습니다.

### TFLite C++ API

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
