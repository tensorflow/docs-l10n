# Android 빠른 시작

To get started with TensorFlow Lite on Android, we recommend exploring the following example.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android">Android 이미지 분류의 예</a>

소스 코드에 대한 설명은 [TensorFlow Lite Android 이미지 분류](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/EXPLORE_THE_CODE.md)를 읽어보세요.

이 예제 앱은 [이미지 분류](https://www.tensorflow.org/lite/models/image_classification/overview)를 사용하여 기기의 후면 카메라에서 보여지는 내용을 지속적으로 분류합니다. 이 앱은 기기 또는 에뮬레이터에서 실행할 수 있습니다.

Inference is performed using the TensorFlow Lite Java API and the [TensorFlow Lite Android Support Library](../inference_with_metadata/lite_support.md). The demo app classifies frames in real-time, displaying the top most probable classifications. It allows the user to choose between a floating point or [quantized](https://www.tensorflow.org/lite/performance/post_training_quantization) model, select the thread count, and decide whether to run on CPU, GPU, or via [NNAPI](https://developer.android.com/ndk/guides/neuralnetworks).

Note: Additional Android applications demonstrating TensorFlow Lite in a variety of use cases are available in [Examples](https://www.tensorflow.org/lite/examples).

## Android Studio에서 빌드하기

To build the example in Android Studio, follow the instructions in [README.md](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md).

## Create your own Android app

고유한 Android 코드 작성을 빠르게 시작하려면 [Android 이미지 분류 예](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android)를 출발점으로 이용하는 것이 좋습니다.

The following sections contain some useful information for working with TensorFlow Lite on Android.

### Use Android Studio ML Model Binding

Note: Required [Android Studio 4.1](https://developer.android.com/studio) or above

To import a TensorFlow Lite (TFLite) model:

1. Right-click on the module you would like to use the TFLite model or click on `File`, then `New` &gt; `Other` &gt; `TensorFlow Lite Model` ![Right-click menus to access the TensorFlow Lite import functionality](../images/android/right_click_menu.png)

2. Select the location of your TFLite file. Note that the tooling will configure the module's dependency on your behalf with ML Model binding and all dependencies automatically inserted into your Android module's `build.gradle` file.

    Optional: Select the second checkbox for importing TensorFlow GPU if you want to use [GPU acceleration](../performance/gpu). ![Import dialog for TFLite model](../images/android/import_dialog.png)

3. Click `Finish`.

4. The following screen will appear after the import is successful. To start using the model, select Kotlin or Java, copy and paste the code under the `Sample Code` section. You can get back to this screen by double clicking the TFLite model under the `ml` directory in Android Studio. ![Model details page in Android Studio](../images/android/model_details.png)

### Use the TensorFlow Lite Task Library

TensorFlow Lite Task Library contains a set of powerful and easy-to-use task-specific libraries for app developers to create ML experiences with TFLite. It provides optimized out-of-box model interfaces for popular machine learning tasks, such as image classification, question and answer, etc. The model interfaces are specifically designed for each task to achieve the best performance and usability. Task Library works cross-platform and is supported on Java, C++, and Swift (coming soon).

To use the Task Library in your Android app, we recommend using the AAR hosted at MavenCentral for [Task Vision library](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-vision) and [Task Text library](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-text) , respectively.

다음과 같이 `build.gradle` 종속성에서 이 요소를 지정할 수 있습니다.

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-task-vision:0.3.0'
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.3.0'
    implementation 'org.tensorflow:tensorflow-lite-task-audio:0.3.0'
}
```

To use nightly snapshots, make sure that you have added [Sonatype snapshot repository](./build_android#use_nightly_snapshots).

See the introduction in the [TensorFlow Lite Task Library overview](../inference_with_metadata/task_library/overview.md) for more details.

### TensorFlow Lite Android 지원 라이브러리 사용하기

TensorFlow Lite Android 지원 라이브러리를 사용하면 모델을 애플리케이션에 통합하기가 쉬워집니다. 이 라이브러리는 원시 입력 데이터를 모델에 필요한 형식으로 변환하고 모델의 출력을 해석하여 필요한 상용구 코드의 양을 줄이는 고급 API를 제공합니다.

이미지 및 배열을 포함하여 입력 및 출력에 대해 공통 데이터 형식이 지원됩니다. 또한 이미지 크기 조정 및 자르기와 같은 작업을 수행하는 전처리 및 후 처리 기능도 제공됩니다.

To use the Support Library in your Android app, we recommend using the [TensorFlow Lite Support Library AAR hosted at MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-support).

다음과 같이 `build.gradle` 종속성에서 이 요소를 지정할 수 있습니다.

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-support:0.3.0'
}
```

To use nightly snapshots, make sure that you have added [Sonatype snapshot repository](./build_android#use_nightly_snapshots).

To get started, follow the instructions in the [TensorFlow Lite Android Support Library](../inference_with_metadata/lite_support.md).

### Use the TensorFlow Lite AAR from MavenCentral

To use TensorFlow Lite in your Android app, we recommend using the [TensorFlow Lite AAR hosted at MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite).

다음과 같이 `build.gradle` 종속성에서 이 요소를 지정할 수 있습니다.

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
}
```

To use nightly snapshots, make sure that you have added [Sonatype snapshot repository](./build_android#use_nightly_snapshots).

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

## Build Android app using C++

NDK로 앱을 빌드하는 경우, C++를 통해 TFLite를 사용하는 두 가지 방법이 있습니다.

### TFLite C API 사용하기

This is the *recommended* approach. Download the [TensorFlow Lite AAR hosted at MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow/tensorflow-lite), rename it to `tensorflow-lite-*.zip`, and unzip it. You must include the four header files in `headers/tensorflow/lite/` and `headers/tensorflow/lite/c/` folder and the relevant `libtensorflowlite_jni.so` dynamic library in `jni/` folder in your NDK project.

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

## Min SDK version of TFLite

Library | `minSdkVersion` | Device Requirements
--- | --- | ---
tensorflow-lite | 19 | NNAPI usage requires
:                             :                 : API 27+                : |  |
tensorflow-lite-gpu | 19 | GLES 3.1 or OpenCL
:                             :                 : (typically only        : |  |
:                             :                 : available on API 21+   : |  |
tensorflow-lite-hexagon | 19 | -
tensorflow-lite-support | 19 | -
tensorflow-lite-task-vision | 21 | android.graphics.Color
:                             :                 : related API requires   : |  |
:                             :                 : API 26+                : |  |
tensorflow-lite-task-text | 21 | -
tensorflow-lite-task-audio | 23 | -
tensorflow-lite-metadata | 19 | -
