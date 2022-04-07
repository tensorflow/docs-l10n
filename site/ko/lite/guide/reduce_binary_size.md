# TensorFlow Lite 바이너리 크기 줄이기

## 개요

기기 내 머신러닝(ODML) 애플리케이션용 모델을 배포할 때 모바일 기기에서 사용할 수 있는 제한된 메모리를 고려하는 것이 중요합니다. 모델의 바이너리 크기는 모델에 사용된 연산의 수와 밀접한 관련이 있습니다. TensorFlow Lite를 사용하면 선택적 빌드를 사용하여 모델의 바이너리 크기를 줄일 수 있습니다. 선택적 빌드는 모델 집합에서 사용되지 않는 연산을 건너뛰고 모바일 기기에서 모델을 실행하는 데 필요한 런타임 및 연산 커널만 있는 간결한 라이브러리를 생성합니다.

선택적 빌드는 다음 세 가지 연산 라이브러리에 적용됩니다.

1. [TensorFlow Lite built-in ops library](https://www.tensorflow.org/lite/guide/ops_compatibility)
2. [TensorFlow Lite custom ops](https://www.tensorflow.org/lite/guide/ops_custom)
3. [Select TensorFlow ops library](https://www.tensorflow.org/lite/guide/ops_select)

아래 표는 몇 가지 일반적인 사용 사례에서 선택적 빌드의 영향을 보여줍니다.

<table>
  <thead>
    <tr>
      <th>모델 이름</th>
      <th>도메인</th>
      <th>대상 아키텍처</th>
      <th>AAR 파일 크기</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="2">
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(float)</a>
    </td>
    <td rowspan="2">이미지 분류</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar(296,635바이트)</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar(382,892바이트)</td>
  </tr>
  <tr>
    <td rowspan="2">
      <a href="https://tfhub.dev/google/lite-model/spice/">SPICE</a>
    </td>
    <td rowspan="2">음높이 추출</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar(375,813바이트) <br>tensorflow-lite-select-tf-ops.aar(1,676,380바이트)</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar(421,826바이트) <br>tensorflow-lite-select-tf-ops.aar(2,298,630바이트)</td>
  </tr>
  <tr>
    <td rowspan="2">
      <a href="https://tfhub.dev/deepmind/i3d-kinetics-400/1">i3d-kinetics-400</a>
    </td>
    <td rowspan="2">비디오 분류</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar(240,085바이트) <br>tensorflow-lite-select-tf-ops.aar(1,708,597바이트)</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar(273,713바이트) <br>tensorflow-lite-select-tf-ops.aar(2,339,697바이트)</td>
  </tr>
 </table>

참고: 이 기능은 현재 실험적 단계로, 버전 2.4부터 사용할 수 있으며 변경될 수 있습니다.

## Bazel을 사용하여 TensorFlow Lite를 선택적으로 빌드하기

이 섹션에서는 TensorFlow 소스 코드를 다운로드하고 Bazel에 [로컬 개발 환경을 설정](https://www.tensorflow.org/lite/guide/android#build_tensorflow_lite_locally)했다고 가정합니다.

### Android 프로젝트용 AAR 파일 빌드하기

다음과 같이 모델 파일 경로를 제공하여 사용자 정의 TensorFlow Lite AAR을 빌드할 수 있습니다.

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

위의 명령은 TensorFlow Lite 내장 및 사용자 정의 연산에 대한 AAR 파일 `bazel-bin/tmp/tensorflow-lite.aar`를 생성합니다. 또한, 모델에 Select TensorFlow 연산이 포함된 경우, 선택적으로 aar 파일 `bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar`을 생성합니다. 그 결과, 여러 개의 아키텍처가 있는 "뚱뚱한" AAR이 빌드됩니다. 아키텍처가 모두 필요하지 않다면 배포 환경에 적합하게 일부만 사용하세요.

### 사용자 정의 연산으로 빌드하기

사용자 정의 연산이 있는 Tensorflow Lite 모델을 개발한 경우, 빌드 명령에 다음 플래그를 추가하여 모델을 빌드할 수 있습니다.

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a \
  --tflite_custom_ops_srcs=/e/f/file1.cc,/g/h/file2.h \
  --tflite_custom_ops_deps=dep1,dep2
```

`tflite_custom_ops_srcs` 플래그에는 사용자 정의 연산의 소스 파일이 포함되고 `tflite_custom_ops_deps` 플래그에는 이러한 소스 파일을 빌드하기 위한 종속성이 포함됩니다. 이러한 종속성은 TensorFlow 리포지토리에 있어야 합니다.

### 고급 사용: 사용자 정의 Bazel 규칙

프로젝트에서 Bazel을 사용 중이고 주어진 모델 세트에 대해 사용자 정의 TFLite 종속성을 정의하려는 경우, 프로젝트 저장소에서 다음 규칙을 정의할 수 있습니다.

내장 연산이 있는 모델의 경우만:

```bazel
load(
    "@org_tensorflow//tensorflow/lite:build_def.bzl",
    "tflite_custom_android_library",
    "tflite_custom_c_library",
    "tflite_custom_cc_library",
)

# A selectively built TFLite Android library.
tflite_custom_android_library(
    name = "selectively_built_android_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)

# A selectively built TFLite C library.
tflite_custom_c_library(
    name = "selectively_built_c_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)

# A selectively built TFLite C++ library.
tflite_custom_cc_library(
    name = "selectively_built_cc_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)
```

[Select TF 연산자](../guide/ops_select.md)가 있는 모델의 경우:

```bazel
load(
    "@org_tensorflow//tensorflow/lite/delegates/flex:build_def.bzl",
    "tflite_flex_android_library",
    "tflite_flex_cc_library",
)

# A Select TF ops enabled selectively built TFLite Android library.
tflite_flex_android_library(
    name = "selective_built_tflite_flex_android_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)

# A Select TF ops enabled selectively built TFLite C++ library.
tflite_flex_cc_library(
    name = "selective_built_tflite_flex_cc_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)
```

### 고급 사용: 사용자 정의 C/C++ 공유 라이브러리 빌드하기

주어진 모델에 대해 자체적인 사용자 정의 TFLite C/C++ 공유 객체를 빌드하려면 다음 단계를 따르세요.

TensorFlow 소스 코드의 루트 디렉터리에서 다음 명령을 실행하여 임시 BUILD 파일을 만듭니다.

```sh
mkdir -p tmp && touch tmp/BUILD
```

#### 사용자 정의 C 공유 객체 빌드하기

사용자 정의 TFLite C 공유 객체를 빌드하려면 `tmp/BUILD` 파일에 다음을 추가합니다.

```bazel
load(
    "//tensorflow/lite:build_def.bzl",
    "tflite_custom_c_library",
    "tflite_cc_shared_object",
)

tflite_custom_c_library(
    name = "selectively_built_c_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)

# Generates a platform-specific shared library containing the TensorFlow Lite C
# API implementation as define in `c_api.h`. The exact output library name
# is platform dependent:
#   - Linux/Android: `libtensorflowlite_c.so`
#   - Mac: `libtensorflowlite_c.dylib`
#   - Windows: `tensorflowlite_c.dll`
tflite_cc_shared_object(
    name = "tensorflowlite_c",
    linkopts = select({
        "//tensorflow:ios": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite/c:exported_symbols.lds)",
        ],
        "//tensorflow:macos": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite/c:exported_symbols.lds)",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script,$(location //tensorflow/lite/c:version_script.lds)",
        ],
    }),
    per_os_targets = True,
    deps = [
        ":selectively_built_c_lib",
        "//tensorflow/lite/c:exported_symbols.lds",
        "//tensorflow/lite/c:version_script.lds",
    ],
)
```

새로 추가된 대상은 다음과 같이 빌드할 수 있습니다.

```sh
bazel build -c opt --cxxopt=--std=c++14 \
  //tmp:tensorflowlite_c
```

Android의 경우(64bit의 경우 `android_arm`을 `android_arm64`로 교체):

```sh
bazel build -c opt --cxxopt=--std=c++14 --config=android_arm \
  //tmp:tensorflowlite_c
```

#### 사용자 정의 C++ 공유 객체 빌드하기

사용자 정의 TFLite C++ 공유 객체를 빌드하려면 `tmp/BUILD` 파일에 다음을 추가합니다.

```bazel
load(
    "//tensorflow/lite:build_def.bzl",
    "tflite_custom_cc_library",
    "tflite_cc_shared_object",
)

tflite_custom_cc_library(
    name = "selectively_built_cc_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)

# Shared lib target for convenience, pulls in the core runtime and builtin ops.
# Note: This target is not yet finalized, and the exact set of exported (C/C++)
# APIs is subject to change. The output library name is platform dependent:
#   - Linux/Android: `libtensorflowlite.so`
#   - Mac: `libtensorflowlite.dylib`
#   - Windows: `tensorflowlite.dll`
tflite_cc_shared_object(
    name = "tensorflowlite",
    # Until we have more granular symbol export for the C++ API on Windows,
    # export all symbols.
    features = ["windows_export_all_symbols"],
    linkopts = select({
        "//tensorflow:macos": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite:tflite_exported_symbols.lds)",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-Wl,-z,defs",
            "-Wl,--version-script,$(location //tensorflow/lite:tflite_version_script.lds)",
        ],
    }),
    per_os_targets = True,
    deps = [
        ":selectively_built_cc_lib",
        "//tensorflow/lite:tflite_exported_symbols.lds",
        "//tensorflow/lite:tflite_version_script.lds",
    ],
)
```

새로 추가된 대상은 다음과 같이 빌드할 수 있습니다.

```sh
bazel build -c opt  --cxxopt=--std=c++14 \
  //tmp:tensorflowlite
```

Android의 경우(64bit의 경우 `android_arm`을 `android_arm64`로 교체):

```sh
bazel build -c opt --cxxopt=--std=c++14 --config=android_arm \
  //tmp:tensorflowlite
```

Select TF 연산자가 있는 모델의 경우, 다음 공유 라이브러리도 빌드해야 합니다.

```bazel
load(
    "@org_tensorflow//tensorflow/lite/delegates/flex:build_def.bzl",
    "tflite_flex_shared_library"
)

# Shared lib target for convenience, pulls in the standard set of TensorFlow
# ops and kernels. The output library name is platform dependent:
#   - Linux/Android: `libtensorflowlite_flex.so`
#   - Mac: `libtensorflowlite_flex.dylib`
#   - Windows: `libtensorflowlite_flex.dll`
tflite_flex_shared_library(
  name = "tensorflowlite_flex",
  models = [
      ":model_one.tflite",
      ":model_two.tflite",
  ],
)

```

새로 추가된 대상은 다음과 같이 빌드할 수 있습니다.

```sh
bazel build -c opt --cxxopt='--std=c++14' \
      --config=monolithic \
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
      //tmp:tensorflowlite_flex
```

Android의 경우(64bit의 경우 `android_arm`을 `android_arm64`로 교체):

```sh
bazel build -c opt --cxxopt='--std=c++14' \
      --config=android_arm \
      --config=monolithic \
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
      //tmp:tensorflowlite_flex
```

## Docker를 사용하여 TensorFlow Lite를 선택적으로 빌드하기

이 섹션에서는 로컬 머신에 [Docker](https://docs.docker.com/get-docker/)를 설치하고 [여기](https://www.tensorflow.org/lite/guide/build_android#set_up_build_environment_using_docker)에서 TensorFlow Lite Dockerfile을 다운로드했다고 가정합니다.

위의 Dockerfile을 다운로드한 후, 다음을 실행하여 Docker 이미지를 빌드할 수 있습니다.

```shell
docker build . -t tflite-builder -f tflite-android.Dockerfile
```

### Android 프로젝트용 AAR 파일 빌드하기

다음을 실행하여 Docker로 빌드하기 위한 스크립트를 다운로드합니다.

```sh
curl -o build_aar_with_docker.sh \
  https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/tools/build_aar_with_docker.sh &&
chmod +x build_aar_with_docker.sh
```

그런 다음, 아래와 같이 모델 파일 경로를 제공하여 사용자 정의 TensorFlow Lite AAR을 빌드할 수 있습니다.

```sh
sh build_aar_with_docker.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a \
  --checkpoint=master \
  [--cache_dir=<path to cache directory>]
```

`checkpoint` 플래그는 라이브러리를 빌드하기 전에 체크아웃하려는 TensorFlow 리포지토리의 커밋, 분기 또는 태그입니다. 기본적으로 이것이 최신 릴리스 분기입니다. 위의 명령으로 TensorFlow Lite 내장 및 사용자 정의 ops를 위한 AAR 파일 `tensorflow-lite.aar`, 그리고 선택적으로 현재 디렉터리에 있는 특정 TensorFlow ops를 위한 AAR 파일 `tensorflow-lite-select-tf-ops.aar`이 생성됩니다.

--cache_dir은 캐시 디렉터리를 지정합니다. 제공되지 않은 경우 스크립트는 캐싱을 위해 현재 작업 디렉터리 아래에 `bazel-build-cache`라는 디렉터리를 만듭니다.

## 프로젝트에 AAR 파일 추가하기

[AAR을 프로젝트로 직접 가져오거나](https://www.tensorflow.org/lite/guide/android#add_aar_directly_to_project), [로컬 Maven 리포지토리에 사용자 정의 AAR을 게시](https://www.tensorflow.org/lite/guide/android#install_aar_to_local_maven_repository)하여 AAR 파일을 추가합니다. `tensorflow-lite-select-tf-ops.aar`에 대한 AAR 파일을 생성했다면 이 파일도 추가해야 합니다.

## iOS에 대한 선택적 빌드

[로컬로 빌드하기 섹션](../guide/build_ios.md#building_locally)을 참조하여 빌드 환경을 설정하고 TensorFlow 작업 공간을 구성한 다음, [가이드](../guide/build_ios.md#selectively_build_tflite_frameworks)에 따라 iOS용 선택적 빌드 스크립트를 사용하세요.
