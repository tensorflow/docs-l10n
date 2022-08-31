# CMake로 TensorFlow Lite 빌드하기

이 페이지에서는 [CMake](https://cmake.org/) 도구로 TensorFlow Lite 라이브러리를 빌드하고 사용하는 방법을 설명합니다.

다음 지침은 Ubuntu 16.04.3 64비트 PC(AMD64), macOS Catalina(x86_64), Windows 10 및 TensorFlow devel Docker 이미지 [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/)에서 테스트되었습니다.

**참고:** 이 기능은 버전 2.4부터 사용할 수 있습니다.

### 1단계. CMake 도구 설치하기

CMake 3.16 이상이 필요합니다. Ubuntu에서는 간단히 다음 명령을 실행할 수 있습니다.

```sh
sudo apt-get install cmake
```

또는 [공식 cmake 설치 가이드](https://cmake.org/install/)를 따를 수 있습니다.

### Step 2. Clone TensorFlow repository

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**Note:** If you're using the TensorFlow Docker image, the repo is already provided in `/tensorflow_src/`.

### 3단계. CMake 빌드 디렉터리 만들기

```sh
mkdir tflite_build
cd tflite_build
```

### 4단계. 구성으로 CMake 도구 실행하기

#### 릴리스 빌드

기본적으로 최적화된 릴리스 바이너리가 생성됩니다. 워크스테이션용으로 빌드하려면 간단히 다음 명령을 실행하면 됩니다.

```sh
cmake ../tensorflow_src/tensorflow/lite
```

#### 디버그 빌드

기호 정보가 있는 디버그 빌드를 생성해야 하는 경우, `-DCMAKE_BUILD_TYPE=Debug` 옵션을 제공해야 합니다.

```sh
cmake ../tensorflow_src/tensorflow/lite -DCMAKE_BUILD_TYPE=Debug
```

#### 커널 단위 테스트로 빌드

커널 테스트를 실행하려면 '-DTFLITE_KERNEL_TEST=on' 플래그를 제공해야 합니다. 단위 테스트 교차 컴파일에 대한 세부 사항은 다음 하위 섹션에서 찾을 수 있습니다.

```sh
cmake ../tensorflow_src/tensorflow/lite -DTFLITE_KERNEL_TEST=on
```

#### 설치 가능한 패키지 빌드

`find_package(tensorflow-lite CONFIG)`를 사용하여 다른 CMake 프로젝트에서 종속성으로 사용할 수 있는 설치 가능한 패키지를 빌드하려면 `-DTFLITE_ENABLE_INSTALL=ON` 옵션을 사용하세요.

이상적으로는, 자체 버전의 라이브러리 종속성을 제공해야 합니다. 이는 TF Lite에 의존하는 프로젝트에서도 사용해야 합니다. `-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON`을 사용하고 라이브러리 설치를 가리키도록 `<PackageName>_DIR` 변수를 설정할 수 있습니다.

```sh
cmake ../tensorflow_src/tensorflow/lite -DTFLITE_ENABLE_INSTALL=ON \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON \
  -Dabsl_DIR=<install path>/lib/cmake/absl \
  -DEigen3_DIR=<install path>/share/eigen3/cmake \
  -DFlatbuffers_DIR=<install path>/lib/cmake/flatbuffers \
  -DNEON_2_SSE_DIR=<install path>/lib/cmake/NEON_2_SSE \
  -Dcpuinfo_DIR=<install path>/share/cpuinfo \
  -Druy_DIR=<install path>/lib/cmake/ruy
```

**참고:** 패키지 처리 및 찾기에 대한 자세한 내용은 [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html)에 대한 CMake 설명서를 참조하세요.

#### 교차 컴파일

CMake를 사용하여 ARM64 또는 Android 타겟 아키텍처용 바이너리를 빌드할 수 있습니다.

TF Lite를 교차 컴파일하려면 `-DCMAKE_TOOLCHAIN_FILE` 플래그를 사용하여 SDK(예: Android의 경우 ARM64 SDK 또는 NDK)에 대한 경로를 제공해야 합니다.

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=<CMakeToolchainFileLoc> ../tensorflow/lite/
```

##### Android 교차 컴파일의 세부 사항

Android 교차 컴파일의 경우 [Android NDK](https://developer.android.com/ndk)를 설치하고 위에서 언급한 `-DCMAKE_TOOLCHAIN_FILE` 플래그와 함께 NDK 경로를 제공해야 합니다. 또한 `-DANDROID_ABI` 플래그를 사용하여 타겟 ABI를 설정해야 합니다.

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=<NDK path>/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a ../tensorflow_src/tensorflow/lite
```

##### 커널(단위) 테스트 교차 컴파일의 세부 사항

단위 테스트의 교차 컴파일에는 호스트 아키텍처용 flatc 컴파일러가 필요합니다. 이를 위해 `tensorflow/lite/tools/cmake/native_tools/flatbuffers`에 CMakeLists가 있어 호스트 툴체인을 사용하여 별도의 빌드 디렉터리에 미리 CMake로 flatc 컴파일러를 빌드합니다.

```sh
mkdir flatc-native-build && cd flatc-native-build
cmake ../tensorflow_src/tensorflow/lite/tools/cmake/native_tools/flatbuffers
cmake --build .
```

사용자 지정 설치 위치(예: CMake 빌드 디렉터리 대신 기본적으로 빌드된 다른 도구가 포함된 디렉터리)에 <em>flatc</em>를 <strong>설치할</strong> 수도 있습니다.

```sh
cmake -DCMAKE_INSTALL_PREFIX=<native_tools_dir> ../tensorflow_src/tensorflow/lite/tools/cmake/native_tools/flatbuffers
cmake --build .
```

TF Lite 교차 컴파일 자체의 경우, 기본 <em>flatc</em> 바이너리가 포함된 디렉터리를 가리키는 추가 매개변수 <code>-DTFLITE_HOST_TOOLS_DIR=&lt;flatc_dir_path&gt;</code>를 위에서 언급한 `-DTFLITE_KERNEL_TEST=on` 플래그와 함께 제공해야 합니다.

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=${OE_CMAKE_TOOLCHAIN_FILE} -DTFLITE_KERNEL_TEST=on -DTFLITE_HOST_TOOLS_DIR=<flatc_dir_path> ../tensorflow/lite/
```

##### 타겟에서 교차 컴파일된 커널(단위) 테스트 시작

단위 테스트는 별도의 실행 파일로 실행하거나 CTest 유틸리티를 사용하여 실행할 수 있습니다. CTest에 관한 한 `TFLITE_ENABLE_NNAPI, TFLITE_ENABLE_XNNPACK` 또는 `TFLITE_EXTERNAL_DELEGATE` 매개변수 중 하나 이상이 TF Lite 빌드에 대해 활성화된 경우, 결과적인 테스트는 두 개의 다른 **레이블**로 생성됩니다(동일한 테스트 실행 파일 사용): - *일반* - CPU 백엔드에서 실행되는 테스트를 나타냅니다. - *대리자* - 사용된 대리자 사양에 사용되는 추가 실행 인수를 예상하는 테스트를 나타냅니다.

`CTestTestfile.cmake` 및 `run-tests.cmake`(아래 참조)는 모두 `<build_dir>/kernels`에서 사용할 수 있습니다.

CPU 백엔드로 단위 테스트 시작(`CTestTestfile.cmake`가 현재 디렉터리의 타겟에 있는 경우):

```sh
ctest -L plain
```

대리자를 사용하여 단위 테스트의 예 시작(`CTestTestfile.cmake` 및 `run-tests.cmake` 파일이 현재 디렉터리의 타겟에 있는 경우):

```sh
cmake -E env TESTS_ARGUMENTS=--use_nnapi=true\;--nnapi_accelerator_name=vsi-npu ctest -L delegate
cmake -E env TESTS_ARGUMENTS=--use_xnnpack=true ctest -L delegate
cmake -E env TESTS_ARGUMENTS=--external_delegate_path=<PATH> ctest -L delegate
```

단위 테스트에 추가 대리자 관련 시작 인수를 제공하는 이 방법의 **알려진 제한 사항**은 **예상 반환 값이 0**인 항목만 효과적으로 지원한다는 것입니다. 다른 반환 값은 테스트 실패로 보고됩니다.

#### OpenCL GPU 대리자

타겟 시스템에 OpenCL 지원이 있는 경우, GPU 성능을 활용할 수 있는 [GPU 대리자](https://www.tensorflow.org/lite/performance/gpu)를 사용할 수 있습니다.

OpenCL GPU 대리자 지원을 구성하려면:

```sh
cmake ../tensorflow_src/tensorflow/lite -DTFLITE_ENABLE_GPU=ON
```

**참고:** 이것은 실험적 단계이며 TensorFlow 2.5부터 사용할 수 있습니다. 호환성 문제가 있을 수 있습니다. Android 기기와 NVidia CUDA OpenCL 1.2에서만 확인되었습니다.

### 5단계. TensorFlow Lite 빌드하기

tflite_build 디렉터리에서,

```sh
cmake --build . -j
```

**참고:** 이렇게 하면 현재 디렉터리에 정적 라이브러리 `libtensorflow-lite.a`가 생성되지만 모든 전이 종속성이 포함되지 않기 때문에 라이브러리는 구성이 완전하지 않습니다. 라이브러리를 제대로 사용하려면 CMake 프로젝트를 생성해야 합니다. ["TensorFlow Lite를 사용하는 CMake 프로젝트 만들기"](#create_a_cmake_project_which_uses_tensorflow_lite) 섹션을 참조하세요.

### 6단계. TensorFlow Lite 벤치마크 도구 및 레이블 이미지 예제 빌드하기(선택 사항)

tflite_build 디렉터리에서,

```sh
cmake --build . -j -t benchmark_model
```

```sh
cmake --build . -j -t label_image
```

## TensorFlow Lite를 빌드하는 데 사용 가능한 옵션

다음은 사용 가능한 옵션 목록입니다. `-D<option_name>=[ON|OFF]`로 재정의 가능합니다. 예를 들어 `-DTFLITE_ENABLE_XNNPACK=OFF`는 기본적으로 활성화된 XNNPACK을 비활성화합니다.

옵션 이름 | 기능 | Android | Linux | macOS | Windows
--- | --- | --- | --- | --- | ---
TFLITE_ENABLE_RUY | RUY 활성화 | ON | OFF | OFF | OFF
:                       : matrix         :         :       :       :         : |  |  |  |  |
:                       : multiplication :         :       :       :         : |  |  |  |  |
:                       : library        :         :       :       :         : |  |  |  |  |
TFLITE_ENABLE_NNAPI | NNAPI 활성화 | ON | OFF | N/A | N/A
:                       : delegate       :         :       :       :         : |  |  |  |  |
TFLITE_ENABLE_GPU | GPU 활성화 | OFF | OFF | N/A | N/A
:                       : delegate       :         :       :       :         : |  |  |  |  |
TFLITE_ENABLE_XNNPACK | XNNPACK 활성화 | ON | ON | ON | ON
:                       : delegate       :         :       :       :         : |  |  |  |  |
TFLITE_ENABLE_MMAP | MMAP 활성화 | ON | ON | ON | N/A

## TensorFlow Lite를 사용하는 CMake 프로젝트 만들기

다음은 [TFLite 최소 예제](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/minimal)의 CMakeLists.txt입니다.

TensorFlow Lite 디렉터리에 대한 add_subdirectory()가 있어야 하고 `tensorflow-lite`를 target_link_libraries()와 연결해야 합니다.

```
cmake_minimum_required(VERSION 3.16)
project(minimal C CXX)

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project" )
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

add_executable(minimal minimal.cc)
target_link_libraries(minimal tensorflow-lite)
```

## TensorFlow Lite C 라이브러리 빌드하기

[C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/README.md)용 TensorFlow Lite 공유 라이브러리를 빌드하려면 먼저 [1단계](#step-1-install-cmake-tool)부터 [3단계](#step-3-create-cmake-build-directory)까지를 따르세요. 그 후에 다음 명령을 실행하세요.

```sh
cmake ../tensorflow_src/tensorflow/lite/c
cmake --build . -j
```

이 명령은 현재 디렉터리에 다음 공유 라이브러리를 생성합니다.

Platform | Library name
--- | ---
Linux | libtensorflowlite_c.so
macOS | libtensorflowlite_c.dylib
Windows | tensorflowlite_c.dll

**참고:** 생성된 공유 라이브러리를 사용하려면 필요한 헤더(c_api.h, c_api_experimental.h 및 common.h)가 있어야 합니다.
