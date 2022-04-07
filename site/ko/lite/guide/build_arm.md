# ARM 보드용 TensorFlow Lite 빌드하기

이 페이지에서는 ARM 기반 컴퓨터용 TensorFlow Lite 라이브러리를 빌드하는 방법을 설명합니다.

TensorFlow Lite는 두 개의 빌드 시스템을 지원하며 각 빌드 시스템에서 지원되는 기능은 동일하지 않습니다. 다음 표를 확인하여 적절한 빌드 시스템을 선택하세요.

기능 | Bazel | CMake
--- | --- | ---
사전 정의된 툴체인 | armhf, aarch64 | armel, armhf, aarch64
사용자 정의 툴체인 | 사용하기 더 어려움 | 사용하기 쉬움
[Select TF ops](https://www.tensorflow.org/lite/guide/ops_select) | 지원됨 | 지원되지 않음
[GPU delegate](https://www.tensorflow.org/lite/performance/gpu) | Android에서만 사용 가능 | OpenCL을 지원하는 모든 플랫폼
XNNPack | 지원됨 | 지원됨
[Python Wheel](https://www.tensorflow.org/lite/guide/build_cmake_pip) | 지원됨 | 지원됨
[C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/README.md) | 지원됨 | [supported](https://www.tensorflow.org/lite/guide/build_cmake#build_tensorflow_lite_c_library)
C++ API | Bazel 프로젝트에 지원됨 | CMake 프로젝트에 지원됨

## CMake를 사용한 ARM의 크로스 컴파일

CMake 프로젝트가 있거나 사용자 정의 툴체인을 사용하려는 경우, 크로스 컴파일에 CMake를 사용하는 것이 좋습니다. 이를 위해 별도의 [CMake를 이용한 크로스 컴파일 TensorFlow Lite](https://www.tensorflow.org/lite/guide/build_cmake_arm) 페이지를 이용할 수 있습니다.

## Bazel을 사용한 ARM 크로스 컴파일

Bazel 프로젝트가 있거나 TF op를 사용하려는 경우, Bazel 빌드 시스템을 사용하는 것이 좋습니다. Bazel과 통합된 [ARM GCC 8.3 툴체인](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/toolchains/embedded/arm-linux)을 사용하여 ARM32/64 공유 라이브러리를 빌드합니다.

대상 아키텍처 | Bazel 구성 | 호환 기기
--- | --- | ---
armhf (ARM32) | --config=elinux_armhf | RPI3, 32bit RPI4
:                     :                         : Raspberry Pi OS            : |  |
AArch64 (ARM64) | --config=elinux_aarch64 | Coral, Ubuntu 64가 설치된 RPI4
:                     :                         : bit                        : |  |

참고: 생성된 공유 라이브러리를 실행하려면 glibc 2.28 이상이 필요합니다.

다음 지침은 Ubuntu 16.04.3 64bit PC(AMD64) 및 TensorFlow devel docker 이미지 [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/)에서 테스트되었습니다.

TensorFlow Lite를 Bazel과 교차 컴파일하려면 다음 단계를 따릅니다.

#### 1단계. Bazel 설치하기

Bazel은 TensorFlow의 기본 빌드 시스템입니다. 최신 버전의 [Bazel 빌드 시스템](https://bazel.build/versions/master/docs/install.html)을 설치합니다.

**참고**: TensorFlow Docker 이미지를 사용하는 경우 Bazel을 이미 사용할 수 있습니다.

#### 2단계. TensorFlow 리포지토리를 복제합니다.

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**참고**: TensorFlow Docker 이미지를 사용하는 경우, 리포지토리는 `/tensorflow_src/`에 이미 제공되어 있습니다.

#### 3단계. ARM 바이너리 빌드하기

##### C 라이브러리

```bash
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/c:libtensorflowlite_c.so
```

공유 라이브러리는 `bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so`에서 찾을 수 있습니다.

**참고:** [32bit ARM 하드 플로트](https://wiki.debian.org/ArmHardFloatPort) 빌드에 <code>elinux_armhf</code>를 사용하세요.

자세한 내용은 [TensorFlow Lite C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/README.md) 페이지를 확인하세요.

##### C++ 라이브러리

```bash
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite:libtensorflowlite.so
```

공유 라이브러리는 `bazel-bin/tensorflow/lite/libtensorflowlite.so`에서 찾을 수 있습니다.

현재, 필요한 모든 헤더 파일을 추출하는 간단한 방법은 없으므로 모든 헤더 파일을 TensorFlow 리포지토리의 tensorflow/lite/에 포함해야 합니다. 또한 FlatBuffers 및 Abseil의 헤더 파일도 필요합니다.

##### 기타

툴체인을 사용하여 다른 Bazel 대상을 빌드할 수도 있습니다. 다음은 몇 가지 유용한 대상입니다.

- //tensorflow/lite/tools/benchmark:benchmark_model
- //tensorflow/lite/examples/label_image:label_image
