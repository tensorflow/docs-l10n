# Raspberry Pi용 TensorFlow Lite 빌드하기

이 페이지에서는 Raspberry Pi용 TensorFlow Lite 정적 및 공유 라이브러리를 빌드하는 방법에 대해 설명합니다. TensorFlow Lite를 사용하여 모델 실행을 시작하려는 경우, 가장 빠른 방법은 [Python 빠른 시작](python.md)에 나와 있는 대로 TensorFlow Lite 런타임 패키지를 설치하는 것입니다.

**참고:** 이 페이지에서는 TensorFlow Lite용 C++ 정적 및 공유 라이브러리를 컴파일하는 방법을 보여줍니다. 대체 설치 방법은 다음과 같습니다. [Python 인터프리터 API만 설치합니다](python.md)(추론 전용). [pip에서 전체 TensorFlow 패키지를 설치합니다](https://www.tensorflow.org/install/pip). 또는 [전체 TensorFlow 패키지를 빌드합니다](https://www.tensorflow.org/install/source_rpi).

**참고:** 여기서는 32bit 빌드만 다룹니다. 64bit 빌드를 찾고 있다면 [ARM64용 빌드](build_arm64.md) 페이지를 확인하세요.

## Make를 사용한 Raspberry Pi의 교차 컴파일

다음 지침은 Ubuntu 16.04.3 64bit PC(AMD64) 및 TensorFlow devel docker 이미지 [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/)에서 테스트되었습니다.

TensorFlow Lite를 교차 컴파일하려면 다음 단계를 따릅니다.

#### 1단계. 공식 Raspberry Pi 교차 컴파일 도구 체인을 복제합니다.

```sh
git clone https://github.com/raspberrypi/tools.git rpi_tools
```

#### 2단계. TensorFlow 리포지토리를 복제합니다.

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**참고:** TensorFlow Docker 이미지를 사용하는 경우, 리포지토리는 `/tensorflow_src/`에 이미 제공되어 있습니다.

#### 3단계. TensorFlow 리포지토리의 루트에서 다음 스크립트를 실행하여 모든 빌드 종속성을

다운로드합니다.

```sh
cd tensorflow_src && ./tensorflow/lite/tools/make/download_dependencies.sh
```

**참고:** 이 작업은 한 번만 수행하면 됩니다.

#### 4a단계. Raspberry Pi 2, 3 및 4용 ARMv7 바이너리를 빌드합니다.

```sh
PATH=../rpi_tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin:$PATH \
  ./tensorflow/lite/tools/make/build_rpi_lib.sh
```

**참고:** `tensorflow/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a`에서 정적 라이브러리를 컴파일해야 합니다.

`build_rpi_lib.sh` 스크립트는 TFLite [Makefile](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/make/Makefile)을 사용하는 Make의 래퍼이므로 추가 Make 옵션 또는 대상 이름을 추가할 수 있습니다. 가능한 옵션은 다음과 같습니다.

```sh
./tensorflow/lite/tools/make/build_rpi_lib.sh clean # clean object files
./tensorflow/lite/tools/make/build_rpi_lib.sh -j 16 # run with 16 jobs to leverage more CPU cores
./tensorflow/lite/tools/make/build_rpi_lib.sh label_image # # build label_image binary
```

#### 4b단계. Raspberry Pi Zero용 ARMv6 바이너리를 빌드합니다.

```sh
PATH=../rpi_tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin:$PATH \
  ./tensorflow/lite/tools/make/build_rpi_lib.sh TARGET_ARCH=armv6
```

**참고:** `tensorflow/lite/tools/make/gen/rpi_armv6/lib/libtensorflow-lite.a`에서 정적 라이브러리를 컴파일해야 합니다.

## Raspberry Pi에서 자체적으로 컴파일하기

다음 지침은 Raspberry Pi Zero, Raspbian GNU/Linux 10(buster), gcc 버전 8.3.0(Raspbian 8.3.0-6+rpi1)에서 테스트되었습니다.

TensorFlow Lite를 네이티브로 컴파일하려면 다음 단계를 따릅니다.

#### 1단계. Raspberry Pi에 로그인하고 도구 체인을 설치합니다.

```sh
sudo apt-get install build-essential
```

#### 2단계. TensorFlow 리포지토리를 복제합니다.

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

#### 3단계. TensorFlow 리포지토리의 루트에서 다음 스크립트를 실행하여 모든 빌드 종속성을 다운로드합니다.

```sh
cd tensorflow_src && ./tensorflow/lite/tools/make/download_dependencies.sh
```

**참고:** 이 작업은 한 번만 수행하면 됩니다.

#### 4단계. 그러면 다음을 사용하여 TensorFlow Lite를 컴파일할 수 있습니다.

```sh
./tensorflow/lite/tools/make/build_rpi_lib.sh
```

**참고: **`tensorflow/lite/tools/make/gen/lib/rpi_armv6/libtensorflow-lite.a`에서 정적 라이브러리를 컴파일해야 합니다.

## Bazel을 사용한 armhf의 교차 컴파일

Bazel과 함께 [ARM GCC 도구 체인](https://github.com/tensorflow/tensorflow/tree/master/third_party/toolchains/embedded/arm-linux)을 사용하여 Raspberry Pi 2, 3 및 4와 호환되는 armhf 공유 라이브러리를 빌드할 수 있습니다.

참고: 생성된 공유 라이브러리를 실행하려면 glibc 2.28 이상이 필요합니다.

다음 지침은 Ubuntu 16.04.3 64bit PC(AMD64) 및 TensorFlow devel docker 이미지 [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/)에서 테스트되었습니다.

TensorFlow Lite를 Bazel과 교차 컴파일하려면 다음 단계를 따릅니다.

#### 1단계. Bazel 설치하기

Bazel은 TensorFlow의 기본 빌드 시스템입니다. 최신 버전의 [Bazel 빌드 시스템](https://bazel.build/versions/master/docs/install.html)을 설치합니다.

**참고:** TensorFlow Docker 이미지를 사용하는 경우 Bazel을 이미 사용할 수 있습니다.

#### 2단계. TensorFlow 리포지토리를 복제합니다.

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**참고:** TensorFlow Docker 이미지를 사용하는 경우, 리포지토리는 `/tensorflow_src/`에 이미 제공되어 있습니다.

#### 3단계. Raspberry Pi 2, 3 및 4용 ARMv7 바이너리를 빌드합니다.

##### C 라이브러리

```bash
bazel build --config=elinux_armhf -c opt //tensorflow/lite/c:libtensorflowlite_c.so
```

자세한 내용은 [TensorFlow Lite C API](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/c) 페이지를 확인하세요.

##### C++ 라이브러리

```bash
bazel build --config=elinux_armhf -c opt //tensorflow/lite:libtensorflowlite.so
```

공유 라이브러리는 `bazel-bin/tensorflow/lite/libtensorflowlite.so`에서 찾을 수 있습니다.

현재, 필요한 모든 헤더 파일을 추출하는 간단한 방법은 없으므로 모든 헤더 파일을 TensorFlow 리포지토리의 <code>tensorflow/lite/</code>에 포함해야 합니다. 또한 <a>FlatBuffers</a> 및 <a>Abseil</a>의 헤더 파일도 필요합니다.

##### 기타

도구 체인을 사용하여 다른 Bazel 대상을 빌드할 수도 있습니다. 다음은 몇 가지 유용한 대상입니다.

- //tensorflow/lite/tools/benchmark:benchmark_model
- //tensorflow/lite/examples/label_image:label_image
