# ARM64 보드용 TensorFlow Lite 빌드하기

이 페이지에서는 ARM64 기반 컴퓨터용 TensorFlow Lite 정적 및 공유 라이브러리를 빌드하는 방법에 대해 설명합니다. TensorFlow Lite를 사용하여 모델 실행을 시작하려는 경우, 가장 빠른 방법은 [Python 빠른 시작](python.md)에 나와 있는 대로 TensorFlow Lite 런타임 패키지를 설치하는 것입니다.

참고: 이 페이지에서는 TensorFlow Lite용 C++ 정적 및 공유 라이브러리를 컴파일하는 방법만 보여줍니다. 대체 설치 방법은 다음과 같습니다. [Python 인터프리터 API만 설치합니다](python.md)(추론 전용). [pip에서 전체 TensorFlow 패키지를 설치합니다](https://www.tensorflow.org/install/pip). 또는 [전체 TensorFlow 패키지를 빌드합니다](https://www.tensorflow.org/install/source).

## Make를 사용한 ARM64의 교차 컴파일

적절한 빌드 환경을 보장하려면 [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/)과 같은 TensorFlow Docker 이미지 중 하나를 사용하는 것이 좋습니다.

시작하려면 도구 체인과 libs를 설치합니다.

```bash
sudo apt-get update
sudo apt-get install crossbuild-essential-arm64
```

Docker를 사용하는 경우 `sudo`를 사용할 수 없습니다.

이제 TensorFlow 리포지토리(`https://github.com/tensorflow/tensorflow`)를 복제(git-clone)합니다. TensorFlow Docker 이미지를 사용하는 경우, 리포지토리는 이미 `/tensorflow_src/`에 제공되어 있습니다. 그런 다음, TensorFlow 리포지토리의 루트에서 다음 스크립트를 실행하여 모든 빌드 종속성을 다운로드합니다.

```bash
./tensorflow/lite/tools/make/download_dependencies.sh
```

이 작업은 한 번만 수행하면 됩니다.

그런 다음 컴파일합니다.

```bash
./tensorflow/lite/tools/make/build_aarch64_lib.sh
```

이 코드는 `tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a`에서 정적 라이브러리를 컴파일합니다.

## ARM64에서 고유하게 컴파일하기

이들 단계는 HardKernel Odroid C2, gcc 버전 5.4.0에서 테스트되었습니다.

보드에 로그인하고 도구 체인을 설치합니다.

```bash
sudo apt-get install build-essential
```

이제 TensorFlow 리포지토리(`https://github.com/tensorflow/tensorflow`)를 복제(git-clone)하고 리포지토리의 루트에서 다음을 실행합니다.

```bash
./tensorflow/lite/tools/make/download_dependencies.sh
```

이 작업은 한 번만 수행하면 됩니다.

그런 다음 컴파일합니다.

```bash
./tensorflow/lite/tools/make/build_aarch64_lib.sh
```

이 코드는 `tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a`에서 정적 라이브러리를 컴파일합니다.

## Bazel을 사용한 ARM64의 교차 컴파일

Bazel과 함께 [ARM GCC 도구 체인](https://github.com/tensorflow/tensorflow/tree/master/third_party/toolchains/embedded/arm-linux)을 사용하여 ARM64 공유 라이브러리를 빌드할 수 있습니다.

참고: 생성된 공유 라이브러리를 실행하려면 glibc 2.28 이상이 필요합니다.

다음 지침은 Ubuntu 16.04.3 64bit PC(AMD64) 및 TensorFlow devel docker 이미지 [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/)에서 테스트되었습니다.

TensorFlow Lite를 Bazel과 교차 컴파일하려면 다음 단계를 따릅니다.

#### 1단계. Bazel 설치하기

Bazel은 TensorFlow의 기본 빌드 시스템입니다. 최신 버전의 [Bazel 빌드 시스템](https://bazel.build/versions/master/docs/install.html)을 설치합니다.

**참고:** TensorFlow Docker 이미지를 사용하는 경우 Bazel을 이미 사용할 수 있습니다.

#### 2단계. TensorFlow 리포지토리 복제하기

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**참고:** TensorFlow Docker 이미지를 사용하는 경우, 리포지토리는 `/tensorflow_src/`에 이미 제공되어 있습니다.

#### 3단계. ARM64 바이너리 빌드하기

##### C 라이브러리

```bash
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/c:libtensorflowlite_c.so
```

자세한 내용은 [TensorFlow Lite C API](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/c) 페이지를 확인하세요.

##### C++ 라이브러리

```bash
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite:libtensorflowlite.so
```

공유 라이브러리는 `bazel-bin/tensorflow/lite/libtensorflowlite.so`에서 찾을 수 있습니다.

현재, 필요한 모든 헤더 파일을 추출하는 간단한 방법은 없으므로 모든 헤더 파일을 TensorFlow 리포지토리의 tensorflow/lite/에 포함해야 합니다. 또한 FlatBuffers 및 Abseil의 헤더 파일도 필요합니다.

##### 기타

도구 체인을 사용하여 다른 Bazel 대상을 빌드할 수도 있습니다. 다음은 몇 가지 유용한 대상입니다.

- //tensorflow/lite/tools/benchmark:benchmark_model
- //tensorflow/lite/examples/label_image:label_image
