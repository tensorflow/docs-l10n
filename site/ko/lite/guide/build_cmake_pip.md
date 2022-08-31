# TensorFlow Lite Python 휠 패키지 빌드

이 페이지에서는 x86_64 및 다양한 ARM 장치용 TensorFlow Lite `tflite_runtime` Python 라이브러리를 빌드하는 방법을 설명합니다.

다음 지침은 Ubuntu 16.04.3 64비트 PC(AMD64), macOS Catalina(x86_64) 및 TensorFlow devel Docker 이미지 [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/)에서 테스트되었습니다.

**참고:** 이 기능은 버전 2.4부터 사용할 수 있습니다.

#### 전제 조건

CMake가 설치되어 있고 TensorFlow 소스 코드 사본이 필요합니다. 자세한 내용은 [CMake로 TensorFlow Lite 빌드하기](https://www.tensorflow.org/lite/guide/build_cmake) 페이지를 확인하세요.

워크스테이션용 PIP 패키지를 빌드하려면 다음 명령을 실행할 수 있습니다.

```sh
PYTHON=python3 tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh native
```

**참고:** 여러 Python 인터프리터를 사용할 수 있는 경우 `PYTHON` 변수를 사용하여 정확한 Python 버전을 지정하세요. (현재 Python 3.7 이상 지원)

## ARM 교차 컴파일

ARM 교차 컴파일의 경우, 교차 빌드 환경을 설정하기 쉽기 때문에 Docker를 사용하는 것이 좋습니다. 또한 대상 아키텍처를 파악하려면 `target` 옵션이 필요합니다.

Makefile `tensorflow/lite/tools/pip_package/Makefile`에 미리 정의된 Docker 컨테이너를 사용하여 빌드 명령을 호출하는 데 사용할 수 있는 도우미 도구가 있습니다. Docker 호스트 머신에서 다음과 같이 빌드 명령을 실행할 수 있습니다.

```sh
make -C tensorflow/lite/tools/pip_package docker-build \
  TENSORFLOW_TARGET=<target> PYTHON_VERSION=<python3 version>
```

**참고:** Python 버전 3.7 이상이 지원됩니다.

### 사용 가능한 대상 이름

`tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh` 스크립트에는 대상 아키텍처를 파악하기 위해 대상 이름이 필요합니다. 다음은 지원 대상 목록입니다.

대상 | 대상 아키텍처 | 주석
--- | --- | ---
armhf | Neon이 포함된 ARMv7 VFP | Raspberry Pi 3 및 4와 호환
rpi0 | ARMv6 | Raspberry Pi Zero와 호환
aarch64 | aarch64(ARM 64비트) | [산호 멘델 리눅스 4.0](https://coral.ai/)<br> [Ubuntu Server 20.04.01 LTS 64비트](https://ubuntu.com/download/raspberry-pi)가 포함된 Raspberry Pi
네이티브 | 자신의 워크스테이션 | "-mnative" 최적화로 빌드됨
<default></default> | 자신의 워크스테이션 | 기본 대상

### 빌드 예제

다음은 사용할 수 있는 몇 가지 예제 명령입니다.

#### Python 3.7용 armhf 대상

```sh
make -C tensorflow/lite/tools/pip_package docker-build \
  TENSORFLOW_TARGET=armhf PYTHON_VERSION=3.7
```

#### Python 3.8용 aarch64 대상

```sh
make -C tensorflow/lite/tools/pip_package docker-build \
  TENSORFLOW_TARGET=aarch64 PYTHON_VERSION=3.8
```

#### 사용자 지정 도구 모음을 사용하는 방법은?

생성된 바이너리가 대상과 호환되지 않는 경우, 자체 툴체인을 사용하거나 사용자 지정 빌드 플래그를 제공해야 합니다. (대상 환경을 이해하려면 [이 내용](https://www.tensorflow.org/lite/guide/build_cmake_arm#check_your_target_environment)을 확인하세요.) 이 경우 `tensorflow/lite/tools/cmake/download_toolchains.sh`를 수정하여 고유한 툴체인을 사용해야 합니다. 툴체인 스크립트는 `build_pip_package_with_cmake.sh` 스크립트에 대해 다음 두 변수를 정의합니다.

변수 | 목적 | 예시
--- | --- | ---
ARMCC_PREFIX | 툴체인 접두사 정의 | arm-linux-gnueabihf-
ARMCC_FLAGS | 컴파일 플래그 | -march=armv7-a -mfpu=neon-vfpv4

**참고:** ARMCC_FLAGS는 Python 라이브러리 포함 경로를 포함해야 할 수 있습니다. `download_toolchains.sh`에서 참조 내용을 확인하세요.
