# C++ 라이브러리 이해하기

TensorFlow Lite for Microcontrollers C++ 라이브러리는 [TensorFlow 리포지토리](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro)의 일부이며, 읽기 쉽고 수정하기 쉬운 외에도 잘 테스트되고 쉽게 통합되며 일반 TensorFlow Lite와 호환되도록 설계되었습니다.

다음 문서에서는 C++ 라이브러리의 기본 구조에 대한 요약과 고유한 프로젝트를 만드는 방법에 대한 정보를 제공합니다.

## 파일 구조

[`micro`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro) 루트 디렉토리는 비교적 간단한 구조를 가지고 있습니다. 그러나, 광범위한 TensorFlow 리포지토리 내에 있으므로 다양한 임베디드 개발 환경 내에서 독립적으로 관련 소스 파일을 제공하는 스크립트와 사전 생성된 프로젝트 파일을 마련했습니다.

### 주요 파일

마이크로컨트롤러용 TensorFlow Lite 인터프리터를 사용하는 데 가장 중요한 파일들은 프로젝트의 루트에 있으며 테스트가 함께 제공됩니다.

- [`all_ops_resolver.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/all_ops_resolver.h) 또는 [`micro_mutable_op_resolver.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/micro_mutable_op_resolver.h)를 사용하여 인터프리터가 모델을 실행하는 데 사용하는 연산을 제공할 수 있습니다. `all_ops_resolver.h`는 사용 가능한 모든 연산을 가져오기 때문에 많은 메모리를 사용합니다. 운영 애플리케이션에서는 `micro_mutable_op_resolver.h`를 사용하여 모델에 필요한 연산만 가져와야 합니다.
- [`micro_error_reporter.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/micro_error_reporter.h)는 디버그 정보를 출력합니다.
- [`micro_interpreter.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/micro_interpreter.h)에는 모델을 처리하고 실행하는 코드가 포함되어 있습니다.

일반적인 사용법에 대한 안내는 [마이크로컨트롤러 시작하기](get_started.md)를 참조하세요.

빌드 시스템은 특정 파일의 플랫폼별 구현을 제공합니다. 이들 구현은 플랫폼 이름을 가진 디렉토리(예: [`sparkfun_edge`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/sparkfun_edge))에 들어 있습니다.

다음을 포함한 다른 여러 디렉토리가 있습니다.

- [`kernel`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/kernels) - 연산 구현 및 관련 코드를 포함합니다.
- [`tools`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/tools) - 빌드 도구 및 해당 출력이 포함됩니다.
- [`examples`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples) - 샘플 코드를 포함합니다.

## 새 프로젝트 시작하기

*Hello World* 예제를 새 프로젝트의 템플릿으로 사용하는 것이 좋습니다. 이 섹션의 지침에 따라 선택한 플랫폼에 맞는 버전을 얻을 수 있습니다.

### Arduino 라이브러리 사용하기

Arduino를 사용하는 경우, *Hello World* 예제는 `Arduino_TensorFlowLite` Arduino 라이브러리에 포함되어 있고, 이 라이브러리는 Arduino IDE 및 [Arduino Create](https://create.arduino.cc/)에서 다운로드할 수 있습니다.

라이브러리가 추가되면 `File -> Examples`로 이동합니다. 목록 하단 근처에 `TensorFlowLite:hello_world`라는 예제가 표시됩니다. 이 예제를 선택하고 `hello_world`를 클릭하여 예제를 로드합니다. 그런 다음 예제의 사본을 저장하여 고유한 프로젝트의 기초로 이용할 수 있습니다.

### 다른 플랫폼용 프로젝트 생성하기

마이크로컨트롤러용 TensorFlow Lite는 `Makefile`을 사용하여 필요한 모든 소스 파일을 포함하는 독립형 프로젝트를 생성할 수 있습니다. 현재 지원되는 환경은 Keil, Make 및 Mbed입니다.

Make로 이러한 프로젝트를 생성하려면 [TensorFlow 리포지토리](http://github.com/tensorflow/tensorflow)를 복제하고 다음 명령을 실행합니다.

```bash
make -f tensorflow/lite/micro/tools/make/Makefile generate_projects
```

종속성에 대한 대용량 도구 체인을 다운로드해야 하므로 몇 분 정도 걸립니다. 완료되면 `tensorflow/lite/micro/tools/make/gen/linux_x86_64/prj/`(정확한 경로는 호스트 운영 체제에 따라 다름)와 같은 경로 내에 생성된 일부 폴더가 나타납니다. 이들 폴더에는 생성된 프로젝트와 소스 파일이 들어 있습니다.

명령을 실행한 후, `tensorflow/lite/micro/tools/make/gen/linux_x86_64/prj/hello_world`에서 *Hello World* 프로젝트를 찾을 수 있습니다. 예를 들어, `hello_world/keil`에는 Keil 프로젝트가 포함됩니다.

## 테스트 실행하기

라이브러리를 빌드하고 모든 단위 테스트를 실행하려면 다음 명령을 사용합니다.

```bash
make -f tensorflow/lite/micro/tools/make/Makefile test
```

개별 테스트를 실행하려면 다음 명령을 사용하여 `<test_name>`을 테스트 이름으로 바꿉니다.

```bash
make -f tensorflow/lite/micro/tools/make/Makefile test_<test_name>
```

프로젝트의 Makefile에서 테스트 이름을 찾을 수 있습니다. 예를 들어, `examples/hello_world/Makefile.inc`는 *Hello World* 예제의 테스트 이름을 지정합니다.

## 바이너리 빌드하기

주어진 프로젝트(예: 예제 애플리케이션)의 실행 가능한 바이너리를 빌드하려면 다음 명령을 사용하여 `<project_name>`을 빌드하려는 프로젝트로 바꿉니다.

```bash
make -f tensorflow/lite/micro/tools/make/Makefile <project_name>_bin
```

예를 들어, 다음 명령은 *Hello World* 애플리케이션용 바이너리를 빌드합니다.

```bash
make -f tensorflow/lite/micro/tools/make/Makefile hello_world_bin
```

기본적으로, 프로젝트는 호스트 운영 체제에 맞게 컴파일됩니다. 다른 대상 아키텍처를 지정하려면 `TARGET=`을 사용하세요. 다음 예는 SparkFun Edge에 적합하게 *Hello World* 예제를 빌드하는 방법을 보여줍니다.

```bash
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=sparkfun_edge hello_world_bin
```

대상이 지정되면 사용 가능한 대상별 소스 파일이 원본 코드 대신 사용됩니다. 예를 들어, `examples/hello_world/sparkfun_edge` 하위 디렉토리에는 대상 `sparkfun_edge`가 지정될 때 사용되는 파일인 `constants.cc` 및 `output_handler.cc`의 SparkFun Edge 구현이 포함됩니다.

프로젝트의 Makefile에서 프로젝트 이름을 찾을 수 있습니다. 예를 들어, `examples/hello_world/Makefile.inc`는 *Hello World* 예제의 바이너리 이름을 지정합니다.

## 최적화된 커널

`tensorflow/lite/micro/kernels`의 루트에 있는 참조 커널은 순수 C/C++로 구현되며 플랫폼별 하드웨어 최적화를 포함하지 않습니다.

최적화된 커널 버전은 하위 디렉토리에 제공됩니다. 예를 들어, `kernels/cmsis-nn`에는 Arm의 CMSIS-NN 라이브러리를 사용하는 여러 최적화된 커널이 포함되어 있습니다.

최적화된 커널을 사용하여 프로젝트를 생성하려면 다음 명령을 사용하여 `<subdirectory_name>`을 최적화가 포함된 하위 디렉토리의 이름으로 바꿉니다.

```bash
make -f tensorflow/lite/micro/tools/make/Makefile TAGS=<subdirectory_name> generate_projects
```

새 하위 폴더를 만들어 고유한 최적화를 추가할 수 있습니다. 새로 최적화된 구현에 대해 pull 요청을 권장합니다.

## Arduino 라이브러리 생성하기

Arduino 라이브러리의 야간 빌드는 Arduino IDE의 라이브러리 관리자를 통해 사용할 수 있습니다.

라이브러리의 새 빌드를 생성해야 하는 경우, TensorFlow 리포지토리에서 다음 스크립트를 실행할 수 있습니다.

```bash
./tensorflow/lite/micro/tools/ci_build/test_arduino.sh
```

결과 라이브러리는 `tensorflow/lite/micro/tools/make/gen/arduino_x86_64/prj/tensorflow_lite.zip`에서 찾을 수 있습니다.

## 새 기기로 이식하기

마이크로컨트롤러용 TensorFlow Lite를 새로운 플랫폼 및 기기로 이식하는 방법에 대한 지침은 [`micro/README.md`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/README.md)에서 찾을 수 있습니다.
