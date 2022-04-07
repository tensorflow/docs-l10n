# Python 빠른 시작

Python에서 TensorFlow Lite를 사용하면 [Raspberry Pi](https://www.raspberrypi.org/){:.external} 및 [Edge TPU를 탑재한 Coral 기기](https://coral.withgoogle.com/){:.external}와 같이 Linux 기반의 임베디드 기기에서 유익한 결과를 거둘 수 있습니다.

이 페이지에서는 단 몇 분 안에 Python으로 TensorFlow Lite 모델 실행을 시작할 수 있는 방법을 보여줍니다. [TensorFlow Lite로 변환된](../convert/) TensorFlow 모델만 있으면 됩니다. 아직 변환된 모델이 없는 경우, 아래 링크된 예제와 함께 제공된 모델을 사용하여 시도해 볼 수 있습니다.

## TensorFlow Lite 런타임 패키지 정보

Python으로 TensorFlow Lite 모델 실행을 빠르게 시작하려면 모든 TensorFlow 패키지 대신 TensorFlow Lite 인터프리터만 설치할 수 있습니다. 이 단순화된 Python 패키지를 `tflite_runtime`이라고 합니다.

`tflite_runtime` 패키지의 크기는 전체 `tensorflow` 패키지의 극히 일부이며 TensorFlow Lite로 추론을 실행하는 데 필요한 최소한의 코드를 포함합니다. 여기에는 기본적으로 [`Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter) Python 클래스가 포함됩니다. 이 작은 패키지는 <code>.tflite</code> 모델만 실행하고 대용량 TensorFlow 라이브러리로 디스크 공간을 낭비하지 않으려는 경우에 이상적입니다.

참고: [TensorFlow Lite Converter](../convert/)와 같은 다른 Python API에 액세스해야 하는 경우 [전체 TensorFlow 패키지](https://www.tensorflow.org/install/)를 설치해야 합니다. 예를 들어 [Select TF ops](https://www.tensorflow.org/lite/guide/ops_select)는 `tflite_runtime` 패키지에 포함되어 있지 않습니다. 모델에 Select TF ops에 대한 종속성이 있는 경우, 대신 전체 TensorFlow 패키지를 사용해야 합니다.

## Python용 TensorFlow Lite 설치하기

You can install on Linux with pip:

<pre class="devsite-terminal devsite-click-to-copy">python3 -m pip install tflite-runtime
</pre>

## 지원되는 플랫폼

The `tflite-runtime` Python wheels are pre-built and provided for these platforms:

- Linux armv7l (e.g. Raspberry Pi 2, 3, 4 and Zero 2 running Raspberry Pi OS 32-bit)
- Linux aarch64 (e.g. Raspberry Pi 3, 4 running Debian ARM64)
- Linux x86_64

If you want to run TensorFlow Lite models on other platforms, you should either use the [full TensorFlow package](https://www.tensorflow.org/install/), or [build the tflite-runtime package from source](build_cmake_pip.md).

If you're using TensorFlow with the Coral Edge TPU, you should instead follow the appropriate [Coral setup documentation](https://coral.ai/docs/setup).

Note: We no longer update the Debian package `python3-tflite-runtime`. The latest Debian package is for TF version 2.5, which you can install by following [these older instructions](https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/lite/g3doc/guide/python.md#install-tensorflow-lite-for-python).

Note: We no longer release pre-built `tflite-runtime` wheels for Windows and macOS. For these platforms, you should use the [full TensorFlow package](https://www.tensorflow.org/install/), or [build the tflite-runtime package from source](build_cmake_pip.md).

## tflite_runtime을 사용하여 추론 실행하기

`tensorflow` 모듈에서 `Interpreter`를 가져오는 대신, 이제 `tflite_runtime`에서 가져와야 합니다.

예를 들어, 위의 패키지를 설치한 후 [`label_image.py`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/python/) 파일을 복사하고 실행합니다. (아마도) `tensorflow` 라이브러리가 설치되어 있지 않기 때문에 이 작업에 실패합니다. 이 문제를 해결하려면 파일의 다음 줄을 편집합니다.

```python
import tensorflow as tf
```

이제 코드는 다음과 같습니다.

```python
import tflite_runtime.interpreter as tflite
```

그리고 다음 줄을 변경합니다.

```python
interpreter = tf.lite.Interpreter(model_path=args.model_file)
```

이제 코드는 다음과 같습니다.

```python
interpreter = tflite.Interpreter(model_path=args.model_file)
```

이제 `label_image.py`를 다시 실행합니다. 모두 끝났습니다! 이제 TensorFlow Lite 모델을 실행합니다.

## 자세히 알아보기

- `Interpreter` API에 대한 자세한 내용은 [Python에서 모델 로드 및 실행하기](inference.md#load-and-run-a-model-in-python)를 참조하세요.

- If you have a Raspberry Pi, check out a [video series](https://www.youtube.com/watch?v=mNjXEybFn98&list=PLQY2H8rRoyvz_anznBg6y3VhuSMcpN9oe) about how to run object detection on Raspberry Pi using TensorFlow Lite.

- Coral ML 가속기를 사용하는 경우, [GitHub에서 Coral 예제](https://github.com/google-coral/tflite/tree/master/python/examples)를 확인하세요.

- To convert other TensorFlow models to TensorFlow Lite, read about the [TensorFlow Lite Converter](../convert/).

- `tflite_runtime` 휠을 빌드하려면 [TensorFlow Lite Python 휠 패키지 빌드](build_cmake_pip.md)를 읽으세요.
