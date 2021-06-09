# Python 빠른 시작

Python에서 TensorFlow Lite를 사용하면 [Raspberry Pi](https://www.raspberrypi.org/){:.external} 및 [Edge TPU를 탑재한 Coral 기기](https://coral.withgoogle.com/){:.external}와 같이 Linux 기반의 임베디드 기기에서 유익한 결과를 거둘 수 있습니다.

이 페이지에서는 단 몇 분 안에 Python으로 TensorFlow Lite 모델 실행을 시작할 수 있는 방법을 보여줍니다. [TensorFlow Lite로 변환된](../convert/) TensorFlow 모델만 있으면 됩니다. 아직 변환된 모델이 없는 경우, 아래 링크된 예제와 함께 제공된 모델을 사용하여 시도해 볼 수 있습니다.

## TensorFlow Lite 런타임 패키지 정보

Python으로 TensorFlow Lite 모델 실행을 빠르게 시작하려면 모든 TensorFlow 패키지 대신 TensorFlow Lite 인터프리터만 설치할 수 있습니다. 이 단순화된 Python 패키지를 `tflite_runtime`이라고 합니다.

`tflite_runtime` 패키지의 크기는 전체 `tensorflow` 패키지의 극히 일부이며 TensorFlow Lite로 추론을 실행하는 데 필요한 최소한의 코드를 포함합니다. 여기에는 기본적으로 [`Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter) Python 클래스가 포함됩니다. 이 작은 패키지는 <code>.tflite</code> 모델만 실행하고 대용량 TensorFlow 라이브러리로 디스크 공간을 낭비하지 않으려는 경우에 이상적입니다.

참고: [TensorFlow Lite Converter](../convert/python_api.md)와 같은 다른 Python API에 액세스해야 하는 경우, [전체 TensorFlow 패키지](https://www.tensorflow.org/install/)를 설치해야 합니다.

## Python용 TensorFlow Lite 설치하기

설치하려면 `pip3 install`을 실행하고 다음 표에서 적절한 Python wheel URL을 전달합니다.

<pre class="devsite-terminal">pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl</pre>

예를 들어, Raspbian Buster(Python 3.7 포함)를 실행하는 Raspberry Pi를 사용 중인 경우, 다음과 같이 Python wheel을 설치합니다.

<pre class="devsite-terminal devsite-click-to-copy">pip3 install --index-url https://google-coral.github.io/py-repo/ tflite_runtime
</pre>

따라서 `Interpreter`를 `tensorflow` 모듈 대신 `tflite_runtime`에서 가져와야 합니다.

참고: Debian Linux에서 pip를 사용하여 `tflite_runtime`을 설치하는 경우 Debian 패키지로 설치하고 TF Lite에 종속된 다른 소프트웨어(예: [Coral 라이브러리](https://coral.ai/software/))를 사용할 때 런타임 오류가 발생할 수 있습니다. pip로 `tflite_runtime`을 제거한 다음 위의 `apt-get` 명령으로 다시 설치하면 문제를 해결할 수 있습니다.

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

`Interpreter` API에 대한 자세한 내용은 [Python에서 모델 로드 및 실행하기](inference.md#load-and-run-a-model-in-python)를 참조하세요.

Raspberry Pi를 사용하는 경우, [classify_picamera.py 예제](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/raspberry_pi)를 사용하여 Pi 카메라 및 TensorFlow Lite로 이미지 분류를 수행해 보세요.

Coral ML 가속기를 사용하는 경우, [GitHub에서 Coral 예제](https://github.com/google-coral/tflite/tree/master/python/examples)를 확인하세요.

다른 TensorFlow 모델을 TensorFlow Lite로 변환하려면 [TensorFlow Lite 변환기](../convert/)에 대해 읽어보세요.

`tflite_runtime` 휠을 빌드하려면 [TensorFlow Lite Python 휠 패키지 빌드](build_cmake_pip.md)를 읽으세요.
