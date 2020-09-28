# Python 빠른 시작

Python에서 TensorFlow Lite를 사용하면 [Raspberry Pi](https://www.raspberrypi.org/){:.external} 및 [Edge TPU를 탑재한 Coral 기기](https://coral.withgoogle.com/){:.external}와 같이 Linux 기반의 임베디드 기기에서 유익한 결과를 거둘 수 있습니다.

이 페이지에서는 단 몇 분 안에 Python으로 TensorFlow Lite 모델 실행을 시작할 수 있는 방법을 보여줍니다. [TensorFlow Lite로 변환된](../convert/) TensorFlow 모델만 있으면 됩니다. 아직 변환된 모델이 없는 경우, 아래 링크된 예제와 함께 제공된 모델을 사용하여 시도해 볼 수 있습니다.

## TensorFlow Lite 인터프리터만 설치하기

Python으로 TensorFlow Lite 모델을 빠르게 실행하려면 전체 TensorFlow 패키지 대신 TensorFlow Lite 인터프리터만 설치할 수 있습니다.

이 인터프리터 전용 패키지의 크기는 전체 TensorFlow 패키지의 극히 일부이며 TensorFlow Lite로 추론을 실행하는 데 필요한 최소한의 코드를 포함합니다. 여기에는 [`tf.lite.Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter) Python 클래스만 포함됩니다. 이 작은 패키지는 `.tflite` 모델만 실행하고 대용량 TensorFlow 라이브러리로 디스크 공간을 낭비하지 않으려는 경우에 이상적입니다.

참고: [TensorFlow Lite Converter](../convert/python_api.md)와 같은 다른 Python API에 액세스해야 하는 경우, [전체 TensorFlow 패키지](https://www.tensorflow.org/install/)를 설치해야 합니다.

설치하려면 `pip3 install`을 실행하고 다음 표에서 적절한 Python wheel URL을 전달합니다.

예를 들어, Raspbian Buster(Python 3.7 포함)를 실행하는 Raspberry Pi를 사용 중인 경우, 다음과 같이 Python wheel을 설치합니다.

<pre class="devsite-terminal devsite-click-to-copy">pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl</pre>

<table>
<tr>
<th>플랫폼</th>
<th>Python</th>
<th>URL</th>
</tr>
<tr>
  <td style="white-space:nowrap" rowspan="4">Linux (ARM 32)</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_armv7l.whl</td>
</tr>
<tr>
  <!-- ARM 32 -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_armv7l.whl</td>
</tr>
<tr>
  <!-- ARM 32 -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl</td>
</tr>
<tr>
  <!-- ARM 32 -->
  <td style="white-space:nowrap">3.8</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp38-cp38-linux_armv7l.whl</td>
</tr>
<tr>
  <td style="white-space:nowrap" rowspan="4">Linux (ARM 64)</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_aarch64.whl</td>
</tr>
<tr>
  <!-- ARM 64 -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_aarch64.whl</td>
</tr>
<tr>
  <!-- ARM 64 -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_aarch64.whl</td>
</tr>
<tr>
  <!-- ARM 64 -->
  <td style="white-space:nowrap">3.8</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp38-cp38-linux_aarch64.whl</td>
</tr>
<tr>
  <td style="white-space:nowrap" rowspan="4">Linux (x86-64)</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_x86_64.whl</td>
</tr>
<tr>
  <!-- x86-64 -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_x86_64.whl</td>
</tr>
<tr>
  <!-- x86-64 -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_x86_64.whl</td>
</tr>
<tr>
  <!-- x86-64 -->
  <td style="white-space:nowrap">3.8</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp38-cp38-linux_x86_64.whl</td>
</tr>
<tr>
  <td style="white-space:nowrap" rowspan="3">macOS 10.14</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-macosx_10_14_x86_64.whl</td>
</tr>
<tr>
  <!-- Mac -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-macosx_10_14_x86_64.whl</td>
</tr>
<tr>
  <!-- Mac -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-macosx_10_14_x86_64.whl</td>
</tr>
<tr>
  <td style="white-space:nowrap" rowspan="3">Windows 10</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-win_amd64.whl</td>
</tr>
<tr>
  <!-- Win -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-win_amd64.whl</td>
</tr>
<tr>
  <!-- Win -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-win_amd64.whl</td>
</tr>
</table>

## tflite_runtime을 사용하여 추론 실행하기

이 인터프리터 전용 패키지를 전체 TensorFlow 패키지와 구분하기 위해(원하면 둘 다 설치할 수 있음) 위의 wheel에 제공된 Python 모듈의 이름은 `tflite_runtime`입니다.

따라서 `Interpreter`를 `tensorflow` 모듈 대신 `tflite_runtime`에서 가져와야 합니다.

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
