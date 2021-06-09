# 훈련 후 양자화

훈련 후 양자화는 모델 정확성을 거의 저하시키지 않으면서 CPU 및 하드웨어 가속기 지연 시간을 개선하고 모델 크기를 줄일 수 있는 변환 기술입니다. [TensorFlow Lite 변환기](../convert/)를 사용하여 TensorFlow Lite 형식으로 변환할 때 이미 훈련된 부동 TensorFlow 모델을 양자화할 수 있습니다.

참고: 해당 페이지의 절차를 따르려면 TensorFlow 1.15 이상이 필요합니다.

### 최적화 메서드

선택할 수 있는 몇 가지 훈련 후 양자화 옵션이 있습니다. 다음은 선택 항목과 선택 항목이 제공하는 이점에 대한 요약표입니다.

기술 | 이점 | 하드웨어
--- | --- | ---
동적 범위 | 4배 작아짐, 2배-3배 속도 향상 | CPU
: 양자화 : : : |  |
전체 정수 | 4배 더 작게, 3배 이상의 속도 향상 | CPU, 에지 TPU
: 양자화 : : 마이크로 컨트롤러 : |  |
Float16 양자화 | 2배 더 작아진 GPU | CPU, GPU
: : 가속 : : |  |

다음 의사 결정 트리는 사용 사례에 가장 적합한 훈련 후 양자화 메서드를 결정하는 데 도움이 될 수 있습니다.

![훈련 후 최적화 옵션](images/optimization.jpg)

### 동적 범위 양자화

훈련 후 양자화의 가장 간단한 형태는 8bit의 정밀도를 가진 부동 소수점에서 정수까지 가중치만 정적으로 양자화합니다.

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

추론에서 가중치는 8bit 정밀도에서 부동 소수점으로 변환되고 부동 소수점 커널을 사용하여 계산됩니다. 이 변환은 한 번 수행되고 캐시되어 지연 시간을 줄입니다.

To further improve latency, "dynamic-range" operators dynamically quantize activations based on their range to 8-bits and perform computations with 8-bit weights and activations. This optimization provides latencies close to fully fixed-point inference. However, the outputs are still stored using floating point so that the speedup with dynamic-range ops is less than a full fixed-point computation.

### 전체 정수 양자화

모든 모델 수학이 정수 양자화되었는지 확인하여 추가 지연 시간 개선, 최대 메모리 사용량 감소, 정수 전용 하드웨어 기기 또는 가속기와의 호환성을 얻을 수 있습니다.

For full integer quantization, you need to calibrate or estimate the range, i.e, (min, max) of all floating-point tensors in the model. Unlike constant tensors such as weights and biases, variable tensors such as model input, activations (outputs of intermediate layers) and model output cannot be calibrated unless we run a few inference cycles. As a result, the converter requires a representative dataset to calibrate them. This dataset can be a small subset (around ~100-500 samples) of the training or validation data. Refer to the `representative_dataset()` function below.

<pre>def representative_dataset():
  for data in tf.data.Dataset.from_tensor_slices((images)).batch(1).take(100):
    yield [tf.dtypes.cast(data, tf.float32)]
</pre>

For testing purposes, you can use a dummy dataset as follows:

<pre>def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 244, 244, 3)
      yield [data.astype(np.float32)]</pre>

#### 부동 폴 백이 있는 정수(기본 부동 입력/출력 사용하기)

모델을 완전히 정수로 양자화하지만 정수 구현이 없는 경우 부동 연산자를 사용하려면(원활하게 변환을 수행하기 위해) 다음 스탭을 사용합니다.

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&amp;lt;b&amp;gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset&amp;lt;/b&amp;gt;
tflite_quant_model = converter.convert()</pre>

참고: `tflite_quant_model`은 정수 전용 기기(예: 8bit 마이크로 컨트롤러) 및 가속기(예: Coral 에지 TPU)와 호환되지 않습니다. 입력 및 출력이 원본 부동 모델과 같은 인터페이스를 갖기 위해 여전히 부동 상태로 남아 있기 때문입니다.

#### 정수 전용

*정수 전용 모델을 만드는 것은 [마이크로 컨트롤러용 TensorFlow Lite](https://www.tensorflow.org/lite/microcontrollers) 및 [Coral 에지 TPU](https://coral.ai/)의 일반적인 사용 사례입니다.*

참고: TensorFlow 2.3.0부터는 `inference_input_type` 및 `inference_output_type` 속성을 지원합니다.

또한 정수 전용 기기(예: 8bit 마이크로 컨트롤러) 및 가속기(예: Coral 에지 TPU)와의 호환성을 보장하기 위해 다음 스탭을 사용하여 입력 및 출력을 포함한 모든 연산에 대해 전체 정수 양자화를 적용할 수 있습니다.

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
&lt;b&gt;converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]&lt;/b&gt;
&lt;b&gt;converter.inference_input_type = tf.int8&lt;/b&gt;  # or tf.uint8
&lt;b&gt;converter.inference_output_type = tf.int8&lt;/b&gt;  # or tf.uint8
tflite_quant_model = converter.convert()
</pre>

참고: 변환기는 현재 양자화할 수 없는 연산이 일어나면 오류를 발생시킵니다.

### Float16 양자화

가중치를 16bit 부동 소수점 숫자에 대한 IEEE 표준인 float16으로 양자화하여 부동 소수점 모델의 크기를 줄일 수 있습니다. 가중치의 float16 양자화를 활성화하려면 다음 스탭을 사용합니다.

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

float16 양자화의 장점은 다음과 같습니다.

- (모든 가중치가 원래 크기의 절반이되므로) 모델 크기를 최대 절반까지 줄입니다.
- 정확성 손실을 최소화합니다.
- float16 데이터에서 직접 동작할 수 있는 일부 대리자(예: GPU 대리자)를 지원하므로 float32 계산보다 빠른 실행이 가능합니다.

float16 양자화의 단점은 다음과 같습니다.

- 고정 소수점 수학에 대한 양자화만큼 지연 시간을 줄이지 않습니다.
- By default, a float16 quantized model will "dequantize" the weights values to float32 when run on the CPU. (Note that the GPU delegate will not perform this dequantization, since it can operate on float16 data.)

### 정수 전용: 8bit 가중치를 사용한 16bit 활성화(실험적)

This is an experimental quantization scheme. It is similar to the "integer only" scheme, but activations are quantized based on their range to 16-bits, weights are quantized in 8-bit integer and bias is quantized into 64-bit integer. This is referred to as 16x8 quantization further.

이 양자화의 주요 장점은 정확성을 크게 향상시키면서도 모델 크기는 아주 약간만 늘릴 수 있다는 것입니다.

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.representative_dataset = representative_dataset
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

If 16x8 quantization is not supported for some operators in the model, then the model still can be quantized, but unsupported operators kept in float. The following option should be added to the target_spec to allow this.

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.representative_dataset = representative_dataset
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
&lt;b&gt;tf.lite.OpsSet.TFLITE_BUILTINS&lt;/b&gt;]
tflite_quant_model = converter.convert()
</pre>

Examples of the use cases where accuracy improvements provided by this quantization scheme include: * super-resolution, * audio signal processing such as noise cancelling and beamforming, * image de-noising, * HDR reconstruction from a single image.

이 양자화의 단점은 다음과 같습니다.

- 현재 추론은 최적화된 커널 구현이 없기 때문에 8bit 정수보다 눈에 띄게 느립니다.
- 현재 기존 하드웨어 가속 TFLite 대리자와 호환되지 않습니다.

참고: 이것은 실험적인 특성입니다.

A tutorial for this quantization mode can be found [here](post_training_integer_quant_16x8.ipynb).

### 모델 정확성

가중치는 훈련 후에 양자화되기 때문에 특히 소규모 네트워크의 경우 정확성 손실이 있을 수 있습니다. [TensorFlow Lite 모델 리포지토리](../models/)에서 특정 네트워크에 대해 사전 훈련된 전체 양자화 모델이 제공됩니다. 정확성 저하가 허용 가능한 한계 내에 있는지 확인하기 위해 양자화된 모델의 정확성을 확인하는 것이 중요합니다. [TensorFlow Lite 모델 정확성](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks){:.external}을 평가하는 도구가 있습니다.

또는 정확성 저하가 너무 크면 [양자화 인식 훈련](https://www.tensorflow.org/model_optimization/guide/quantization/training)을 사용해 볼 수 있습니다. 그러나 해당 훈련을 사용하려면 모델 훈련 중에 수정하여 가짜 양자화 노드를 추가해야 하지만, 이 페이지의 훈련 후 양자화 기술은 기존의 사전 훈련된 모델을 사용합니다.

### 양자화된 텐서 표현

8bit 양자화는 다음 공식을 사용하여 부동 소수점 값을 근사화합니다.

$$real_value = (int8_value - zero_point) \times scale$$

표현에는 두 가지 주요 부분이 있습니다.

- 축당(일명 채널당) 또는 텐서당 가중치는 0과 같은 영점이 있는 [-127, 127] 범위의 int8 2의 보수 값으로 표시됩니다.

- 텐서별 활성화/입력은 [-128, 127] 범위의 int8 2의 보수 값으로 표시되며 [-128, 127] 범위의 영점을 포함합니다.

양자화 체계에 대한 자세한 내용은 [양자화 사양](./quantization_spec.md)을 참조하세요. TensorFlow Lite의 대리자 인터페이스에 연결하려는 하드웨어 공급 업체는 여기에 설명된 양자화 체계를 구현해보는 것이 좋습니다.
