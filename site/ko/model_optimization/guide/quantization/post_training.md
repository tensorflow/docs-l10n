# 훈련 후 양자화

훈련 후 양자화에는 모델 정확성의 저하 없이 CPU 및 하드웨어 가속기의지연 시간, 처리, 전력 및 모델 크기를 줄여주는 일반적인 기술이 포함됩니다. 이러한 기술은 이미 훈련된 float TensorFlow 모델에서 수행할 수 있으며 TensorFlow Lite 변환 중에 적용할 수 있습니다. 이러한 기술은 [TensorFlow Lite 변환기](https://www.tensorflow.org/lite/convert/)의 옵션으로 활성화됩니다.

엔드 투 엔드 예제로 바로 이동하려면, 다음 튜토리얼을 참조하세요.

- [훈련 후 동적 범위 양자화](https://www.tensorflow.org/lite/performance/post_training_quant)
- [훈련 후 전체 정수 양자화](https://www.tensorflow.org/lite/performance/post_training_integer_quant)
- [훈련 후 float16 양자화](https://www.tensorflow.org/lite/performance/post_training_float16_quant)

## 가중치 양자화

가중치는 16bit 부동 소수점 또는 8bit 정수와 같이 정밀도가 낮은 유형으로 변환할 수 있습니다. 일반적으로 GPU 가속에는 16bit 부동 소수점, CPU 실행에는 8bit 정수를 권장합니다.

예를 들어, 다음은 8bit 정수 가중치 양자화를 지정하는 방법입니다.

```
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
```

추론에서 가장 중요한 부분은 부동 소수점 대신 8bit로 계산됩니다. 아래의 가중치와 활성화를 양자화하는 것과 관련된 추론 시간 성능 오버헤드가 있습니다.

자세한 내용은 TensorFlow Lite [훈련 후 양자화](https://www.tensorflow.org/lite/performance/post_training_quantization) 가이드를 참조하세요.

## 가중치 및 활성화의 전체 정수 양자화

지연 시간, 처리 및 전력 사용량을 개선하고, 가중치와 활성화가 모두 양자화되었는지 확인하여 정수 전용 하드웨어 가속기에 액세스할 수 있습니다. 이를 위해서는 대표적인 작은 데이터세트가 필요합니다.

```
import tensorflow as tf

def representative_dataset_gen():
  for _ in range(num_calibration_steps):
    # Get sample input data as a numpy array in a method of your choosing.
    yield [input]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
```

결과 모델은 편의를 위해 여전히 float 입력 및 출력을 사용합니다.

자세한 내용은 TensorFlow Lite [훈련 후 양자화](https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization_of_weights_and_activations) 가이드를 참조하세요.
