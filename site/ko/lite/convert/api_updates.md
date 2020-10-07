# API 업데이트 <a name="api_updates"></a>

이 페이지에서는 TensorFlow 2.x의 `tf.lite.TFLiteConverter` [Python API](index.md)에 적용된 업데이트 정보를 제공합니다.

참고: 우려되는 변경 사항이 있는 경우, [GitHub 문제](https://github.com/tensorflow/tensorflow/issues/new?template=60-tflite-converter-issue.md)를 제출해 주세요.

- TensorFlow 2.3

    - 새로운 `inference_input_type` 및 `inference_output_type` 속성을 사용하여 정수 양자화 모델에서 정수 입력/출력 유형을 지원합니다(이전에는 부동 소수점만 지원됨). 이 [예제 사용법](../performance/post_training_quantization.md#integer_only)을 참조하세요.
    - 동적 차원을 이용한 모델의 변환과 크기 조정을 지원합니다.
    - 16bit 활성화 및 8bit 가중치를 사용하여 새로운 실험적 양자화 모드를 추가했습니다.

- TensorFlow 2.2

    - 기본적으로, 머신러닝을 위한 Google의 첨단 컴파일러 기술인 [MLIR 기반 변환](https://mlir.llvm.org/)을 활용합니다. 이를 통해 Mask R-CNN, Mobile BERT 등 새로운 부류의 모델 변환이 가능하고 기능적 제어 흐름이 있는 모델을 지원합니다.

- TensorFlow 2.0과 TensorFlow 1.x의 차이

    - `target_ops` 속성의 이름을 `target_spec.supported_ops`로 변경했습니다.
    - 다음 속성을 제거했습니다.
        - *양자화*: `inference_type`, `quantized_input_stats`, `post_training_quantize`, `default_ranges_stats`, `reorder_across_fake_quant`, `change_concat_input_ranges`, `get_input_arrays()`. 대신, `tf.keras` API를 통해 [양자화 인식 훈련](https://www.tensorflow.org/model_optimization/guide/quantization/training)이 지원되며, [훈련 후 양자화](../performance/post_training_quantization.md)에 더 적은 속성이 사용됩니다.
        - *시각화*: `output_format`, `dump_graphviz_dir`, `dump_graphviz_video`. 대신, TensorFlow Lite 모델 시각화에 [visualize.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py) 사용이 권장됩니다.
        - *고정 그래프*: `drop_control_dependency`, 고정 그래프는 TensorFlow 2.x에서 지원되지 않습니다.
    - `tf.lite.toco_convert` 및 `tf.lite.TocoConverter`와 같은 기타 변환기 API를 제거했습니다.
    - `tf.lite.OpHint` 및 `tf.lite.constants`와 같은 기타 관련 API를 제거했습니다(중복을 줄이기 위해 `tf.lite.constants.*` 유형이 `tf.*` TensorFlow 데이터 유형에 매핑됨).
