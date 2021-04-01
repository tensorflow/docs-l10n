# TensorFlow Lite 및 TensorFlow 연산자 호환성

TensorFlow Lite는 공통 추론 모델에 사용되는 여러 TensorFlow 연산을 지원합니다. 이들 연산은 TensorFlow Lite 최적화 변환기에서 처리되므로, 지원되는 연산이 TensorFlow Lite의 해당 연산에 매핑되기 전에 제거되거나 통합될 수 있습니다.

Since the TensorFlow Lite builtin operator library only supports a limited number of TensorFlow operators, not every model is convertible. Even for supported operations, very specific usage patterns are sometimes expected, for performance reasons. We expect to expand the set of supported operations in future TensorFlow Lite releases.

TensorFlow Lite에서 동작하는 TensorFlow 모델을 빌드하는 방법을 이해하는 가장 좋은 방법은 이 프로세스에 적용되는 제한 사항과 함께 연산이 어떻게 변환되고 최적화되는지를 신중하게 고려하는 것입니다.

## 지원되는 유형

대부분의 TensorFlow Lite 연산은 부동 소수점(`float32`) 및 양자화된(`uint8`, `int8`) 추론을 모두 대상으로 하지만, 많은 연산은 아직 `tf.float16` 및 문자열과 같은 다른 유형을 처리하지 않습니다.

Apart from using different version of the operations, the other difference between floating-point and quantized models is the way they are converted. Quantized conversion requires dynamic range information for tensors. This requires "fake-quantization" during model training, getting range information via a calibration data set, or doing "on-the-fly" range estimation. See [quantization](../performance/model_optimization.md).

## Supported operations and restrictions

TensorFlow Lite supports a subset of TensorFlow operations with some limitations. For full list of operations and limitations see [TF Lite Ops page](https://www.tensorflow.org/mlir/tfl_ops).

## Straight-forward conversions, constant-folding and fusing

A number of TensorFlow operations can be processed by TensorFlow Lite even though they have no direct equivalent. This is the case for operations that can be simply removed from the graph (`tf.identity`), replaced by tensors (`tf.placeholder`), or fused into more complex operations (`tf.nn.bias_add`). Even some supported operations may sometimes be removed through one of these processes.

Here is a non-exhaustive list of TensorFlow operations that are usually removed from the graph:

- `tf.add`
- `tf.check_numerics`
- `tf.constant`
- `tf.div`
- `tf.divide`
- `tf.fake_quant_with_min_max_args`
- `tf.fake_quant_with_min_max_vars`
- `tf.identity`
- `tf.maximum`
- `tf.minimum`
- `tf.multiply`
- `tf.no_op`
- `tf.placeholder`
- `tf.placeholder_with_default`
- `tf.realdiv`
- `tf.reduce_max`
- `tf.reduce_min`
- `tf.reduce_sum`
- `tf.rsqrt`
- `tf.shape`
- `tf.sqrt`
- `tf.square`
- `tf.subtract`
- `tf.tile`
- `tf.nn.batch_norm_with_global_normalization`
- `tf.nn.bias_add`
- `tf.nn.fused_batch_norm`
- `tf.nn.relu`
- `tf.nn.relu6`

Note: Many of those operations don't have TensorFlow Lite equivalents, and the corresponding model will not be convertible if they can't be elided or fused.

## Experimental Operations

The following TensorFlow Lite operations are present, but not ready for custom models:

- `CALL`
- `CONCAT_EMBEDDINGS`
- `CUSTOM`
- `EMBEDDING_LOOKUP_SPARSE`
- `HASHTABLE_LOOKUP`
- `LSH_PROJECTION`
- `SKIP_GRAM`
- `SVDF`
