# 양자화 인식 훈련

<sub>TensorFlow 모델 최적화로 유지</sub>

There are two forms of quantization: post-training quantization and quantization aware training. Start with [post-training quantization](post_training.md) since it's easier to use, though quantization aware training is often better for model accuracy.

This page provides an overview on quantization aware training to help you determine how it fits with your use case.

- To dive right into an end-to-end example, see the [quantization aware training example](training_example.md).
- To quickly find the APIs you need for your use case, see the [quantization aware training comprehensive guide](training_comprehensive_guide.md).

## 개요

Quantization aware training emulates inference-time quantization, creating a model that downstream tools will use to produce actually quantized models. The quantized models use lower-precision (e.g. 8-bit instead of 32-bit float), leading to benefits during deployment.

### Deploy with quantization

Quantization brings improvements via model compression and latency reduction. With the API defaults, the model size shrinks by 4x, and we typically see between 1.5 - 4x improvements in CPU latency in the tested backends. Eventually, latency improvements can be seen on compatible machine learning accelerators, such as the [EdgeTPU](https://coral.ai/docs/edgetpu/benchmarks/) and NNAPI.

The technique is used in production in speech, vision, text, and translate use cases. The code currently supports a [subset of these models](#general-support-matrix).

### Experiment with quantization and associated hardware

Users can configure the quantization parameters (e.g. number of bits) and to some degree, the underlying algorithms. With these changes from the API defaults, there is no supported path to deployment.

이 구성과 관련된 API는 실험적이며 이전 버전과의 호환성이 적용되지 않습니다.

### API 호환성

Users can apply quantization with the following APIs:

- Model building: `tf.keras` with only Sequential and Functional models.
- TensorFlow versions: TF 2.x for tf-nightly.
    - `tf.compat.v1` with a TF 2.X package is not supported.
- TensorFlow execution mode: eager execution

It is on our roadmap to add support in the following areas:

<!-- TODO(tfmot): file Github issues. -->

- Model building: clarify how Subclassed Models have limited to no support
- Distributed training: `tf.distribute`

### General support matrix

Support is available in the following areas:

- Model coverage: models using [whitelisted layers](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/quantization/keras/default_8bit/default_8bit_quantize_registry.py), BatchNormalization when it follows Conv2D and DepthwiseConv2D layers, and in limited cases, `Concat`.
    <!-- TODO(tfmot): add more details and ensure they are all correct. -->
- Hardware acceleration: our API defaults are compatible with acceleration on EdgeTPU, NNAPI, and TFLite backends, amongst others. See the caveat in the roadmap.
- Deploy with quantization: only per-axis quantization for convolutional layers, not per-tensor quantization, is currently supported.

It is on our roadmap to add support in the following areas:

<!-- TODO(tfmot): file Github issue. Update as more functionality is added prior
to launch. -->

- Model coverage: extended to include RNN/LSTMs and general Concat support.
- Hardware acceleration: ensure the TFLite converter can produce full-integer models. See [this issue](https://github.com/tensorflow/tensorflow/issues/38285) for details.
- Experiment with quantization use cases:
    - Experiment with quantization algorithms that span Keras layers or require the training step.
    - API를 안정화합니다.

## 결과

### 도구를 사용한 이미지 분류

<figure>
  <table>
    <tr>
      <th>모델</th>
      <th>Non-quantized Top-1 Accuracy </th>
      <th>8-bit Quantized Accuracy </th>
    </tr>
    <tr>
      <td>MobilenetV1 224</td>
      <td>71.03%</td>
      <td>71.06%</td>
    </tr>
    <tr>
      <td>Resnet v1 50</td>
      <td>76.3%</td>
      <td>76.1%</td>
    </tr>
    <tr>
      <td>MobilenetV2 224</td>
      <td>70.77%</td>
      <td>70.01%</td>
    </tr>
 </table>
</figure>

모델은 Imagenet에서 테스트되었으며 TensorFlow 및 TFLite에서 평가되었습니다.

### 기술에 대한 이미지 분류

<figure>
  <table>
    <tr>
      <th>모델</th>
      <th>Non-quantized Top-1 Accuracy </th>
      <th>8-Bit Quantized Accuracy </th>
    </tr>
<tr>
      <td>Nasnet-Mobile</td>
      <td>74 %</td>
      <td>73 %</td>
    </tr>
    <tr>
      <td>Resnet-v2 50</td>
      <td>75.6 %</td>
      <td>75 %</td>
    </tr>
 </table>
</figure>

모델은 Imagenet에서 테스트되었으며 TensorFlow 및 TFLite에서 평가되었습니다.

## Examples

In addition to the [quantization aware training example](training_example.md), see the following examples:

- CNN model on the MNIST handwritten digit classification task with quantization: [code](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/quantization/keras/quantize_functional_test.py)

For background on something similar, see the *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* [paper](https://arxiv.org/abs/1712.05877). This paper introduces some concepts that this tool uses. The implementation is not exactly the same, and there are additional concepts used in this tool (e.g. per-axis quantization).
