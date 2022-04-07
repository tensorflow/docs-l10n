# 모바일용 TFX

## 시작하기

이 가이드는 Tensorflow Extended(TFX)가 기기에 배포될 머신러닝 모델을 만들고 평가하는 방법을 보여줍니다. 이제 TFX는 [TFLite](https://www.tensorflow.org/lite)에 대한 기본 지원을 제공하므로 모바일 기기에서 매우 효율적인 추론을 수행할 수 있습니다.

이 가이드는 TFLite 모델을 생성하고 평가하기 위해 파이프라인에 적용할 수 있는 변경 사항을 안내합니다. [여기](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py)서는 [MNIST](http://yann.lecun.com/exdb/mnist/) 데이터세트에서 훈련된 TFLite 모델을 TFX가 어떻게 훈련하고 평가할 수 있는 지를 보여주는 완전한 예를 제공합니다. 또한 동일한 파이프라인을 사용하여 표준 Keras 기반 [SavedModel](https://www.tensorflow.org/guide/saved_model)과 TFLite 기반 SavedModel을 동시에 내보내 사용자가 두 개의 품질을 비교할 수 있도록 하는 과정도 보여줍니다.

TFX, 구성 요소 및 파이프라인에 익숙하다고 가정합니다. 그렇지 않은 경우 이 [튜토리얼](https://www.tensorflow.org/tfx/tutorials/tfx/components)을 참조하세요.

## 단계

TFX에서 TFLite 모델을 만들고 평가하는 데 두 단계만 필요합니다. 첫 번째 단계는 [TFX Trainer](https://www.tensorflow.org/tfx/guide/trainer) 컨텍스트 내에서 TFLite rewriter를 호출하여 훈련된 TensorFlow 모델을 TFLite 모델로 변환하는 것입니다. 두 번째 단계는 TFLite 모델을 평가하도록 Evaluator를 구성하는 것입니다. 이제 각각에 대해 차례로 논의합니다.

### Trainer 내에서 TFLite rewriter 호출하기

TFX Trainer는 사용자 정의 `run_fn`이 모듈 파일에 지정될 것으로 예상합니다. 이 `run_fn`은 훈련할 모델을 정의하고 지정된 반복 횟수만큼 훈련하고 훈련된 모델을 내보냅니다.

이 섹션의 나머지 부분에서는 TFLite rewriter를 호출하고 TFLite 모델을 내보내는 데 필요한 변경 사항을 보여주는 코드 조각을 제공합니다. 이 모든 코드는 [MNIST TFLite 모듈](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_utils_native_keras_lite.py)의 `run_fn`에 있습니다.

아래 코드에서 볼 수 있듯이 먼저 모든 특성에 대한 `Tensor`를 입력으로 취하는 서명을 생성해야 합니다. 직렬화된 [tf.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example) proto를 입력으로 취하는 대부분의 기존 TFX 모델과는 다른 부분입니다.

```python
 signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(
              model, tf_transform_output).get_concrete_function(
                  tf.TensorSpec(
                      shape=[None, 784],
                      dtype=tf.float32,
                      name='image_floats'))
  }
```

그런 다음 Keras 모델은 평소와 같은 방식으로 SavedModel로 저장됩니다.

```python
  temp_saving_model_dir = os.path.join(fn_args.serving_model_dir, 'temp')
  model.save(temp_saving_model_dir, save_format='tf', signatures=signatures)
```

마지막으로 TFLite rewriter(`tfrw`)의 인스턴스를 만들고 SavedModel에서 호출하여 TFLite 모델을 얻습니다. `run_fn`의 호출자가 제공하는 `serving_model_dir`에 이 TFLite 모델을 저장합니다. 그러면 TFLite 모델이 모든 다운스트림 TFX 구성 요소가 모델을 찾을 것으로 예상되는 위치에 저장됩니다.

```python
  tfrw = rewriter_factory.create_rewriter(
      rewriter_factory.TFLITE_REWRITER, name='tflite_rewriter')
  converters.rewrite_saved_model(temp_saving_model_dir,
                                 fn_args.serving_model_dir,
                                 tfrw,
                                 rewriter.ModelType.TFLITE_MODEL)
```

### TFLite 모델 평가하기

[TFX Evaluator](https://www.tensorflow.org/tfx/guide/evaluator)를 이용하면 훈련된 모델을 분석하여 광범위한 메트릭에 걸쳐 품질을 이해할 수 있습니다. SavedModel을 분석하는 외에도 TFX Evaluator는 이제 TFLite 모델도 분석할 수 있습니다.

다음 코드 조각([MNIST 파이프라인](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py)에서 재현)은 TFLite 모델을 분석하는 Evaluator의 구성 방법을 보여줍니다.

```python
  # Informs the evaluator that the model is a TFLite model.
  eval_config_lite.model_specs[0].model_type = 'tf_lite'

  ...

  # Uses TFMA to compute the evaluation statistics over features of a TFLite
  # model.
  model_analyzer_lite = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer_lite.outputs['model'],
      eval_config=eval_config_lite,
  ).with_id('mnist_lite')
```

위에 표시된 것처럼 필요한 유일한 변경은 `model_type` 필드를 `tf_lite`로 설정하는 것입니다. TFLite 모델을 분석하기 위해 다른 구성 변경이 필요하지 않습니다. TFLite 모델을 분석하든 SavedModel을 분석하든 상관없이 `Evaluator`의 출력은 정확히 동일한 구조를 갖습니다.

단, Evaluator는 TFLite 모델이 trainer_lite.outputs['model'] 내의 `tflite`라는 파일에 저장되어 있다고 가정합니다.
