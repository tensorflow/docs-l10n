# TFX para dispositivos móveis

## Introdução

Este guia demonstra como o Tensorflow Extended (TFX) pode criar e avaliar modelos de aprendizado de máquina que serão implantados em dispositivos. O TFX agora oferece suporte nativo ao [TFLite](https://www.tensorflow.org/lite), o que possibilita realizar inferências altamente eficientes em dispositivos móveis.

Este guia orienta você nas alterações que podem ser feitas em qualquer pipeline para gerar e avaliar modelos TFLite. Fornecemos um exemplo completo [aqui](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py), demonstrando como o TFX pode treinar e avaliar modelos TFLite que são treinados no dataset [MNIST](http://yann.lecun.com/exdb/mnist/). Além disso, mostramos como o mesmo pipeline pode ser usado para exportar simultaneamente tanto o [SavedModel](https://www.tensorflow.org/guide/saved_model) padrão baseado em Keras quanto o TFLite, permitindo aos usuários comparar a qualidade dos dois.

Presumimos que você esteja familiarizado com o TFX, nossos componentes e pipelines. Caso contrário, consulte este [tutorial](https://www.tensorflow.org/tfx/tutorials/tfx/components).

## Passos

Apenas dois passos são necessários para criar e avaliar um modelo TFLite no TFX. O primeiro passo é chamar o TFLite rewriter no contexto do [TFX Trainer](https://www.tensorflow.org/tfx/guide/trainer) para converter o modelo TensorFlow treinado num modelo TFLite. O segundo passo é configurar o Evaluator para avaliar modelos TFLite. Agora discutiremos cada um desses passos.

### Chamando o TFLite rewriter dentro do Trainer.

O TFX Trainer espera que um `run_fn` definido pelo usuário seja especificado num arquivo de módulo. Este `run_fn` define o modelo a ser treinado, treina-o para o número especificado de iterações e exporta o modelo treinado.

No restante desta seção, fornecemos trechos de código que mostram as alterações necessárias para chamar o TFLite rewriter e exportar um modelo TFLite. Todo esse código está localizado no `run_fn` do [módulo MNIST TFLite](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_utils_native_keras_lite.py).

Conforme mostrado no código abaixo, devemos primeiro criar uma assinatura que receba um `Tensor` para cada característica como entrada. Observe que isso é diferente da maioria dos modelos existentes no TFX, que usam protos [tf.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example) serializados como entrada.

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

Em seguida, o modelo Keras é salvo como SavedModel da mesma forma que normalmente é.

```python
  temp_saving_model_dir = os.path.join(fn_args.serving_model_dir, 'temp')
  model.save(temp_saving_model_dir, save_format='tf', signatures=signatures)
```

Finalmente, criamos uma instância do TFLite rewriter (`tfrw`) e o chamamos no SavedModel para obter o modelo TFLite. Armazenamos este modelo TFLite no `serving_model_dir` fornecido pelo chamador de `run_fn`. Dessa forma, o modelo TFLite é armazenado no local onde todos os componentes downstream do TFX esperam encontrar o modelo.

```python
  tfrw = rewriter_factory.create_rewriter(
      rewriter_factory.TFLITE_REWRITER, name='tflite_rewriter')
  converters.rewrite_saved_model(temp_saving_model_dir,
                                 fn_args.serving_model_dir,
                                 tfrw,
                                 rewriter.ModelType.TFLITE_MODEL)
```

### Avaliando o modelo TFLite.

O [TFX Evaluator](https://www.tensorflow.org/tfx/guide/evaluator) oferece a capacidade de analisar modelos treinados para compreender sua qualidade numa ampla variedade de métricas. Além de analisar SavedModels, o TFX Evaluator agora também é capaz de analisar modelos TFLite.

O trecho de código a seguir (reproduzido do [pipeline MNIST](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py)) mostra como configurar um Evaluator que analisa um modelo TFLite.

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

Conforme mostrado acima, a única alteração que precisamos fazer é definir o campo `model_type` como `tf_lite`. Nenhuma outra alteração de configuração é necessária para analisar o modelo TFLite. Independentemente de ser analisado um modelo TFLite ou um SavedModel, a saída do `Evaluator` terá exatamente a mesma estrutura.

No entanto, observe que o Evaluator assume que o modelo TFLite é salvo num arquivo chamado `tflite` dentro de trainer_lite.outputs['model'].
