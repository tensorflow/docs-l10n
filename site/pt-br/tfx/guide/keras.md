# TensorFlow 2.x no TFX

O [TensorFlow 2.0 foi lançado em 2019](https://blog.tensorflow.org/2019/09/tensorflow-20-is-now-available.html), com [forte integração com o Keras](https://www.tensorflow.org/guide/keras/overview), [execução eager](https://www.tensorflow.org/guide/eager) por padrão e [execução de funções no estilo Python](https://www.tensorflow.org/guide/function), entre outros [novos recursos e melhorias](https://www.tensorflow.org/guide/effective_tf2#a_brief_summary_of_major_changes).

Este guia fornece uma visão geral técnica abrangente do TF 2.x no TFX.

## Qual versão usar?

O TFX é compatível com o TensorFlow 2.x, e as APIs de alto nível que existiam no TensorFlow 1.x (principalmente Estimators) continuam funcionando.

### Inicie novos projetos no TensorFlow 2.x

Já que o TensorFlow 2.x mantém os recursos de alto nível do TensorFlow 1.x, não há vantagem em usar a versão mais antiga em novos projetos, mesmo que você não planeje usar os novos recursos.

Portanto, se você estiver iniciando um novo projeto TFX, recomendamos usar o TensorFlow 2.x. Talvez você queira atualizar seu código posteriormente, à medida que o suporte completo para Keras e outros novos recursos estiver disponível, e o escopo das alterações será muito menor se você começar logo com o TensorFlow 2.x, em vez de tentar atualizar do TensorFlow 1.x em o futuro.

### Convertendo projetos existentes para o TensorFlow 2.x

O código escrito para o TensorFlow 1.x é amplamente compatível com o TensorFlow 2.x e continuará funcionando no TFX.

No entanto, se quiser aproveitar as melhorias e os novos recursos à medida que forem disponibilizados no TF 2.x, você pode seguir as [instruções para migrar para o TF 2.x](https://www.tensorflow.org/guide/migrate).

## Estimator

A API Estimator foi mantida no TensorFlow 2.x, mas não é o foco de novos recursos e desenvolvimento. O código escrito no TensorFlow 1.x ou 2.x usando Estimators continuará funcionando conforme esperado no TFX.

Aqui está um exemplo de TFX ponta a ponta usando o Estimator puro: [Exemplo do táxi (Estimator)](https://github.com/tensorflow/tfx/blob/r0.21/tfx/examples/chicago_taxi_pipeline/taxi_utils.py)

## Keras com `model_to_estimator`

Os modelos Keras podem ser agrupados com a função `tf.keras.estimator.model_to_estimator`, que permite que funcionem como se fossem Estimators. Para usar:

1. Construa um modelo Keras.
2. Passe o modelo compilado para `model_to_estimator`.
3. Use o resultado de `model_to_estimator` no Trainer, da mesma forma que você normalmente usaria um Estimator.

```py
# Build a Keras model.
def _keras_model_builder():
  """Creates a Keras model."""
  ...

  model = tf.keras.Model(inputs=inputs, outputs=output)
  model.compile()

  return model


# Write a typical trainer function
def trainer_fn(trainer_fn_args, schema):
  """Build the estimator, using model_to_estimator."""
  ...

  # Model to estimator
  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=_keras_model_builder(), config=run_config)

  return {
      'estimator': estimator,
      ...
  }
```

Além do arquivo do módulo do usuário do Trainer, o restante do pipeline permanece inalterado.

## Keras nativo (ou seja, Keras sem `model_to_estimator`)

Observação: O suporte completo para todos os recursos do Keras está em andamento; na maioria dos casos, o Keras no TFX funcionará conforme o esperado. Ainda não funciona com Sparse Features para FeatureColumns.

### Exemplos e Colab

Aqui estão vários exemplos com Keras nativo:

- [Penguin](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py) ([arquivo de módulo](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_keras.py)): exemplo 'Hello world' completo.
- [MNIST](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py) ([arquivo de módulo](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_utils_native_keras.py)): exemplo com Image e TFLite completo.
- [Taxi](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_native_keras.py) ([arquivo de módulo](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_utils_native_keras.py)): exemplo completo com uso de Transform.

Também temos um [Keras Colab](https://www.tensorflow.org/tfx/tutorials/tfx/components_keras) por componente.

### Componentes TFX

As seguintes seções explicam como os componentes TFX relacionados oferecem suporte ao Keras nativo.

#### Transform

Atualmente, o Transform tem suporte experimental para modelos Keras.

O próprio componente Transform pode ser usado no Keras nativo sem alterações. A definição `preprocessing_fn` permanece a mesma, usando ops do [TensorFlow](https://www.tensorflow.org/api_docs/python/tf) e [tf.Transform](https://www.tensorflow.org/tfx/transform/api_docs/python/tft).

A função serving e a função eval foram alteradas para Keras nativo. Os detalhes serão discutidos nas seções a seguir do Trainer e do Evaluator.

Observação: As transformações dentro de `preprocessing_fn` não podem ser aplicadas à característica de rótulo para treinamento ou avaliação.

#### Trainer

Para configurar o Keras nativo, o `GenericExecutor` precisa ser definido para o componente Trainer para substituir o executor padrão baseado no Estimator. Para mais detalhes, clique [aqui](trainer.md#configuring-the-trainer-component-to-use-the-genericexecutor).

##### Arquivo Keras Module com Transform

O arquivo do módulo de treinamento deve conter um `run_fn` que será chamado pelo `GenericExecutor`. Um `run_fn` típico do Keras ficaria assim:

```python
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  # Train and eval files contains transformed examples.
  # _input_fn read dataset based on transformed schema from tft.
  train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor,
                            tf_transform_output.transformed_metadata.schema)
  eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor,
                           tf_transform_output.transformed_metadata.schema)

  model = _build_keras_model()

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
```

No `run_fn` acima, uma assinatura de serviço é necessária ao exportar o modelo treinado para que o modelo possa obter exemplos brutos para previsão. Uma função de serviço típica seria assim:

```python
def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""

  # the layer is added as an attribute to the model in order to make sure that
  # the model assets are handled correctly when exporting.
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = model.tft_layer(parsed_features)

    return model(transformed_features)

  return serve_tf_examples_fn
```

Na função de serviço acima, as transformações tf.Transform precisam ser aplicadas aos dados brutos para inferência, usando a camada [`tft.TransformFeaturesLayer`](https://www.tensorflow.org/tfx/transform/api_docs/python/tft/TransformFeaturesLayer). O `_serving_input_receiver_fn` anterior que era necessário para Estimators não será mais necessário com Keras.

##### Arquivo Keras Module sem Transform

Isto é semelhante ao arquivo do módulo mostrado acima, mas sem as transformações:

```python
def _get_serve_tf_examples_fn(model, schema):

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    feature_spec = _get_raw_feature_spec(schema)
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    return model(parsed_features)

  return serve_tf_examples_fn


def run_fn(fn_args: TrainerFnArgs):
  schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())

  # Train and eval files contains raw examples.
  # _input_fn reads the dataset based on raw data schema.
  train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, schema)
  eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, schema)

  model = _build_keras_model()

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model, schema).get_concrete_function(
              tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
```

##### [tf.distribute.Strategy](https://www.tensorflow.org/guide/distributed_training)

No momento, o TFX oferece suporte apenas a estratégias de worker único (por exemplo, [MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy), [OneDeviceStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/OneDeviceStrategy)).

Para usar uma estratégia de distribuição, crie um tf.distribute.Strategy apropriado e mova a criação e compilação do modelo Keras dentro de um escopo de estratégia.

Por exemplo, substitua `model = _build_keras_model()` acima por:

```python
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model()

  # Rest of the code can be unchanged.
  model.fit(...)
```

Para verificar o dispositivo (CPU/GPU) usado por `MirroredStrategy`, habilite o log do TensorFlow para o nível info:

```python
import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)
```

e você deverá ver `Using MirroredStrategy with devices (...)` no log.

Observação: A variável de ambiente `TF_FORCE_GPU_ALLOW_GROWTH=true` pode ser necessária para um problema de falta de memória da GPU. Para mais detalhes, veja o [Guia da GPU Tensorflow](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth).

#### Evaluator

No TFMA v0.2x, ModelValidator e Evaluator foram combinados num único [novo componente Evaluator](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-combining-model-validator-with-evaluator.md). O novo componente Evaluator pode tanto realizar a avaliação de modelo único, como também validar o modelo atual em comparação com modelos anteriores. Com essa alteração, o componente Pusher agora consome um resultado de blessing do Evaluator em vez do ModelValidator.

O novo Evaluator oferece suporte a modelos Keras e também a modelos Estimator. O `_eval_input_receiver_fn` e o modelo salvo eval que eram exigidos anteriormente não serão mais necessários com Keras, pois o Evaluator agora é baseado no mesmo `SavedModel` usado ao servir.

[Veja Evaluator para mais informações](evaluator.md).
