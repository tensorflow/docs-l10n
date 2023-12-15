# O componente de pipeline Trainer TFX

O componente de pipeline Trainer TFX treina um modelo do TensorFlow.

## Trainer e TensorFlow

O Trainer faz uso extensivo da API Python [TensorFlow](https://www.tensorflow.org) para treinar modelos.

Observação: o TFX oferece suporte ao TensorFlow 1.15 e 2.x.

## Componente

O Trainer recebe:

- tf.Examples usados ​​para treinamento e avaliação.
- Um arquivo de módulo fornecido pelo usuário que define a lógica do treinador.
- Definição [protobuf](https://developers.google.com/protocol-buffers) de argumentos de treinamento e avaliação.
- (Opcional) Um esquema de dados criado por um componente de pipeline SchemaGen e opcionalmente alterado pelo desenvolvedor.
- (Opcional) grafo de transformação produzido por um componente Transform em etapa anterior do pipeline.
- (Opcional) modelos pré-treinados usados ​​para cenários como warmstart.
- (Opcional) Hiperparâmetros, que serão passados ​​para a função do módulo do usuário. Detalhes da integração com o Tuner podem ser encontrados [aqui](tuner.md).

O Trainer produz: pelo menos um modelo para inferência/exibição (normalmente em SavedModelFormat) e opcionalmente outro modelo para avaliação (normalmente um EvalSavedModel).

Fornecemos suporte para formatos de modelo alternativos, como [TFLite](https://www.tensorflow.org/lite), por meio da [Model Rewriting Library](https://github.com/tensorflow/tfx/blob/master/tfx/components/trainer/rewriting/README.md). Siga o link para a Model Rewriting Library para exemplos de como converter modelos Estimator e Keras.

## Trainer genérico

Um Trainer genérico permite que os desenvolvedores usem qualquer API de modelo do TensorFlow com o componente Trainer. Além dos Estimators do TensorFlow, os desenvolvedores podem usar modelos Keras ou loops de treinamento personalizados. Para mais detalhes, veja a [RFC para trainer genérico](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-generic-trainer.md).

### Configurando o componente Trainer

O código DSL de pipeline típico para o Trainer genérico é o seguinte:

```python
from tfx.components import Trainer

...

trainer = Trainer(
    module_file=module_file,
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    train_args=trainer_pb2.TrainArgs(num_steps=10000),
    eval_args=trainer_pb2.EvalArgs(num_steps=5000))
```

O Trainer invoca um módulo de treinamento, que é especificado no parâmetro `module_file` (arquivo de módulo). Em vez de `trainer_fn`, um `run_fn` será necessário no arquivo do módulo se `GenericExecutor` for especificado em `custom_executor_spec`. O `trainer_fn` foi o responsável pela criação do modelo. Além disso, `run_fn` também precisa cuidar da parte de treinamento e enviar o modelo treinado para o local desejado fornecido por [FnArgs](https://github.com/tensorflow/tfx/blob/master/tfx/components/trainer/fn_args_utils.py):

```python
from tfx.components.trainer.fn_args_utils import FnArgs

def run_fn(fn_args: FnArgs) -> None:
  """Build the TF model and train it."""
  model = _build_keras_model()
  model.fit(...)
  # Save model to fn_args.serving_model_dir.
  model.save(fn_args.serving_model_dir, ...)
```

Eis um [exemplo de arquivo de módulo](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_keras.py) com `run_fn`.

Observe que se o componente Transform não for usado no pipeline, o Trainer pegaria os exemplos diretamente do ExampleGen:

```python
trainer = Trainer(
    module_file=module_file,
    examples=example_gen.outputs['examples'],
    schema=infer_schema.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=10000),
    eval_args=trainer_pb2.EvalArgs(num_steps=5000))
```

Mais detalhes estão disponíveis na [referência da API Trainer](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Trainer).
