# O componente de pipeline TFX Tuner

O componente Tuner ajusta os hiperparâmetros do modelo.

## Componente Tuner e Biblioteca KerasTuner

O componente Tuner faz uso extensivo da API Python [KerasTuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) para ajustar hiperparâmetros.

Observação: A biblioteca KerasTuner pode ser usada para a tunagem de hiperparâmetros, independentemente da API de modelagem, e não apenas para modelos Keras.

## Componente

O Tuner recebe:

- tf.Examples usados ​​para treinamento e avaliação.
- Um arquivo de módulo fornecido pelo usuário (ou módulo fn) que define a lógica dos ajustes de tuning, incluindo definição de modelo, espaço de pesquisa de hiperparâmetros, objetivo etc.
- Definição [protobuf](https://developers.google.com/protocol-buffers) de args de treinamento e avaliação.
- (Opcional) Definição [protobuf](https://developers.google.com/protocol-buffers) de argumentos de tuning.
- (Opcional) grafo de transformação produzido por um componente Transform upstream.
- (Opcional) Um esquema de dados criado por um componente de pipeline SchemaGen e opcionalmente alterado pelo desenvolvedor.

Com os dados, modelo e objetivo fornecidos, o Tuner ajusta os hiperparâmetros e retorna o melhor resultado.

## Instruções

Uma função de módulo de usuário `tuner_fn` com a seguinte assinatura é necessária para o Tuner:

```python
...
from keras_tuner.engine import base_tuner

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
  """Build the tuner using the KerasTuner API.
  Args:
    fn_args: Holds args as name/value pairs.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.
  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """
  ...
```

Nesta função, você define tanto o modelo como os espaços de pesquisa do hiperparâmetro e escolhe o objetivo e o algoritmo para tuning. O componente Tuner recebe o código deste módulo como entrada, ajusta os hiperparâmetros e devolve o melhor resultado.

O Trainer pode receber os hiperparâmetros de saída do Tuner como entrada e utilizá-los no seu código de módulo do usuário. A definição do pipeline é semelhante ao mostrado a seguir:

```python
...
tuner = Tuner(
    module_file=module_file,  # Contains `tuner_fn`.
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    train_args=trainer_pb2.TrainArgs(num_steps=20),
    eval_args=trainer_pb2.EvalArgs(num_steps=5))

trainer = Trainer(
    module_file=module_file,  # Contains `run_fn`.
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    # This will be passed to `run_fn`.
    hyperparameters=tuner.outputs['best_hyperparameters'],
    train_args=trainer_pb2.TrainArgs(num_steps=100),
    eval_args=trainer_pb2.EvalArgs(num_steps=5))
...
```

Você talvez não queira ajustar os hiperparâmetros sempre que treinar novamente seu modelo. Depois de usar o Tuner para determinar um bom conjunto de hiperparâmetros, você pode remover o Tuner do pipeline e usar `ImporterNode` para importar o artefato do Tuner de uma execução de treinamento anterior para alimentar o Trainer.

```python
hparams_importer = Importer(
    # This can be Tuner's output file or manually edited file. The file contains
    # text format of hyperparameters (keras_tuner.HyperParameters.get_config())
    source_uri='path/to/best_hyperparameters.txt',
    artifact_type=HyperParameters,
).with_id('import_hparams')

trainer = Trainer(
    ...
    # An alternative is directly use the tuned hyperparameters in Trainer's user
    # module code and set hyperparameters to None here.
    hyperparameters = hparams_importer.outputs['result'])
```

## Tuning no Google Cloud Platform (GCP)

Ao ser executado no Google Cloud Platform (GCP), o componente Tuner pode aproveitar dois serviços:

- [AI Platform Vizier](https://cloud.google.com/ai-platform/optimizer/docs/overview) (por meio da implementação do CloudTuner)
- [AI Platform Training](https://cloud.google.com/ai-platform/training/docs) (como gerente de rebanho para tuning distribuído)

### AI Platform Vizier como back-end da tunagem de hiperparâmetros

O [AI Platform Vizier](https://cloud.google.com/ai-platform/optimizer/docs/overview) é um serviço gerenciado que realiza otimização de caixa preta, baseado na tecnologia [Google Vizier](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/bcb15507f4b52991a0783013df4222240e942381.pdf).

O [CloudTuner](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/tuner/tuner.py) é uma implementação do [KerasTuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) que se comunica com o serviço AI Platform Vizier como back-end do estudo. Como o CloudTuner é uma subclasse de `keras_tuner.Tuner`, ele pode ser usado como um substituto imediato no módulo `tuner_fn` e executado como parte do componente TFX Tuner.

Abaixo está um trecho de código que mostra como usar o `CloudTuner`. Observe que a configuração do `CloudTuner` requer itens específicos do GCP, como `project_id` e `region`.

```python
...
from tensorflow_cloud import CloudTuner

...
def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
  """An implementation of tuner_fn that instantiates CloudTuner."""

  ...
  tuner = CloudTuner(
      _build_model,
      hyperparameters=...,
      ...
      project_id=...,       # GCP Project ID
      region=...,           # GCP Region where Vizier service is run.
  )

  ...
  return TuneFnResult(
      tuner=tuner,
      fit_kwargs={...}
  )

```

### Tuning paralelo no rebanho de workers distribuídos do Cloud AI Platform Training

O framework KerasTuner como implementação subjacente do componente Tuner tem a capacidade de conduzir pesquisas de hiperparâmetros em paralelo. Embora o componente Tuner básico não tenha a capacidade de rodar mais de um worker de pesquisa em paralelo, ao usar o componente de extensão para o Tuner do [Google Cloud AI Platform](https://github.com/tensorflow/tfx/blob/master/tfx/extensions/google_cloud_ai_platform/tuner/component.py), ele passa a ter a capacidade de executar o tuning paralelo, usando um job de treinamento do AI Platform como gerente de rebanho de workers distribuídos. [TuneArgs](https://github.com/tensorflow/tfx/blob/master/tfx/proto/tuner.proto) é a configuração dada a este componente. Esta é uma substituição imediata do componente Tuner básico.

```python
tuner = google_cloud_ai_platform.Tuner(
    ...   # Same kwargs as the above stock Tuner component.
    tune_args=proto.TuneArgs(num_parallel_trials=3),  # 3-worker parallel
    custom_config={
        # Configures Cloud AI Platform-specific configs . For for details, see
        # https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#traininginput.
        TUNING_ARGS_KEY:
            {
                'project': ...,
                'region': ...,
                # Configuration of machines for each master/worker in the flock.
                'masterConfig': ...,
                'workerConfig': ...,
                ...
            }
    })
...

```

O comportamento e a saída do componente Tuner de extensão são os mesmos do componente Tuner básico, exceto que várias pesquisas de hiperparâmetros são executadas em paralelo em diferentes máquinas de trabalho e, como resultado, os `num_trials` serão concluídos mais rapidamente. Isto é particularmente eficaz quando o algoritmo de busca é embaraçosamente paralelizável, como `RandomSearch`. No entanto, se o algoritmo de busca utilizar informações de resultados de ensaios anteriores, como faz o algoritmo Google Vizier implementado na AI Platform Vizier, uma pesquisa excessivamente paralela poderia afetar negativamente a eficácia da pesquisa.

Observação: Cada tentativa em cada busca paralela é conduzida numa única máquina no rebanho de workers, ou seja, as tentativas não aproveitam o treinamento distribuído de múltiplos workers. Se a distribuição de vários workers for desejada para cada execução, veja [`DistributingCloudTuner`](https://github.com/tensorflow/cloud/blob/b9c8752f5c53f8722dfc0b5c7e05be52e62597a8/src/python/tensorflow_cloud/tuner/tuner.py#L384-L676), em vez de `CloudTuner`.

Observação: tanto `CloudTuner` quanto o componente Tuner das extensões do Google Cloud AI Platform podem ser usados ​​juntos. Nesse caso, ele permite o tuning paralelo distribuído apoiado pelo algoritmo de pesquisa de hiperparâmetros do AI Platform Vizier. No entanto, para fazer isso, o job do Cloud AI Platform deve ter acesso ao serviço AI Platform Vizier. Consulte este [guia](https://cloud.google.com/ai-platform/training/docs/custom-service-account#custom) para configurar uma conta de serviço personalizada. Depois disso, você precisa especificar a conta de serviço personalizada para seu job de treinamento no código do pipeline. Para mais detalhes, veja [E2E CloudTuner no exemplo GCP](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_kubeflow.py) .

## Links

[Exemplo E2E](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py)

[CloudTuner E2E no Exemplo GCP](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_kubeflow.py)

[Tutorial KerasTuner](https://www.tensorflow.org/tutorials/keras/keras_tuner)

[Tutorial CloudTuner](https://github.com/GoogleCloudPlatform/ai-platform-samples/blob/master/notebooks/samples/optimizer/ai_platform_vizier_tuner.ipynb)

[Proposta](https://github.com/tensorflow/community/blob/master/rfcs/20200420-tfx-tuner-component.md)

Mais detalhes estão disponíveis na [Referência da API do Tuner](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Tuner).
