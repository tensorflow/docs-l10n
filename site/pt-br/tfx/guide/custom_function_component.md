# Componentes de função Python personalizados

A definição de componentes baseados em funções Python facilita a criação de componentes personalizados do TFX, economizando o esforço de definir uma classe de especificação do componente, uma classe de executor e uma classe de interface do componente. Neste estilo de definição de componentes, você escreve uma função anotada com dicas de tipo. As dicas de tipo descrevem os artefatos de entrada, os artefatos de saída e os parâmetros do seu componente.

Escrever seu componente personalizado neste estilo é muito simples, como no exemplo a seguir.

```python
class MyOutput(TypedDict):
  accuracy: float

@component
def MyValidationComponent(
    model: InputArtifact[Model],
    blessing: OutputArtifact[Model],
    accuracy_threshold: Parameter[int] = 10,
) -> MyOutput:
  '''My simple custom model validation component.'''

  accuracy = evaluate_model(model)
  if accuracy >= accuracy_threshold:
    write_output_blessing(blessing)

  return {
    'accuracy': accuracy
  }
```

Nos bastidores, isto define um componente personalizado que é uma subclasse de [`BaseComponent`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/components/base/base_component.py){: .external } e suas classes Spec e Executor.

Observação: o recurso (componente baseado em BaseBeamComponent anotando uma função com `@component(use_beam=True)`) descrito abaixo é experimental e não há garantias públicas de compatibilidade com versões anteriores.

Se você quer definir uma subclasse de [`BaseBeamComponent`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/components/base/base_beam_component.py){: .external } de modo que possa usar um pipeline Beam com configuração compartilhada do pipeline TFX, ou seja, `beam_pipeline_args` ao compilar o pipeline ([Chicago Taxi Pipeline Exemplo](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_simple.py#L192){: .external }), você poderia definir `use_beam=True` no decorador e adicionar outro `BeamComponentParameter` com valor padrão `None` em sua função como no exemplo a seguir:

```python
@component(use_beam=True)
def MyDataProcessor(
    examples: InputArtifact[Example],
    processed_examples: OutputArtifact[Example],
    beam_pipeline: BeamComponentParameter[beam.Pipeline] = None,
    ) -> None:
  '''My simple custom model validation component.'''

  with beam_pipeline as p:
    # data pipeline definition with beam_pipeline begins
    ...
    # data pipeline definition with beam_pipeline ends
```

Se você é novato em pipelines TFX, [aprenda mais sobre os principais conceitos dos pipelines TFX](understanding_tfx_pipelines) antes de continuar.

## Entradas, saídas e parâmetros

No TFX, as entradas e saídas são rastreadas como objetos de artefato que descrevem a localização e as propriedades de metadados associadas aos dados subjacentes; essas informações são armazenadas em metadados de aprendizado de máquina (ML Metadata). Os artefatos podem descrever tipos de dados complexos ou tipos de dados simples, como: int, float, bytes ou strings unicode.

Um parâmetro é um argumento (int, float, bytes ou string unicode) para um componente conhecido no momento da construção do pipeline. Os parâmetros são úteis para especificar argumentos e hiperparâmetros, como contagem de iterações de treinamento, taxa de dropout e outras configurações para seu componente. Os parâmetros são armazenados como propriedades de execuções de componentes quando rastreados no ML Metadata.

Observação: Atualmente, os valores de tipo de dados simples de saída não podem ser usados ​​como parâmetros, pois não são conhecidos em tempo de execução. Da mesma forma, os valores de tipo de dados simples de entrada atualmente não podem assumir valores concretos conhecidos no momento da construção do pipeline. Poderemos remover esta restrição numa versão futura do TFX.

## Definição

Para criar um componente personalizado, escreva uma função que implemente sua lógica personalizada e decore-a com o [`@component` decorator](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/decorators.py) {: .external } do módulo `tfx.dsl.component.experimental.decorators`. Para definir o esquema de entrada e saída do seu componente, anote os argumentos da sua função e o valor de retorno usando anotações do módulo `tfx.dsl.component.experimental.annotations`:

- Para cada **entrada de artefato**, aplique a anotação de dica de tipo `InputArtifact[ArtifactType]`. Substitua `ArtifactType` pelo tipo do artefato, que é uma subclasse de `tfx.types.Artifact`. Essas entradas podem ser argumentos opcionais.

- Para cada **artefato de saída**, aplique a anotação de dica de tipo `OutputArtifact[ArtifactType]`. Substitua `ArtifactType` pelo tipo do artefato, que é uma subclasse de `tfx.types.Artifact`. Os artefatos de saída do componente devem ser passados ​​como argumentos de entrada da função, para que seu componente possa gravar saídas em um local gerenciado pelo sistema e configurar propriedades apropriadas de metadados do artefato. Este argumento pode ser opcional ou pode ser definido com um valor padrão.

- Para cada **parâmetro**, use a anotação de dica de tipo `Parameter[T]`. Substitua `T` pelo tipo do parâmetro. Atualmente, oferecemos suporte apenas a tipos python primitivos: `bool`, `int`, `float`, `str` ou `bytes`.

- Para o **beam pipeline**, use a anotação de dica de tipo `BeamComponentParameter[beam.Pipeline]`. Defina o valor padrão como `None`. O valor `None` será substituído por um pipeline beam instanciado criado por `_make_beam_pipeline()` de [`BaseBeamExecutor`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/components/base/base_beam_executor.py){: .external }

- Para cada **entrada de tipo de dados simples** (`int`, `float`, `str` ou `bytes`) não conhecida no momento da construção do pipeline, use a dica de tipo `T`. Observe que na versão 0.22 do TFX, valores concretos não podem ser passados no momento da construção do pipeline para esse tipo de entrada (em vez disso, use a anotação `Parameter`, conforme descrito na seção anterior). Este argumento pode ser opcional ou pode ser definido com um valor padrão. Se o seu componente tiver saídas de tipo de dados simples (`int`, `float`, `str` ou `bytes`), você poderá retornar essas saídas usando um `TypedDict` como uma anotação de tipo de retorno e retornando um objeto dict apropriado.

No corpo da sua função, os artefatos de entrada e saída são passados ​​como objetos `tfx.types.Artifact`; você pode inspecionar sua `.uri` para obter sua localização gerenciada pelo sistema e ler/definir quaisquer propriedades. Parâmetros de entrada e entradas de tipo de dados simples são passados ​​como objetos do tipo especificado. As saídas de tipo de dados simples devem ser retornadas como um dicionário, onde as chaves são os nomes de saída apropriados e os valores são os valores de retorno desejados.

O componente de função concluído pode ser similar ao código a seguir:

```python
from typing import TypedDict
import tfx.v1 as tfx
from tfx.dsl.component.experimental.decorators import component

class MyOutput(TypedDict):
  loss: float
  accuracy: float

@component
def MyTrainerComponent(
    training_data: tfx.dsl.components.InputArtifact[tfx.types.standard_artifacts.Examples],
    model: tfx.dsl.components.OutputArtifact[tfx.types.standard_artifacts.Model],
    dropout_hyperparameter: float,
    num_iterations: tfx.dsl.components.Parameter[int] = 10
) -> MyOutput:
  '''My simple trainer component.'''

  records = read_examples(training_data.uri)
  model_obj = train_model(records, num_iterations, dropout_hyperparameter)
  model_obj.write_to(model.uri)

  return {
    'loss': model_obj.loss,
    'accuracy': model_obj.accuracy
  }

# Example usage in a pipeline graph definition:
# ...
trainer = MyTrainerComponent(
    examples=example_gen.outputs['examples'],
    dropout_hyperparameter=other_component.outputs['dropout'],
    num_iterations=1000)
pusher = Pusher(model=trainer.outputs['model'])
# ...
```

O exemplo anterior define `MyTrainerComponent` como um componente personalizado baseado em função Python. Este componente consome um artefato de `examples` como entrada e produz um artefato `model` como saída. O componente usa o `artifact_instance.uri` para ler ou gravar o artefato em seu local gerenciado pelo sistema. O componente usa um parâmetro de entrada `num_iterations` e um valor de tipo de dados simples `dropout_hyperparameter`, e o componente gera métricas `loss` e `accuracy` como valores de saída de tipo de dados simples. O artefato `model` de saída é então usado pelo componente `Pusher`.
