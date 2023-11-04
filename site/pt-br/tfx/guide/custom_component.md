# Construindo componentes totalmente personalizados

Este guia descreve como usar a API TFX para criar um componente totalmente personalizado. Componentes totalmente personalizados permitem construir componentes definindo a especificação do componente, o executor e as classes de interface do componente. Essa abordagem permite reutilizar e estender um componente padrão para atender às suas necessidades.

Se você é novato em pipelines TFX, [aprenda mais sobre os principais conceitos dos pipelines TFX](understanding_tfx_pipelines) antes de continuar.

## Executor personalizado ou componente personalizado

Se apenas a lógica de processamento customizada for necessária enquanto as entradas, saídas e propriedades de execução do componente forem as mesmas de um componente existente, um executor customizado será suficiente. Um componente totalmente personalizado é necessário quando qualquer uma das entradas, saídas ou propriedades de execução é diferente de qualquer componente TFX existente.

## Como criar um componente personalizado?

O desenvolvimento de um componente totalmente personalizado requer:

- Um conjunto definido de especificações de artefatos de entrada e saída para o novo componente. Mais importante é que os tipos dos artefatos de entrada sejam consistentes com os tipos de artefatos de saída dos componentes que produzem os artefatos (componentes upstream), e que os tipos dos artefatos de saída sejam consistentes com os tipos de artefatos de entrada dos componentes que consomem os artefatos (componentes downstream), se houver.
- Os parâmetros de execução que não são artefatos e que são necessários para o novo componente.

### ComponentSpec

A classe `ComponentSpec` define o contrato do componente definindo os artefatos de entrada e saída para um componente, bem como os parâmetros que são usados ​​para a execução do componente. Tem três partes:

- *INPUTS*: Um dicionário de parâmetros digitados para os artefatos de entrada que são passados ​​para o executor do componente. Normalmente, os artefatos de entrada são as saídas dos componentes upstream e, portanto, compartilham o mesmo tipo.
- *OUTPUTS*: Um dicionário de parâmetros digitados para os artefatos de saída que o componente produz.
- *PARAMETERS*: Um dicionário de itens [ExecutionParameter](https://github.com/tensorflow/tfx/blob/54aa6fbec6bffafa8352fe51b11251b1e44a2bf1/tfx/types/component_spec.py#L274) adicionais que serão passados ​​para o executor do componente. Esses são parâmetros que não são artefatos e que queremos definir de forma flexível no pipeline DSL e passar para execução.

Aqui está um exemplo do ComponentSpec:

```python
class HelloComponentSpec(types.ComponentSpec):
  """ComponentSpec for Custom TFX Hello World Component."""

  PARAMETERS = {
      # These are parameters that will be passed in the call to
      # create an instance of this component.
      'name': ExecutionParameter(type=Text),
  }
  INPUTS = {
      # This will be a dictionary with input artifacts, including URIs
      'input_data': ChannelParameter(type=standard_artifacts.Examples),
  }
  OUTPUTS = {
      # This will be a dictionary which this component will populate
      'output_data': ChannelParameter(type=standard_artifacts.Examples),
  }
```

### Executor

Em seguida, escreva o código executor do novo componente. Basicamente, uma nova subclasse de `base_executor.BaseExecutor` precisa ser criada com sua função `Do` sobreposta. Na função `Do`, os argumentos `input_dict`, `output_dict` e `exec_properties` que são passados ​​​​no mapa para `INPUTS`, `OUTPUTS` e `PARAMETERS` que são definidos em ComponentSpec respectivamente. Para `exec_properties`, o valor pode ser obtido diretamente através de uma pesquisa no dicionário. Para artefatos em `input_dict` e `output_dict`, existem funções convenientes disponíveis na classe [artefato_utils](https://github.com/tensorflow/tfx/blob/41823f91dbdcb93195225a538968a80ba4bb1f55/tfx/types/artifact_utils.py) que podem ser usadas para buscar a instância ou URI do artefato.

```python
class Executor(base_executor.BaseExecutor):
  """Executor for HelloComponent."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    ...

    split_to_instance = {}
    for artifact in input_dict['input_data']:
      for split in json.loads(artifact.split_names):
        uri = artifact_utils.get_split_uri([artifact], split)
        split_to_instance[split] = uri

    for split, instance in split_to_instance.items():
      input_dir = instance
      output_dir = artifact_utils.get_split_uri(
          output_dict['output_data'], split)
      for filename in tf.io.gfile.listdir(input_dir):
        input_uri = os.path.join(input_dir, filename)
        output_uri = os.path.join(output_dir, filename)
        io_utils.copy_file(src=input_uri, dst=output_uri, overwrite=True)
```

#### Teste de unidade de um executor personalizado

Testes de unidade para o executor personalizado podem ser criados de forma semelhante a [este](https://github.com/tensorflow/tfx/blob/r0.15/tfx/components/transform/executor_test.py).

### Interface do componente

Agora que a parte mais complexa está concluída, o próximo passo é montar essas peças numa interface de componente, para permitir que o componente seja usado em um pipeline. São vários passos:

- Torne a interface do componente uma subclasse de `base_component.BaseComponent`
- Atribua uma variável de classe `SPEC_CLASS` com a classe `ComponentSpec` que foi definida anteriormente
- Atribua uma variável de classe `EXECUTOR_SPEC` com a classe Executor que foi definida anteriormente
- Defina a função construtora `__init__()` usando os argumentos da função para construir uma instância da classe ComponentSpec e invocar a superfunção com esse valor, junto com um nome opcional

Quando uma instância do componente é criada, a lógica de verificação de tipo na classe `base_component.BaseComponent` será invocada para garantir que os argumentos que foram passados ​​sejam compatíveis com as informações de tipo definidas na classe `ComponentSpec`.

```python
from tfx.types import standard_artifacts
from hello_component import executor

class HelloComponent(base_component.BaseComponent):
  """Custom TFX Hello World Component."""

  SPEC_CLASS = HelloComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               input_data: types.Channel = None,
               output_data: types.Channel = None,
               name: Optional[Text] = None):
    if not output_data:
      examples_artifact = standard_artifacts.Examples()
      examples_artifact.split_names = input_data.get()[0].split_names
      output_data = channel_utils.as_channel([examples_artifact])

    spec = HelloComponentSpec(input_data=input_data,
                              output_data=output_data, name=name)
    super(HelloComponent, self).__init__(spec=spec)
```

### Instalação do componente num pipeline TFX

A última etapa é conectar o novo componente personalizado a um pipeline do TFX. Além de adicionar uma instância do novo componente, também são necessários os seguintes itens:

- Conecte corretamente os componentes upstream e downstream do novo componente. Isto é feito referenciando as saídas do componente upstream no novo componente e referenciando as saídas do novo componente nos componentes downstream.
- Adicione a nova instância do componente à lista de componentes ao construir o pipeline.

O exemplo abaixo destaca as alterações mencionadas. Um exemplo completo pode ser encontrado no [repositório TFX GitHub](https://github.com/tensorflow/tfx/tree/master/tfx/examples/custom_components/hello_world).

```python
def _create_pipeline():
  ...
  example_gen = CsvExampleGen(input_base=examples)
  hello = component.HelloComponent(
      input_data=example_gen.outputs['examples'], name='HelloWorld')
  statistics_gen = StatisticsGen(examples=hello.outputs['output_data'])
  ...
  return pipeline.Pipeline(
      ...
      components=[example_gen, hello, statistics_gen, ...],
      ...
  )
```

## Implantação de um componente totalmente personalizado

Além das alterações de código, todas as partes recém-adicionadas (`ComponentSpec` , `Executor`, interface do componente) precisam estar acessíveis no ambiente de execução do pipeline para executar o pipeline corretamente.
