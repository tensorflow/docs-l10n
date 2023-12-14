# Cómo compilar componentes totalmente personalizados

En esta guía se describe cómo usar la API de TFX para compilar un componente totalmente personalizado. Los componentes totalmente personalizados le permiten compilar componentes a través de la definición de la especificación del componente, el ejecutor y las clases de interfaz del componente. Este enfoque le permite reutilizar y ampliar un componente estándar para adaptarlo a sus necesidades.

Si no tiene experiencia con las canalizaciones de TFX, [obtenga más información sobre los conceptos básicos de las canalizaciones de TFX](understanding_tfx_pipelines).

## Ejecutor personalizado o componente personalizado

Si solo se necesita una lógica de procesamiento personalizada mientras las entradas, salidas y propiedades de ejecución del componente son idénticas a las de un componente existente, basta con un ejecutor personalizado. Se necesita un componente totalmente personalizado cuando alguna de las entradas, salidas o propiedades de ejecución es diferente de cualquier componente de TFX existente.

## ¿Cómo crear un componente personalizado?

Para desarrollar un componente totalmente personalizado se requiere lo siguiente:

- Un conjunto definido de especificaciones de artefactos de entrada y salida para el nuevo componente. Sobre todo, los tipos de artefactos de entrada deben ser coherentes con los tipos de artefactos de salida de los componentes que producen los artefactos y los tipos de artefactos de salida deben ser coherentes con los tipos de artefactos de entrada de los componentes que consumen los artefactos, si los hubiera.
- Los parámetros de ejecución que no son artefactos necesarios para el nuevo componente.

### ComponentSpec

La clase `ComponentSpec` define el contrato del componente mediante la definición de los artefactos de entrada y salida a un componente, así como los parámetros que se utilizan para la ejecución del componente. Consta de tres partes:

- *ENTRADAS*: un diccionario de parámetros escritos para los artefactos de entrada que se pasan al ejecutor del componente. Normalmente, los artefactos de entrada son las salidas de los componentes ascendentes y, por lo tanto, comparten el mismo tipo.
- *SALIDAS*: un diccionario de parámetros escritos para los artefactos de salida que produce el componente.
- *PARÁMETROS*: un diccionario de elementos [ExecutionParameter](https://github.com/tensorflow/tfx/blob/54aa6fbec6bffafa8352fe51b11251b1e44a2bf1/tfx/types/component_spec.py#L274) adicionales que se pasarán al ejecutor del componente. Estos son parámetros que no son artefactos y que queremos definir de manera flexible en el DSL de la canalización y pasar a ejecución.

A continuación se muestra un ejemplo de ComponentSpec.

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

### Ejecutor

A continuación, escriba el código ejecutor para el nuevo componente. Básicamente, se debe crear una nueva subclase de `base_executor.BaseExecutor` con su función `Do` anulada. En la función `Do`, los argumentos `input_dict`, `output_dict` y `exec_properties` que se pasan en el mapa a `INPUTS`, `OUTPUTS` y `PARAMETERS` que se definen en ComponentSpec respectivamente. Para `exec_properties`, el valor se puede extraer directamente mediante una búsqueda en el diccionario. Para los artefactos en `input_dict` y `output_dict`, hay funciones convenientes disponibles en la clase [artifact_utils](https://github.com/tensorflow/tfx/blob/41823f91dbdcb93195225a538968a80ba4bb1f55/tfx/types/artifact_utils.py) que se pueden usar para extraer la instancia del artefacto o el URI del artefacto.

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

#### Pruebas unitarias de un ejecutor personalizado

Se pueden crear pruebas unitarias para el ejecutor personalizado similares a [esta](https://github.com/tensorflow/tfx/blob/r0.15/tfx/components/transform/executor_test.py).

### Interfaz de componentes

Ahora que ya completamos la parte más compleja, el siguiente paso consiste en ensamblar estas piezas para formar una interfaz de componente, que permita utilizar el componente en una canalización. Este proceso consta de varios pasos:

- Haga que la interfaz de componente sea una subclase de `base_component.BaseComponent`
- Asigne una variable de clase `SPEC_CLASS` con la clase `ComponentSpec` que se definió anteriormente
- Asigne una variable de clase `EXECUTOR_SPEC` con la clase Ejecutor que se definió anteriormente
- Defina la función constructora `__init__()` con los argumentos de la función para construir una instancia de la clase ComponentSpec e invocar la superfunción con ese valor, junto con un nombre opcional.

Cuando se cree una instancia del componente, se invocará la lógica de verificación de tipos en la clase `base_component.BaseComponent` para garantizar que los argumentos que se pasaron sean compatibles con la información de tipo definida en la clase `ComponentSpec`.

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

### Ensamblado en una canalización de TFX

Como último paso, debe conectar el nuevo componente personalizado a una canalización de TFX. Además de agregar una instancia del nuevo componente, también debe hacer lo siguiente:

- Conectar correctamente los componentes ascendentes y descendentes del nuevo componente. Para esto, debe hacer referencia a las salidas del componente ascendente en el nuevo componente y hacer referencia a las salidas del nuevo componente en los componentes descendentes.
- Agregar la nueva instancia del componente a la lista de componentes al construir la canalización.

En el siguiente ejemplo se destacan los cambios antes mencionados. El ejemplo completo se puede encontrar en el [repositorio GitHub de TFX](https://github.com/tensorflow/tfx/tree/master/tfx/examples/custom_components/hello_world).

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

## Implemente un componente totalmente personalizado

Más allá de los cambios de código, es necesario que pueda acceder a todas las partes recién agregadas (`ComponentSpec`, `Executor`, interfaz de componente) en el entorno de ejecución de la canalización para poder ejecutar la canalización correctamente.
