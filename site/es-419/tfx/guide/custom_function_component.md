# Componentes personalizados de funciones de Python

La definición de componentes basada en funciones de Python facilita la creación de componentes personalizados de TFX, ya que le ahorra el esfuerzo de definir una clase de especificación de componentes, una clase de ejecutor y una clase de interfaz de componente. En este estilo de definición de componentes, se escribe una función que está anotada con sugerencias de tipo. Las sugerencias de tipo describen los artefactos de entrada, los artefactos de salida y los parámetros del componente.

Escribir un componente personalizado en este estilo es muy sencillo, como se muestra en el siguiente ejemplo.

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

A nivel interno, esto define un componente personalizado que es una subclase de [`BaseComponent`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/components/base/base_component.py) {: .external } y sus clases Spec y Executor.

Nota: La característica (componente basado en BaseBeamComponent al anotar una función con `@component(use_beam=True)`) que se describe a continuación es experimental y no hay garantías públicas de compatibilidad con versiones anteriores.

Si desea definir una subclase de [`BaseBeamComponent`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/components/base/base_beam_component.py) {: .external } de modo que pueda usar una canalización de Beam con configuración compartida de canalización de TFX, es decir, `beam_pipeline_args` al compilar la canalización ([Ejemplo de canalización taxi de Chicago](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_simple.py#L192) {: .external }) podría configurar `use_beam=True` en el decorador y agregar otro `BeamComponentParameter` con el valor predeterminado `None` en su función como en el siguiente ejemplo:

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

Si no tiene experiencia con las canalizaciones de TFX, [obtenga más información sobre los conceptos básicos de las canalizaciones de TFX](understanding_tfx_pipelines).

## Entradas, salidas y parámetros

En TFX, las entradas y salidas se rastrean como objetos Artifact que describen la ubicación y las propiedades de metadatos asociadas con los datos subyacentes; esta información se almacena en ML Metadata. Los artefactos pueden describir tipos de datos complejos o tipos de datos simples, como: int, float, bytes o cadenas Unicode.

Un parámetro es un argumento (int, float, bytes o cadena Unicode) de un componente conocido en el momento de la construcción de la canalización. Los parámetros son útiles para especificar argumentos e hiperparámetros como el recuento de iteraciones de entrenamiento, la tasa de abandonos y otras configuraciones de su componente. Los parámetros se almacenan como propiedades de las ejecuciones de componentes cuando se rastrean en ML Metadata.

Nota: Actualmente, los valores de salida de tipo de datos simples no se pueden usar como parámetros ya que no se conocen en el momento de la ejecución. De manera similar, los valores de entrada de tipos de datos simples actualmente no pueden tomar valores concretos conocidos en el momento de la construcción de la canalización. Es posible que eliminemos esta restricción en una próxima versión de TFX.

## Definición

Para crear un componente personalizado, escriba una función que implemente su lógica personalizada y decórela con el [decorador `@component`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/decorators.py) {: .external } del módulo `tfx.dsl.component.experimental.decorators`. Para definir el esquema de entrada y salida de su componente, use las anotaciones del módulo `tfx.dsl.component.experimental.annotations` para anotar los argumentos de su función y el valor de retorno:

- Para cada **entrada de artefacto**, aplique la anotación de sugerencia de tipo `InputArtifact[ArtifactType]`. Reemplace `ArtifactType` con el tipo de artefacto, que es una subclase de `tfx.types.Artifact`. Estas entradas pueden ser argumentos opcionales.

- Para cada **artefacto de salida**, aplique la anotación de sugerencia de tipo `OutputArtifact[ArtifactType]`. Reemplace `ArtifactType` con el tipo de artefacto, que es una subclase de `tfx.types.Artifact`. Los artefactos de salida de los componentes deben pasarse como argumentos de entrada de la función, de modo que su componente pueda escribir salidas en una ubicación administrada por el sistema y establecer las propiedades de metadatos de los artefactos apropiadas. Este argumento puede ser opcional o puede definirse con un valor predeterminado.

- Para cada **parámetro**, use la anotación de sugerencia de tipo `Parameter[T]`. Reemplace `T` con el tipo de parámetro. Actualmente solo se admiten tipos primitivos de Python: `bool`, `int`, `float`, `str` o `bytes`.

- Para la **canalización de Beam**, utilice la anotación de sugerencia de tipo `BeamComponentParameter[beam.Pipeline]`. Establezca el valor predeterminado en `None`. El valor `None` se reemplazará una canalización de Beam basada en una instancia que se crea con `_make_beam_pipeline()` de [`BaseBeamExecutor`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/components/base/base_beam_executor.py) {: .external }

- Para cada **entrada de tipo de datos simples** (`int`, `float`, `str` o `bytes`) no conocida en el momento de la construcción de la canalización, use la sugerencia de tipo `T`. Tenga en cuenta que, en la versión TFX 0.22, no se pueden pasar valores concretos en el momento de la construcción de la canalización para este tipo de entrada (en lugar de esto, use la anotación `Parameter`, como se describe en la sección anterior). Este argumento puede ser opcional o puede definirse con un valor predeterminado. Si su componente tiene salidas de tipo de datos simples (`int`, `float`, `str` o `bytes`), puede devolver estas salidas por medio de `TypedDict` como anotación de tipo de retorno y devolver un objeto dict apropiado.

En el cuerpo de su función, los artefactos de entrada y salida se pasan como objetos `tfx.types.Artifact`; puede inspeccionar su `.uri` para obtener la ubicación gestionada por el sistema y leer/configurar cualquier propiedad. Los parámetros de entrada y las entradas de tipos de datos simples se pasan como objetos del tipo especificado. Las salidas de tipo de datos simples deben devolverse como un diccionario, donde las claves son los nombres de salida correspondientes y los valores son los valores de retorno deseados.

El componente de función completo puede verse así:

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

En el ejemplo anterior se define `MyTrainerComponent` como un componente personalizado basado en funciones de Python. Este componente consume un artefacto `examples` como entrada y produce un artefacto `model` como salida. El componente utiliza `artifact_instance.uri` para leer o escribir el artefacto en su ubicación gestionada por el sistema. El componente toma un parámetro de entrada `num_iterations` y un valor de tipo de datos simple `dropout_hyperparameter`, y el componente genera métricas `loss` y `accuracy` como valores de salida de tipo de datos simples. Luego, el componente `Pusher` usa el artefacto `model` como salida.
