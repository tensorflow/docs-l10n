# Agregar una nueva colección de conjuntos de datos

Siga esta guía para crear una nueva colección de conjuntos de datos (ya sea en TFDS o en su propio repositorio).

## Descripción general

Para agregar una nueva colección de conjuntos de datos `my_collection` a TFDS, los usuarios deben generar una carpeta `my_collection` que contenga los siguientes archivos:

```sh
my_collection/
  __init__.py
  my_collection.py # Dataset collection definition
  my_collection_test.py # (Optional) test
  description.md # (Optional) collection description (if not included in my_collection.py)
  citations.md # (Optional) collection citations (if not included in my_collection.py)
```

Como convención, se deben agregar las nuevas colecciones de conjuntos de datos a la carpeta `tensorflow_datasets/dataset_collections/` en el repositorio TFDS.

## Escriba su colección de conjuntos de datos

Todas las colecciones de conjuntos de datos son subclases implementadas de [`tfds.core.dataset_collection_builder.DatasetCollection`](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_collection_builder.py).

A continuación, se muestra un ejemplo mínimo de un generador de colecciones de conjuntos de datos, definido en el archivo `my_collection.py`:

```python
import collections
from typing import Mapping
from tensorflow_datasets.core import dataset_collection_builder
from tensorflow_datasets.core import naming

class MyCollection(dataset_collection_builder.DatasetCollection):
  """Dataset collection builder my_dataset_collection."""

  @property
  def info(self) -> dataset_collection_builder.DatasetCollectionInfo:
    return dataset_collection_builder.DatasetCollectionInfo.from_cls(
        dataset_collection_class=self.__class__,
        description="my_dataset_collection description.",
        release_notes={
            "1.0.0": "Initial release",
        },
    )

  @property
  def datasets(
      self,
  ) -> Mapping[str, Mapping[str, naming.DatasetReference]]:
    return collections.OrderedDict({
        "1.0.0":
            naming.references_for({
                "dataset_1": "natural_questions/default:0.0.2",
                "dataset_2": "media_sum:1.0.0",
            }),
        "1.1.0":
            naming.references_for({
                "dataset_1": "natural_questions/longt5:0.1.0",
                "dataset_2": "media_sum:1.0.0",
                "dataset_3": "squad:3.0.0"
            })
    })
```

En las siguientes secciones, se describen los 2 métodos abstractos para sobrescribir.

### `info`: metadatos de recopilación de conjuntos de datos

El método `info` devuelve [`dataset_collection_builder.DatasetCollectionInfo`](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/core/dataset_collection_builder.py#L66) que contiene los metadatos de la colección.

La información de recopilación del conjunto de datos contiene cuatro campos:

- nombre: el nombre de la colección del conjunto de datos.
- descripción: una descripción con formato markdown de la colección del conjunto de datos. Hay dos formas de definir la descripción de una colección de conjuntos de datos: (1) En una cadena de texto (de varias líneas) directamente en el archivo `my_collection.py` de la colección, similar a como ya se hace para los conjuntos de datos TFDS; (2) en un archivo `description.md`, que debe colocarse en la carpeta de recopilación del conjunto de datos.
- release_notes: una asignación de la versión de la colección del conjunto de datos a las notas de la versión correspondientes.
- cita: una (lista de) cita(s) BibTeX opcionales para la colección del conjunto de datos. Hay dos formas de definir la cita de una colección de conjuntos de datos: (1) Como una cadena de texto (de varias líneas) directamente en el archivo `my_collection.py` de la colección, de manera similar a como ya se hace para los conjuntos de datos TFDS; (2) en un archivo `citations.bib`, que debe colocarse en la carpeta de recopilación del conjunto de datos.

### `datasets`: define los conjuntos de datos en la colección

El método de `datasets` devuelve los conjuntos de datos TFDS de la colección.

Se define como un diccionario de versiones, que describen la evolución de la recopilación de conjuntos de datos.

Para cada versión, los conjuntos de datos TFDS incluidos se almacenan como un diccionario desde los nombres de los conjuntos de datos en [`naming.DatasetReference`](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/core/naming.py#L187). Por ejemplo:

```python
class MyCollection(dataset_collection_builder.DatasetCollection):
  ...
  @property
  def datasets(self):
    return {
        "1.0.0": {
            "yes_no":
                naming.DatasetReference(
                    dataset_name="yes_no", version="1.0.0"),
            "sst2":
                naming.DatasetReference(
                    dataset_name="glue", config="sst2", version="2.0.0"),
            "assin2":
                naming.DatasetReference(
                    dataset_name="assin2", version="1.0.0"),
        },
        ...
    }
```

El método [`naming.references_for`](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/core/naming.py#L257) proporciona una forma más compacta de expresar lo mismo que el anterior:

```python
class MyCollection(dataset_collection_builder.DatasetCollection):
  ...
  @property
  def datasets(self):
    return {
        "1.0.0":
            naming.references_for({
                "yes_no": "yes_no:1.0.0",
                "sst2": "glue/sst:2.0.0",
                "assin2": "assin2:1.0.0",
            }),
        ...
    }
```

## Prueba unitaria de su colección de conjuntos de datos

[DatasetCollectionTestBase](https://github.com/tensorflow/datasets/blob/4854e55ddf5fb68c63ddbd502ad0ef4ec6e08b40/tensorflow_datasets/testing/dataset_collection_builder_testing.py#L28) es una clase de prueba base para colecciones de conjuntos de datos. Proporciona una serie de comprobaciones simples para garantizar que la recopilación del conjunto de datos esté registrada correctamente y que sus conjuntos de datos existan en TFDS.

El único atributo de clase que se debe establecer es `DATASET_COLLECTION_CLASS`, que especifica el objeto de la clase de la colección de conjuntos de datos que se va a probar.

Además, los usuarios pueden configurar los siguientes atributos de clase:

- `VERSION`: La versión de la colección del conjunto de datos que se usa para ejecutar la prueba (el valor predeterminado es la última versión).
- `DATASETS_TO_TEST`: Lista que contiene los conjuntos de datos para probar la existencia en TFDS (el valor predeterminado es todos los conjuntos de datos de la colección).
- `CHECK_DATASETS_VERSION`: ya sea para verificar la existencia de los conjuntos de datos versionados en la colección de conjuntos de datos o sus versiones predeterminadas (el valor predeterminado es verdadero).

La prueba válida más simple para una recopilación de conjuntos de datos sería:

```python
from tensorflow_datasets.testing.dataset_collection_builder_testing import DatasetCollectionTestBase
from . import my_collection

class TestMyCollection(DatasetCollectionTestBase):
  DATASET_COLLECTION_CLASS = my_collection.MyCollection
```

Ejecute el siguiente comando para probar la colección del conjunto de datos.

```sh
python my_dataset_test.py
```

## Comentarios

Siempre buscamos mejorar el flujo de trabajo de la creación de conjuntos de datos, pero solo podemos hacerlo si conocemos los problemas que hay. ¿Qué problemas o errores encontró al crear la colección del conjunto de datos? ¿Hubo alguna parte que resultó confusa o no funcionó la primera vez?

Deje sus comentarios en [GitHub](https://github.com/tensorflow/datasets/issues).
