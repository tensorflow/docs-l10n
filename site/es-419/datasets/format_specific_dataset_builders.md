# Generadores de conjuntos de datos de formato específico

[TDC]

En esta guía, se documentan todos los generadores de conjuntos de datos de formatos específicos disponibles actualmente en TFDS.

Los generadores de conjuntos de datos de formato específico son subclases de [`tfds.core.GeneratorBasedBuilder`](https://www.tensorflow.org/datasets/api_docs/python/tfds/core/GeneratorBasedBuilder) que se enc.argan de la mayor parte del procesamiento de datos para un formato de datos específico.

## Conjuntos de datos que se basan ​​en `tf.data.Dataset`

Si quiere crear un conjunto de datos TFDS a partir de un conjunto de datos en formato `tf.data.Dataset` ([referencia](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)), puede usar `tfds.dataset_builders.TfDataBuilder` (consulte [los documentos de la API](https://www.tensorflow.org/datasets/api_docs/python/tfds/dataset_builders/TfDataBuilder)).

Podemos concebir dos usos típicos de esta clase:

- Crear conjuntos de datos experimentales en un entorno similar a un bloc de notas
- Definir un generador de conjuntos de datos en código

### Crear un conjunto de datos nuevo desde un bloc de notas

Imagine que estamos trabajando en un bloc de notas, carga algunos datos como `tf.data.Dataset`, aplica varias transformaciones (asignación, filtro, etc.) y ahora desea almacenar estos datos y compartirlos fácilmente con su equipo o cargarlos en otros bloc de notas. En vez de tener que definir una nueva clase de creación de conjuntos de datos, también puede crear una instancia de `tfds.dataset_builders.TfDataBuilder` y llamar a `download_and_prepare` para almacenar su conjunto de datos como un conjunto de datos TFDS.

Ya que es un conjunto de datos TFDS, puede versionarlo, usar configuraciones, tener diferentes divisiones y documentarlo para usarlo de forma más fácil en el futuro. Esto significa que también debe decirle a TFDS cuáles son las funciones de su conjunto de datos.

Aquí tienes un ejemplo ficticio de cómo se puede usar.

```python
import tensorflow as tf
import tensorflow_datasets as tfds

my_ds_train = tf.data.Dataset.from_tensor_slices({"number": [1, 2, 3]})
my_ds_test = tf.data.Dataset.from_tensor_slices({"number": [4, 5]})

# Optionally define a custom `data_dir`.
# If None, then the default data dir is used.
custom_data_dir = "/my/folder"

# Define the builder.
single_number_builder = tfds.dataset_builders.TfDataBuilder(
    name="my_dataset",
    config="single_number",
    version="1.0.0",
    data_dir=custom_data_dir,
    split_datasets={
        "train": my_ds_train,
        "test": my_ds_test,
    },
    features=tfds.features.FeaturesDict({
        "number": tfds.features.Scalar(dtype=tf.int64),
    }),
    description="My dataset with a single number.",
    release_notes={
        "1.0.0": "Initial release with numbers up to 5!",
    }
)

# Make the builder store the data as a TFDS dataset.
single_number_builder.download_and_prepare()
```

El método `download_and_prepare` iterará sobre los `tf.data.Dataset` de entrada y almacenará el conjunto de datos TFDS correspondiente en `/my/folder/my_dataset/single_number/1.0.0`, que tendrá las divisiones de entrenamiento y de prueba.

El argumento `config` es opcional y puede resultar útil si desea almacenar diferentes configuraciones en el mismo conjunto de datos.

Se puede usar el argumento `data_dir` para almacenar el conjunto de datos TFDS que se generó en una carpeta diferente, por ejemplo, en su espacio aislado propio de si no desea compartirlo con otros (todavía). Tenga en cuenta que al hacer esto, también debe pasar `data_dir` a `tfds.load`. Si no se especifica el argumento `data_dir`, se usará el directorio de datos TFDS predeterminado.

#### Cargar su conjunto de datos

Una vez que se almacena el conjunto de datos TFDS, se puede cargar desde otros scripts o sus compañeros pueden hacerlo si tienen acceso a los datos:

```python
# If no custom data dir was specified:
ds_test = tfds.load("my_dataset/single_number", split="test")

# When there are multiple versions, you can also specify the version.
ds_test = tfds.load("my_dataset/single_number:1.0.0", split="test")

# If the TFDS was stored in a custom folder, then it can be loaded as follows:
custom_data_dir = "/my/folder"
ds_test = tfds.load("my_dataset/single_number:1.0.0", split="test", data_dir=custom_data_dir)
```

#### Agregar una nueva versión o configuración

Después de más iteraraciones en su conjunto de datos, es posible que haya agregado o cambiado algunas de las transformaciones de los datos de origen. Para almacenar y compartir este conjunto de datos, puede almacenarlo fácilmente como una nueva versión.

```python
def add_one(example):
  example["number"] = example["number"] + 1
  return example

my_ds_train_v2 = my_ds_train.map(add_one)
my_ds_test_v2 = my_ds_test.map(add_one)

single_number_builder_v2 = tfds.dataset_builders.TfDataBuilder(
    name="my_dataset",
    config="single_number",
    version="1.1.0",
    data_dir=custom_data_dir,
    split_datasets={
        "train": my_ds_train_v2,
        "test": my_ds_test_v2,
    },
    features=tfds.features.FeaturesDict({
        "number": tfds.features.Scalar(dtype=tf.int64, doc="Some number"),
    }),
    description="My dataset with a single number.",
    release_notes={
        "1.1.0": "Initial release with numbers up to 6!",
        "1.0.0": "Initial release with numbers up to 5!",
    }
)

# Make the builder store the data as a TFDS dataset.
single_number_builder_v2.download_and_prepare()
```

### Definir una nueva clase de generador de conjuntos de datos

También puede definir un nuevo `DatasetBuilder` en base a esta clase.

```python
import tensorflow as tf
import tensorflow_datasets as tfds

class MyDatasetBuilder(tfds.dataset_builders.TfDataBuilder):
  def __init__(self):
    ds_train = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    ds_test = tf.data.Dataset.from_tensor_slices([4, 5])
    super().__init__(
        name="my_dataset",
        version="1.0.0",
        split_datasets={
            "train": ds_train,
            "test": ds_test,
        },
        features=tfds.features.FeaturesDict({
            "number": tfds.features.Scalar(dtype=tf.int64),
        }),
        config="single_number",
        description="My dataset with a single number.",
        release_notes={
            "1.0.0": "Initial release with numbers up to 5!",
        })
```

## CoNLL

### El formato

[CoNLL](https://aclanthology.org/W03-0419.pdf) es un formato popular que se usa para representar datos de texto anotados.

Los datos con formato CoNLL normalmente contienen un token con sus anotaciones lingüísticas por línea; dentro de una misma línea, las anotaciones se suelen separar por espacios o tabulaciones. Las líneas vacías representan los límites de las oraciones.

Considere como ejemplo la siguiente oración del conjunto de datos [conll2003](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/conll2003/conll2003.py), que sigue el formato de anotación CoNLL:

```markdown
U.N. NNP I-NP I-ORG official
NN I-NP O
Ekeus NNP I-NP I-PER
heads VBZ I-VP O
for IN I-PP O
Baghdad NNP I-NP
I-LOC . . O O
```

### `ConllDatasetBuilder`

Para agregar un conjunto de datos nuevo en base al CoNLL en el TFDS, puede basar su clase de generador de conjunto de datos en `tfds.dataset_builders.ConllDatasetBuilder`. Esta clase base contiene código común para abordar las especificidades de los conjuntos de datos CoNLL (iteración sobre el formato basado en columnas, listas precompiladas de funciones y etiquetas, ...).

`tfds.dataset_builders.ConllDatasetBuilder` implementa un `GeneratorBasedBuilder` específico de CoNLL. Consulte la siguiente clase como ejemplo mínimo de un generador de conjuntos de datos CoNLL:

```python
from tensorflow_datasets.core.dataset_builders.conll import conll_dataset_builder_utils as conll_lib
import tensorflow_datasets.public_api as tfds

class MyCoNNLDataset(tfds.dataset_builders.ConllDatasetBuilder):
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'Initial release.'}

  # conllu_lib contains a set of ready-to-use CONLL-specific configs.
  BUILDER_CONFIGS = [conll_lib.CONLL_2003_CONFIG]

  def _info(self) -> tfds.core.DatasetInfo:
    return self.create_dataset_info(
        # ...
    )

  def _split_generators(self, dl_manager):
    path = dl_manager.download_and_extract('https://data-url')

    return {'train': self._generate_examples(path=path / 'train.txt'),
            'test': self._generate_examples(path=path / 'train.txt'),
    }
```

En cuanto a los generadores de conjuntos de datos estándar, es necesario sobrescribir los métodos de clase `_info` y `_split_generators`. Según el conjunto de datos, es posible que deba actualizar también [conll_dataset_builder_utils.py](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_builders/conll/conll_dataset_builder_utils.py) para incluir las funciones y la lista de etiquetas específicas de su conjunto de datos.

El método `_generate_examples` no debería sobreescribirse más, a menos que su conjunto de datos necesite una implementación específica.

### Ejemplos

Considere [conll2003](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/conll2003/conll2003.py) como un ejemplo de un conjunto de datos que se implementa con el generador de conjuntos de datos específico de CoNLL.

### CLI

La forma más fácil de escribir un nuevo conjunto de datos en base al CoNLL es con la [CLI de TFDS](https://www.tensorflow.org/datasets/cli):

```sh
cd path/to/my/project/datasets/
tfds new my_dataset --format=conll   # Create `my_dataset/my_dataset.py` CoNLL-specific template files
```

## CoNLL-U

### El formato

[CoNLL-U](https://universaldependencies.org/format.html) es un formato popular que se usa para representar datos de texto anotados.

CoNLL-U mejora el formato CoNLL al agregar una serie de funciones, como la compatibilidad con [palabras de varios tokens](https://universaldependencies.org/u/overview/tokenization.html). Los datos que formatea CoNLL-U normalmente contienen un token con sus anotaciones lingüísticas por línea; dentro de la misma línea, las anotaciones suelen separarse por caracteres de tabulación única. Las líneas vacías representan los límites de las oraciones.

Por lo general, cada línea de palabra anotada de CoNLL-U contiene los siguientes campos, como se explica en la [documentación oficial](https://universaldependencies.org/format.html):

- ID: índice de palabras, número entero que comienza en 1 para cada nueva oración; puede ser un rango para tokens de varias palabras; puede ser un número decimal para nodos vacíos (los números decimales pueden ser inferiores a 1 pero deben ser mayores que 0).
- FORM: Forma de la palabra o símbolo de puntuación.
- LEMMA: Lema o raíz de una palabra.
- UPOS: Etiqueta universal de la parte de la oración.
- XPOS: etiqueta de la parte del discurso específica del idioma; subrayado si no está disponible.
- FEATS: Lista de funciones morfológicas del inventario de funciones universales o de una extensión específica del idioma definida; subrayado si no está disponible.
- HEAD: encabezado de la palabra actual, que es un valor de ID o cero (0).
- DEPREL: relación de dependencia universal con HEAD (raíz si HEAD = 0) o un subtipo definido específico del lenguaje de uno.
- DEPS: gráfico de dependencia mejorado en forma de lista de pares head-deprel.
- MISC: Cualquier otra anotación.

Considere como ejemplo la siguiente oración comentada con CoNLL-U de la [documentación oficial](https://universaldependencies.org/format.html):

```markdown
1-2    vámonos   _
1      vamos     ir
2      nos       nosotros
3-4    al        _
3      a         a
4      el        el
5      mar       mar
```

### `ConllUDatasetBuilder`

Para agregar un conjunto de datos nuevo en base a CoNLL-U en TFDS, puede basar su clase de generador de conjunto de datos en `tfds.dataset_builders.ConllUDatasetBuilder`. Esta clase base contiene código común para abordar las especificidades de los conjuntos de datos CoNLL-U (iteración sobre el formato basado en columnas, listas precompiladas de funciones y etiquetas, ...).

`tfds.dataset_builders.ConllUDatasetBuilder` implementa un `GeneratorBasedBuilder` específico de CoNLL-U. Consulte la siguiente clase como ejemplo mínimo de un generador de conjuntos de datos CoNLL-U:

```python
from tensorflow_datasets.core.dataset_builders.conll import conllu_dataset_builder_utils as conllu_lib
import tensorflow_datasets.public_api as tfds

class MyCoNNLUDataset(tfds.dataset_builders.ConllUDatasetBuilder):
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'Initial release.'}

  # conllu_lib contains a set of ready-to-use features.
  BUILDER_CONFIGS = [
      conllu_lib.get_universal_morphology_config(
          language='en',
          features=conllu_lib.UNIVERSAL_DEPENDENCIES_FEATURES,
      )
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    return self.create_dataset_info(
        # ...
    )

  def _split_generators(self, dl_manager):
    path = dl_manager.download_and_extract('https://data-url')

    return {
        'train':
            self._generate_examples(
                path=path / 'train.txt',
                # If necessary, add optional custom processing (see conllu_lib
                # for examples).
                # process_example_fn=...,
            )
    }
```

En cuanto a los generadores de conjuntos de datos estándar, es necesario sobrescribir los métodos de clase `_info` y `_split_generators`. Según el conjunto de datos, es posible que también deba actualizar [conllu_dataset_builder_utils.py](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_builders/conll/conllu_dataset_builder_utils.py) para incluir las funciones y la lista de etiquetas específicas de su conjunto de datos.

El método `_generate_examples` no debería sobreescribirse más, a menos que su conjunto de datos necesite una implementación específica. Tenga en cuenta que, si su conjunto de datos requiere un preprocesamiento específico (por ejemplo, si considera [funciones de dependencia universal](https://universaldependencies.org/guidelines.html) no clásicas), es posible que necesite actualizar el atributo `process_example_fn` de su función [`generate_examples`](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/dataset_builders/conll/conllu_dataset_builder.py#L192) (consulte el conjunto de datos [xtreme_pos](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/universal_dependencies/xtreme_pos.py) como ejemplo).

### Ejemplos

Considere los siguientes conjuntos de datos, que usan el generador de conjuntos de datos específico de CoNNL-U, como ejemplos:

- [universal_dependencies](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/universal_dependencies/universal_dependencies.py)
- [xtreme_pos](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/universal_dependencies/xtreme_pos.py)

### CLI

La forma más fácil de escribir un nuevo conjunto de datos en base a CoNLL-U es con [CLI de TFDS](https://www.tensorflow.org/datasets/cli):

```sh
cd path/to/my/project/datasets/
tfds new my_dataset --format=conllu   # Create `my_dataset/my_dataset.py` CoNLL-U specific template files
```
