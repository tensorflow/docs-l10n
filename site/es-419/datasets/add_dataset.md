# Escribir conjuntos de datos personalizados

Siga esta guía para crear un nuevo conjunto de datos (ya sea en TFDS o en su propio repositorio).

Consulte nuestra [lista de conjuntos de datos](catalog/overview.md) para ver si el conjunto de datos que desea ya está presente.

## TL;DR

La forma más simple de escribir un nuevo conjunto de datos es usar la [interfaz de la línea de comandos de TFDS](https://www.tensorflow.org/datasets/cli):

```sh
cd path/to/my/project/datasets/
tfds new my_dataset  # Create `my_dataset/my_dataset.py` template files
# [...] Manually modify `my_dataset/my_dataset_dataset_builder.py` to implement your dataset.
cd my_dataset/
tfds build  # Download and prepare the dataset to `~/tensorflow_datasets/`
```

Para usar el nuevo conjunto de datos con `tfds.load('my_dataset')`:

- `tfds.load` detectará y cargará automáticamente el conjunto de datos generado en `~/tensorflow_datasets/my_dataset/` (por ejemplo, mediante `tfds build`).
- De forma alternativa, puede `import my.project.datasets.my_dataset` para registrar su conjunto de datos:

```python
import my.project.datasets.my_dataset  # Register `my_dataset`

ds = tfds.load('my_dataset')  # `my_dataset` registered
```

## Descripción general

Los conjuntos de datos se distribuyen en todo tipo de formatos y en todo tipo de lugares, y no siempre se almacenan en un formato que esté listo para alimentar un proceso de aprendizaje automático. Introduzca TFDS.

TFDS procesa esos conjuntos de datos en un formato estándar (datos externos -&gt; archivos serializados), que luego se pueden cargar como canal de aprendizaje automático (archivos serializados -&gt; `tf.data.Dataset`). La serialización se realiza solo una vez. El acceso posterior leerá directamente esos archivos preprocesados.

La mayor parte del preprocesamiento se realiza automáticamente. Cada conjunto de datos implementa una subclase de `tfds.core.DatasetBuilder`, que especifica:

- De dónde provienen los datos (es decir, sus URL);
- Cómo se ve el conjunto de datos (es decir, sus características);
- Cómo se deben dividir los datos (por ejemplo, `TRAIN` y `TEST`);
- y los ejemplos individuales en el conjunto de datos.

## Escriba su conjunto de datos

### Plantilla predeterminada: `tfds new`

Use la [interfaz de la línea de comandos de TFDS](https://www.tensorflow.org/datasets/cli) para generar los archivos de plantilla de Python necesarios.

```sh
cd path/to/project/datasets/  # Or use `--dir=path/to/project/datasets/` below
tfds new my_dataset
```

Este comando generará una nueva carpeta `my_dataset/` con la siguiente estructura:

```sh
my_dataset/
    __init__.py
    README.md # Markdown description of the dataset.
    CITATIONS.bib # Bibtex citation for the dataset.
    TAGS.txt # List of tags describing the dataset.
    my_dataset_dataset_builder.py # Dataset definition
    my_dataset_dataset_builder_test.py # Test
    dummy_data/ # (optional) Fake data (used for testing)
    checksum.tsv # (optional) URL checksums (see `checksums` section).
```

Busque `TODO(my_dataset)` en la carpeta y modifíquelo como corresponde.

### Ejemplo de conjunto de datos

Todos los conjuntos de datos son subclases implementadas de `tfds.core.DatasetBuilder`, que se encarga de la mayor parte del texto estándar. Admite:

- Conjuntos de datos pequeños/medianos que se pueden generar en una sola máquina (este tutorial).
- Conjuntos de datos muy grandes que requieren generación distribuida (con [Apache Beam](https://beam.apache.org/), consulte nuestra [guía de conjuntos de datos gigantes](https://www.tensorflow.org/datasets/beam_datasets#implementing_a_beam_dataset))

A continuación se muestra un ejemplo mínimo de un generador de conjuntos de datos que se basa en `tfds.core.GeneratorBasedBuilder`:

```python
class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata (homepage, citation,...)."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(256, 256, 3)),
            'label': tfds.features.ClassLabel(
                names=['no', 'yes'],
                doc='Whether this is a picture of a cat'),
        }),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    extracted_path = dl_manager.download_and_extract('http://data.org/data.zip')
    # dl_manager returns pathlib-like objects with `path.read_text()`,
    # `path.iterdir()`,...
    return {
        'train': self._generate_examples(path=extracted_path / 'train_images'),
        'test': self._generate_examples(path=extracted_path / 'test_images'),
    }

  def _generate_examples(self, path) -> Iterator[Tuple[Key, Example]]:
    """Generator of examples for each split."""
    for img_path in path.glob('*.jpeg'):
      # Yields (key, example)
      yield img_path.name, {
          'image': img_path,
          'label': 'yes' if img_path.name.startswith('yes_') else 'no',
      }
```

Tenga en cuenta que, para algunos formatos de datos específicos, proporcionamos [generadores de conjuntos de datos](https://www.tensorflow.org/datasets/format_specific_dataset_builders) listos para usar para que se encarguen de la mayor parte del procesamiento de datos.

Veamos en detalle los 3 métodos abstractos para sobrescribir.

### `_info`: metadatos del conjunto de datos

`_info` devuelve `tfds.core.DatasetInfo` que contiene los [metadatos del conjunto de datos](https://www.tensorflow.org/datasets/overview#access_the_dataset_metadata).

```python
def _info(self):
  # The `dataset_info_from_configs` base method will construct the
  # `tfds.core.DatasetInfo` object using the passed-in parameters and
  # adding: builder (self), description/citations/tags from the config
  # files located in the same package.
  return self.dataset_info_from_configs(
      homepage='https://dataset-homepage.org',
      features=tfds.features.FeaturesDict({
          'image_description': tfds.features.Text(),
          'image': tfds.features.Image(),
          # Here, 'label' can be 0-4.
          'label': tfds.features.ClassLabel(num_classes=5),
      }),
      # If there's a common `(input, target)` tuple from the features,
      # specify them here. They'll be used if as_supervised=True in
      # builder.as_dataset.
      supervised_keys=('image', 'label'),
      # Specify whether to disable shuffling on the examples. Set to False by default.
      disable_shuffling=False,
  )
```

La mayoría de los campos deberían explicarse por sí solos. Algunas precisiones:

- `features`: Esto especifica la estructura del conjunto de datos, la forma. Admite tipos de datos complejos (audio, vídeo, secuencias anidadas). Consulte las [funciones disponibles](https://www.tensorflow.org/datasets/api_docs/python/tfds/features#classes) o la [guía del conector de funciones](https://www.tensorflow.org/datasets/features) para obtener más información.
- `disable_shuffling`: consulte la sección [Mantener el orden del conjunto de datos](#maintain-dataset-order).

Escribir el archivo `BibText` `CITATIONS.bib`:

- Busque en el sitio web del conjunto de datos instrucciones para cita (úselo en formato BibTex).
- Para artículos [arXiv](https://arxiv.org/): busque el artículo y haga clic en el enlace `BibText` en el lado derecho.
- Busque el artículo en [Google Scholar](https://scholar.google.com) y haga clic en las comillas dobles debajo del título y, en la ventana emergente, haga clic en `BibTeX` .
- Si no hay ningún documento asociado (por ejemplo, solo hay un sitio web), puede usar el [Editor en línea de BibTeX](https://truben.no/latex/bibtex/) para crear una entrada BibTeX personalizada (el menú desplegable tiene un tipo de entrada `Online`).

Actualizar el archivo `TAGS.txt`:

- Todas las etiquetas permitidas se completan previamente en el archivo generado.
- Elimine todas las etiquetas que no se apliquen al conjunto de datos.
- Las etiquetas válidas se enumeran en [tensorflow_datasets/core/valid_tags.txt](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/valid_tags.txt).
- Para agregar una etiqueta a esa lista, envíe un PR.

#### Mantener el orden del conjunto de datos

De forma predeterminada, los registros de los conjuntos de datos se mezclan cuando se almacenan para que la distribución de clases sea más uniforme en todo el conjunto de datos, ya que a menudo los registros que pertenecen a la misma clase son contiguos. Para especificar que el conjunto de datos debe estar ordenado según la clave generada que proporciona `_generate_examples`, se debe establecer el campo `disable_shuffling` en `True`. Está configurado en `False`, de forma predetarminada.

```python
def _info(self):
  return self.dataset_info_from_configs(
    # [...]
    disable_shuffling=True,
    # [...]
  )
```

Tenga en cuenta que deshabilitar la reproducción aleatoria afecta al rendimiento, ya que los fragmentos ya no se pueden leer en paralelo.

### `_split_generators`: descarga y divide datos

#### Descargar y extraer datos de origen

La mayoría de los conjuntos de datos necesitan descargar datos de la web. Esto se hace con el argumento de entrada `tfds.download.DownloadManager` de `_split_generators`. `dl_manager` tiene los siguientes métodos:

- `download`: admite `http(s)://`, `ftp(s)://`
- `extract`: actualmente admite archivos `.zip`, `.gz` y `.tar`.
- `download_and_extract`: Igual que `dl_manager.extract(dl_manager.download(urls))`

Todos estos métodos devuelven `tfds.core.Path` (alias para [`epath.Path`](https://github.com/google/etils)), que son objetos [similares a pathlib.Path](https://docs.python.org/3/library/pathlib.html).

Estos métodos admiten estructuras anidadas arbitrarias (`list`, `dict`), como:

```python
extracted_paths = dl_manager.download_and_extract({
    'foo': 'https://example.com/foo.zip',
    'bar': 'https://example.com/bar.zip',
})
# This returns:
assert extracted_paths == {
    'foo': Path('/path/to/extracted_foo/'),
    'bar': Path('/path/extracted_bar/'),
}
```

#### Descarga y extracción manual

Algunos datos no se pueden descargar automáticamente (por ejemplo, requieren un inicio de sesión); en este caso, el usuario descargará manualmente los datos de origen y los colocará en `manual_dir/` (el valor predeterminado es `~/tensorflow_datasets/downloads/manual/`).

Luego se puede acceder a los archivos a través de `dl_manager.manual_dir`:

```python
class MyDataset(tfds.core.GeneratorBasedBuilder):

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Register into https://example.org/login to get the data. Place the `data.zip`
  file in the `manual_dir/`.
  """

  def _split_generators(self, dl_manager):
    # data_path is a pathlib-like `Path('<manual_dir>/data.zip')`
    archive_path = dl_manager.manual_dir / 'data.zip'
    # Extract the manually downloaded `data.zip`
    extracted_path = dl_manager.extract(archive_path)
    ...
```

La ubicación `manual_dir` se puede personalizar con `tfds build --manual_dir=` o con `tfds.download.DownloadConfig`.

#### Leer archivo directamente

`dl_manager.iter_archive` lee archivos secuencialmente sin extraerlos. Esto puede ahorrar espacio de almacenamiento y mejorar el rendimiento en algunos sistemas de archivos.

```python
for filename, fobj in dl_manager.iter_archive('path/to/archive.zip'):
  ...
```

`fobj` tiene los mismos métodos que `with open('rb') as fobj:` (por ejemplo, `fobj.read()`)

#### Especificar divisiones de conjuntos de datos

Si el conjunto de datos viene con divisiones predefinidas (por ejemplo, `MNIST` tiene divisiones de `train` y `test`), consérvelas. De lo contrario, especifique solo una única división `all`. Los usuarios pueden crear dinámicamente sus propias subdivisiones con la [API de subdivisión](https://www.tensorflow.org/datasets/splits) (por ejemplo, `split='train[80%:]'`). Tenga en cuenta que cualquier cadena de texto en orden alfabpetico se puede utilizar como nombre dividido, aparte del `all`.

```python
def _split_generators(self, dl_manager):
  # Download source data
  extracted_path = dl_manager.download_and_extract(...)

  # Specify the splits
  return {
      'train': self._generate_examples(
          images_path=extracted_path / 'train_imgs',
          label_path=extracted_path / 'train_labels.csv',
      ),
      'test': self._generate_examples(
          images_path=extracted_path / 'test_imgs',
          label_path=extracted_path / 'test_labels.csv',
      ),
  }
```

### `_generate_examples`: generador de ejemplo

`_generate_examples` genera los ejemplos para cada división a partir de los datos de origen.

Este método normalmente leerá artefactos del conjunto de datos de origen (por ejemplo, un archivo CSV) y generará tuplas `(key, feature_dict)`:

- `key`: identificador de ejemplo. Se utiliza para mezclar de manera determinista los ejemplos con `hash(key)` o para ordenar por clave cuando la reproducción aleatoria está deshabilitada (consulte la sección [Mantener el orden del conjunto de datos](#maintain-dataset-order)). Sería:
    - **único**: si dos ejemplos usan la misma clave, se generará una excepción.
    - **determinista**: si no debe depender del orden `download_dir`, `os.path.listdir`,... Si se generan los datos dos veces, se debería generar la misma clave.
    - **comparable**: si la reproducción aleatoria está deshabilitada, la clave se usará para ordenar el conjunto de datos.
- `feature_dict`: un `dict` que contiene los valores de ejemplo.
    - La estructura debe coincidir con la estructura `features=` definida en `tfds.core.DatasetInfo`.
    - Los tipos de datos complejos (imagen, vídeo, audio,...) se codificarán automáticamente.
    - Cada característica suele aceptar varios tipos de entrada (por ejemplo, un video acepta `/path/to/vid.mp4`, `np.array(shape=(l, h, w, c))`, `List[paths]`, `List[np.array(shape=(h, w, c)]`, `List[img_bytes]`,...)
    - Consulte la [guía del conector de funciones](https://www.tensorflow.org/datasets/features) para obtener más información.

```python
def _generate_examples(self, images_path, label_path):
  # Read the input data out of the source files
  with label_path.open() as f:
    for row in csv.DictReader(f):
      image_id = row['image_id']
      # And yield (key, feature_dict)
      yield image_id, {
          'image_description': row['description'],
          'image': images_path / f'{image_id}.jpeg',
          'label': row['label'],
      }
```

Advertencia: al analizar valores booleanos de cadenas de texto o números enteros, use la función util `tfds.core.utils.bool_utils.parse_bool` para evitar errores de análisis (por ejemplo, `bool("False") == True`).

#### Acceso a archivos y `tf.io.gfile`

Para admitir sistemas de almacenamiento en la nube, evite el uso de operaciones de E/S integradas en Python.

En cambio, `dl_manager` devuelve objetos [similares a pathlib](https://docs.python.org/3/library/pathlib.html) directamente compatibles con el almacenamiento de Google Cloud:

```python
path = dl_manager.download_and_extract('http://some-website/my_data.zip')

json_path = path / 'data/file.json'

json.loads(json_path.read_text())
```

Alternativamente, use la API `tf.io.gfile` en lugar de la integrada para operaciones de archivos:

- `open` -&gt; `tf.io.gfile.GFile`
- `os.rename` -&gt; `tf.io.gfile.rename`
- ...

Pathlib es mejor que `tf.io.gfile` (consulte [racional](https://www.tensorflow.org/datasets/common_gotchas#prefer_to_use_pathlib_api)).

#### Dependencias adicionales

Algunos conjuntos de datos requieren dependencias adicionales de Python solo durante la generación. Por ejemplo, el conjunto de datos SVHN usa `scipy` para cargar algunos datos.

Si se va a agregar un conjunto de datos al repositorio TFDS, use `tfds.core.lazy_imports` para que el tamaño del paquete `tensorflow-datasets` siga siendo pequeño. Los usuarios instalarán dependencias adicionales solo cuando sea necesario.

Para usar `lazy_imports`:

- Agregue una entrada para su conjunto de datos en `DATASET_EXTRAS` en [`setup.py`](https://github.com/tensorflow/datasets/tree/master/setup.py). Esto hace que los usuarios puedan hacer, por ejemplo, `pip install 'tensorflow-datasets[svhn]'` para instalar las dependencias adicionales.
- Agregue una entrada para la importación a [`LazyImporter`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/lazy_imports_lib.py) y a [`LazyImportsTest`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/lazy_imports_lib_test.py).
- Use `tfds.core.lazy_imports` para acceder a la dependencia (por ejemplo, `tfds.core.lazy_imports.scipy`) en su `DatasetBuilder`.

#### Datos dañados

Algunos conjuntos de datos no están perfectamente limpios y contienen algunos datos dañados (por ejemplo, las imágenes están en archivos JPEG pero algunas no son JPEG no válidas). Estos ejemplos se deben omitir, pero deje una nota en la descripción del conjunto de datos sobre cuántos ejemplos se eliminaron y por qué.

### Configuración/variantes del conjunto de datos (tfds.core.BuilderConfig)

Algunos conjuntos de datos pueden tener varias variantes u opciones sobre cómo se preprocesan y escriben los datos en el disco. Por ejemplo, [Cycle_gan](https://www.tensorflow.org/datasets/catalog/cycle_gan) tiene una configuración por par de objetos (`cycle_gan/horse2zebra`, `cycle_gan/monet2photo`,...).

Esto se hace a través de los `tfds.core.BuilderConfig`:

1. Defina su objeto de configuración como una subclase de `tfds.core.BuilderConfig`. Por ejemplo, `MyDatasetConfig`.

    ```python
    @dataclasses.dataclass
    class MyDatasetConfig(tfds.core.BuilderConfig):
      img_size: Tuple[int, int] = (0, 0)
    ```

    Nota: Los valores predeterminados son obligatorios debido a https://bugs.python.org/issue33129.

2. Defina el miembro de la clase `BUILDER_CONFIGS = []` en `MyDataset` que enumera los `MyDatasetConfig` que expone el conjunto de datos.

    ```python
    class MyDataset(tfds.core.GeneratorBasedBuilder):
      VERSION = tfds.core.Version('1.0.0')
      # pytype: disable=wrong-keyword-args
      BUILDER_CONFIGS = [
          # `name` (and optionally `description`) are required for each config
          MyDatasetConfig(name='small', description='Small ...', img_size=(8, 8)),
          MyDatasetConfig(name='big', description='Big ...', img_size=(32, 32)),
      ]
      # pytype: enable=wrong-keyword-args
    ```

    Nota: `# pytype: disable=wrong-keyword-args` es necesario debido a [un error de Pytype](https://github.com/google/pytype/issues/628) con la herencia de clases de datos.

3. Use `self.builder_config` en `MyDataset` para configurar la generación de datos (por ejemplo, `shape=self.builder_config.img_size`). Esto puede incluir establecer valores diferentes en `_info()` o cambiar el acceso a los datos de descarga.

Notas:

- Cada configuración tiene un nombre único. El nombre completo de una configuración es `dataset_name/config_name` (por ejemplo, `coco/2017`).
- Si no se especifica, se usará la primera configuración en `BUILDER_CONFIGS` (por ejemplo `tfds.load('c4')` de forma predeterminada es `c4/en`)

Consulte [`anli`](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/anli.py#L69) para ver un ejemplo de un conjunto de datos que usan `BuilderConfig`.

### Versión

Versión puede referirse a dos significados diferentes:

- La versión de datos originales "externa": por ejemplo, COCO v2019, v2017,...
- La versión "interna" del código TFDS: por ejemplo, cambiar el nombre de una función en `tfds.features.FeaturesDict`, corregir un error en `_generate_examples`

Para actualizar un conjunto de datos:

- Para la actualización de datos "externos": es posible que varios usuarios quieran acceder a un año/versión específica simultáneamente. Esto se hace con un `tfds.core.BuilderConfig` por versión (por ejemplo, `coco/2017`, `coco/2019`) o una clase por versión (por ejemplo, `Voc2007`, `Voc2012`).
- Para la actualización de código "interno": los usuarios solo descargan la versión más reciente. Cualquier actualización de código debe aumentar el atributo de la clase `VERSION` (por ejemplo, de `1.0.0` a `VERSION = tfds.core.Version('2.0.0')`) después del [control de versiones semántico](https://www.tensorflow.org/datasets/datasets_versioning#semantic).

### Agregar una importación para el registro

No olvide importar el módulo del conjunto de datos a su proyecto `__init__` para que se registre automáticamente en `tfds.load`, `tfds.builder`.

```python
import my_project.datasets.my_dataset  # Register MyDataset

ds = tfds.load('my_dataset')  # MyDataset available
```

Por ejemplo, si contribuye con `tensorflow/datasets`, agregue la importación del módulo al `__init__.py` de su subdirectorio (por ejemplo, [`image/__init__.py`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/image/__init__.py).

### Verificar los errores de implementación comunes

Verifique los [errores de implementación comunes](https://www.tensorflow.org/datasets/common_gotchas).

## Pruebe su conjunto de datos

### Descargar y preparar: `tfds build`

Para generar el conjunto de datos, ejecute `tfds build` desde el directorio `my_dataset/`:

```sh
cd path/to/datasets/my_dataset/
tfds build --register_checksums
```

Algunos indicadores útiles para el desarrollo:

- `--pdb`: ingresa al modo de depuración si se genera una excepción.
- `--overwrite`: elimina archivos existentes si ya se generó el conjunto de datos.
- `--max_examples_per_split`: genera solo los primeros X ejemplos (predeterminado en 1), en lugar del conjunto de datos completo.
- `--register_checksums`: registra las sumas de verificación de las URL descargadas. Sólo debe usarse durante el desarrollo.

Consulte la [documentación de la interfaz de la línea de comandos](https://www.tensorflow.org/datasets/cli#tfds_build_download_and_prepare_a_dataset) para obtener una lista completa de indicadores.

### Sumas de comprobación

Se recomienda registrar las sumas de comprobación de sus conjuntos de datos para garantizar el determinismo, ayudar con la documentación,... Esto se hace generando el conjunto de datos con `--register_checksums` (consulte la sección anterior).

Si obtiene la versión de sus conjuntos de datos a través de PyPI, no olvide exportar los archivos `checksums.tsv` (por ejemplo, en `package_data` de su `setup.py`).

### Pruebe unitariamente su conjunto de datos

`tfds.testing.DatasetBuilderTestCase` es un `TestCase` base para ejercitar completamente un conjunto de datos. Usa "datos ficticios" como datos de prueba que imitan la estructura del conjunto de datos de origen.

- Los datos de prueba deben colocarse en el directorio `my_dataset/dummy_data/` y deben imitar los artefactos del conjunto de datos de origen tal como se descargan y extraen. Se puede crear de forma manual o automática con un script ([script de ejemplo](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/datasets/bccd/dummy_data_generation.py)).
- Asegúrese de usar datos diferentes en las divisiones de datos de su prueba, ya que la prueba fallará si las divisiones de su conjunto de datos se superponen.
- **Los datos de la prueba no deben contener ningún material protegido por derechos de autor**. En caso de duda, no cree los datos con material del conjunto de datos original.

```python
import tensorflow_datasets as tfds
from . import my_dataset_dataset_builder


class MyDatasetTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for my_dataset dataset."""
  DATASET_CLASS = my_dataset_dataset_builder.Builder
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  DL_EXTRACT_RESULT = {
      'name1': 'path/to/file1',  # Relative to my_dataset/dummy_data dir.
      'name2': 'file2',
  }


if __name__ == '__main__':
  tfds.testing.test_main()
```

Ejecute el siguiente comando para probar el conjunto de datos.

```sh
python my_dataset_test.py
```

## Envíenos sus comentarios

Siempre buscamos mejorar el flujo de trabajo de la creación de conjuntos de datos, pero solo podemos hacerlo si conocemos los problemas que hay. ¿Qué problemas o errores encontró al crear el conjunto de datos? ¿Hubo alguna parte que le resultó confusa o no funcionó la primera vez?

Deje sus comentarios en [GitHub](https://github.com/tensorflow/datasets/issues).
