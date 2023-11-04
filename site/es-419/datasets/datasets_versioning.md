# Control de versiones del conjuntos de datos

## Definición

El control de versiones puede tener diferentes significados:

- La versión de la API TFDS (versión pip): `tfds.__version__`
- La versión del conjunto de datos público, independiente de TFDS (por ejemplo, [Voc2007](https://pjreddie.com/projects/pascal-voc-dataset-mirror/), Voc2012). En TFDS, cada versión del conjunto de datos público debe implementarse como un conjunto de datos independiente:
    - Ya sea a través de [configuraciones del generador](https://www.tensorflow.org/datasets/add_dataset#dataset_configurationvariants_tfdscorebuilderconfig): por ejemplo `voc/2007`, `voc/2012`
    - Ya sea como 2 conjuntos de datos independientes: por ejemplo, `wmt13_translate`, `wmt14_translate`
- La versión del código de generación del conjunto de datos en TFDS (`my_dataset:1.0.0`): por ejemplo, si se encuentra un error en la implementación de TFDS de `voc/2007`, se actualizará el código de generación `voc.py` (`voc/2007:1.0.0` - &gt; `voc/2007:2.0.0`).

En el resto de esta guía, solo nos centraremos en la última definición (versión del código del conjunto de datos en el repositorio TFDS).

## Versiones compatibles

Como regla general:

- Solo se puede generar la última versión actual.
- Se pueden leer todos los conjuntos de datos generados previamente (nota: esto requiere conjuntos de datos generados con TFDS 4+).

```python
builder = tfds.builder('my_dataset')
builder.info.version  # Current version is: '2.0.0'

# download and load the last available version (2.0.0)
ds = tfds.load('my_dataset')

# Explicitly load a previous version (only works if
# `~/tensorflow_datasets/my_dataset/1.0.0/` already exists)
ds = tfds.load('my_dataset:1.0.0')
```

## Semántico

Cada `DatasetBuilder` definido en TFDS viene con una versión, por ejemplo:

```python
class MNIST(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('2.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release',
      '2.0.0': 'Update dead download url',
  }
```

La versión sigue el [Control de versiones semántico 2.0.0](https://semver.org/spec/v2.0.0.html): `MAJOR.MINOR.PATCH`. El propósito de la versión es poder garantizar la reproducibilidad: cargar un conjunto de datos determinado en una versión fija produce los mismos datos. De forma más específica:

- Si se incrementa la versión `PATCH`, los datos que lee el cliente son los mismos, aunque es posible que los datos se serialicen de manera diferente en el disco o que los metadatos hayan cambiado. Para cualquier segmento determinado, la API de segmento devuelve el mismo conjunto de registros.
- Si se incrementa la versión `MINOR`, los datos existentes que lee el cliente son los mismos, pero hay datos adicionales (características en cada registro). Para cualquier segmento determinado, la API de segmento devuelve el mismo conjunto de registros.
- Si se incrementa la versión `MAJOR`, se cambiaron los datos existentes o la API de segmentación no devuelve necesariamente el mismo conjunto de registros para un segmento determinado.

Cuando se realiza un cambio de código en la biblioteca TFDS y ese cambio de código afecta la forma en que el cliente serializa o lee un conjunto de datos, la versión del generador correspondiente se incrementa de acuerdo con las pautas anteriores.

Tenga en cuenta que la semántica anterior es lo mejor que se puede hacer y es posible que no se detecten algunos errores que afecten a un conjunto de datos mientras la versión no se incrementa. Estos errores finalmente se solucionan, pero depende mucho del control de versiones, le recomendamos que use un TFDS de una versión publicada (a diferencia de `HEAD`).

Tenga en cuenta también que algunos conjuntos de datos tienen otro esquema de versiones independiente de la versión TFDS. Por ejemplo, el conjunto de datos Open Images tiene varias versiones y en TFDS, los generadores correspondientes son `open_images_v4`, `open_images_v5`, ...

## Cargar una versión específica

Al cargar un conjunto de datos o un `DatasetBuilder`, se puede especificar la versión que se usará. Por ejemplo:

```python
tfds.load('imagenet2012:2.0.1')
tfds.builder('imagenet2012:2.0.1')

tfds.load('imagenet2012:2.0.0')  # Error: unsupported version.

# Resolves to 3.0.0 for now, but would resolve to 3.1.1 if when added.
tfds.load('imagenet2012:3.*.*')
```

Si usa TFDS para una publicación, le recomendamos:

- **arreglar el componente `MAJOR` de la versión solamente**;
- **anunciar qué versión del conjunto de datos se usó en sus resultados.**

Al hacerlo, sería más fácil para usted, sus lectores y revisores reproducir sus resultados en el futuro.

## BUILDER_CONFIGS y versiones

Algunos conjuntos de datos definen varios `BUILDER_CONFIGS`. En este caso, `version` y `supported_versions` se definen en los propios objetos de configuración. Aparte de eso, la semántica y el uso son idénticos. Por ejemplo:

```python
class OpenImagesV4(tfds.core.GeneratorBasedBuilder):

  BUILDER_CONFIGS = [
      OpenImagesV4Config(
          name='original',
          version=tfds.core.Version('0.2.0'),
          supported_versions=[
            tfds.core.Version('1.0.0', "Major change in data"),
          ],
          description='Images at their original resolution and quality.'),
      ...
  ]

tfds.load('open_images_v4/original:1.*.*')
```

## Versión experimental

Nota: A continuación le mostraremos una mala práctica, que es propensa a errores y no es aconsejable.

Es posible permitir que se generen 2 versiones al mismo tiempo. Una versión predeterminada y otra experimental. Por ejemplo:

```python
class MNIST(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version("1.0.0")  # Default version
  SUPPORTED_VERSIONS = [
      tfds.core.Version("2.0.0"),  # Experimental version
  ]


# Download and load default version 1.0.0
builder = tfds.builder('mnist')

#  Download and load experimental version 2.0.0
builder = tfds.builder('mnist', version='experimental_latest')
```

En el código, debe asegurarse de admitir las 2 versiones:

```python
class MNIST(tfds.core.GeneratorBasedBuilder):

  ...

  def _generate_examples(self, path):
    if self.info.version >= '2.0.0':
      ...
    else:
      ...
```
