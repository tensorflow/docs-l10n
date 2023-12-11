# Errores comunes de implementación

En esta página, se describen los errores de implementación comunes al implementar un conjunto de datos nuevo.

## Se debe evitar el `SplitGenerator` heredado

La API `tfds.core.SplitGenerator` antigua está en desuso.

```python
def _split_generator(...):
  return [
      tfds.core.SplitGenerator(name='train', gen_kwargs={'path': train_path}),
      tfds.core.SplitGenerator(name='test', gen_kwargs={'path': test_path}),
  ]
```

Se debe reemplazar con:

```python
def _split_generator(...):
  return {
      'train': self._generate_examples(path=train_path),
      'test': self._generate_examples(path=test_path),
  }
```

**Justificación**: la nueva API es menos detallada y más explícita. La antigua API se eliminará en las versiones futuras.

## Los nuevos conjuntos de datos deben estar autónomos en una carpeta

Al agregar un conjunto de datos adentro del repositorio `tensorflow_datasets/`, asegúrese de seguir la estructura del conjunto de datos como carpeta (todas las sumas de comprobación, los datos ficticios, el código de implementación autónomo en una carpeta).

- Conjuntos de datos antiguos (malos): `<category>/<ds_name>.py`
- Conjuntos de datos nuevos (buenos): `<category>/<ds_name>/<ds_name>.py`

Use la [CLI de TFDS](https://www.tensorflow.org/datasets/cli#tfds_new_implementing_a_new_dataset) ( `tfds new` o `gtfds new` para trabajadores de Google) para generar la plantilla.

**Justificación**: la estructura antigua requería rutas absolutas para las sumas de comprobación, datos falsos y distribuía los archivos del conjunto de datos en muchos lugares. Dificultaba la implementación de conjuntos de datos fuera del repositorio TFDS. Para mantener la coherencia, ahora la nueva estructura debería usarse en todas partes.

## Las listas de descripciones deben tener formato de Markdown

El `str` de `DatasetInfo.description` tiene el formato Markdown. Las listas Markdown requieren una línea vacía antes del primer elemento:

```python
_DESCRIPTION = """
Some text.
                      # << Empty line here !!!
1. Item 1
2. Item 1
3. Item 1
                      # << Empty line here !!!
Some other text.
"""
```

**Justificación**: La descripción mal formateada crea artefactos visuales en la documentación de nuestro catálogo. Sin las líneas vacías, el texto anterior se representaría así:

Algún texto. 1. Elemento 1 2. Elemento 1 3. Elemento 1 Algún otro texto

## Olvidarse los nombres de ClassLabel

Cuando use `tfds.features.ClassLabel`, intente proporcionar las etiquetas `str` que puedan leer por humanos con `names=` o `names_file=` (en lugar de `num_classes=10`).

```python
features = {
    'label': tfds.features.ClassLabel(names=['dog', 'cat', ...]),
}
```

**Justificación**: Las etiquetas que pueden leer los humanos se usan en muchos lugares:

- Permiten generar el `str` directamente en `_generate_examples`: `yield {'label': 'dog'}`
- Se exponen en los usuarios como `info.features['label'].names` (método de conversión `.str2int('dog')`,... también disponible)
- Se usan en las [utilidades de visualización](https://www.tensorflow.org/datasets/overview#tfdsas_dataframe) `tfds.show_examples`, `tfds.as_dataframe`

## Olvidarse la forma de la imagen

Cuando se usan `tfds.features.Image`, `tfds.features.Video`, si las imágenes tienen forma estática, se deben especificar explícitamente:

```python
features = {
    'image': tfds.features.Image(shape=(256, 256, 3)),
}
```

**Justificación**: permite la inferencia de formas estáticas (por ejemplo, `ds.element_spec['image'].shape`), que es necesaria para el procesamiento por lotes (el procesamiento por lotes de imágenes de forma desconocida requeriría que primero se cambie su tamaño).

## Mejor un tipo más específico que `tfds.features.Tensor`

Cuando sea posible, elija los tipos más específicos `tfds.features.ClassLabel`, `tfds.features.BBoxFeatures`,... en lugar del `tfds.features.Tensor` genérico.

**Justificación**: además de ser semánticamente más correctas, las funciones específicas proporcionan metadatos adicionales a los usuarios y las herramientas pueden detectarlas.

## Importaciones perezosas en el espacio global

Las importaciones perezosas no deberían llamarse desde el espacio global. Por ejemplo, lo siguiente es incorrecto:

```python
tfds.lazy_imports.apache_beam # << Error: Import beam in the global scope

def f() -> beam.Map:
  ...
```

**Justificación**: el uso de importaciones diferidas en el espacio global importaría el módulo para todos los usuarios de tfds y anularía el propósito de las importaciones diferidas.

## Cálculo dinámico de divisiones en el entrenamiento/la prueba

Si el conjunto de datos no proporciona divisiones oficiales, TFDS tampoco debería hacerlo. Se debe evitar lo siguiente:

```python
_TRAIN_TEST_RATIO = 0.7

def _split_generator():
  ids = list(range(num_examples))
  np.random.RandomState(seed).shuffle(ids)

  # Split train/test
  train_ids = ids[_TRAIN_TEST_RATIO * num_examples:]
  test_ids = ids[:_TRAIN_TEST_RATIO * num_examples]
  return {
      'train': self._generate_examples(train_ids),
      'test': self._generate_examples(test_ids),
  }
```

**Justificación**: el TFDS intenta proporcionar conjuntos de datos que sean lo más parecidos a los datos originales. En su lugar, se debe usar la [API de subdivisión](https://www.tensorflow.org/datasets/splits) para permitir a los usuarios crear dinámicamente las subdivisiones que deseen:

```python
ds_train, ds_test = tfds.load(..., split=['train[:80%]', 'train[80%:]'])
```

## Guía de estilo de Python

### Mejor usar la API pathlib

En lugar de la API `tf.io.gfile`, es mejor usar la [API pathlib](https://docs.python.org/3/library/pathlib.html). Todos los métodos `dl_manager` devuelven objetos similares a pathlib compatibles con GCS, S3,...

```python
path = dl_manager.download_and_extract('http://some-website/my_data.zip')

json_path = path / 'data/file.json'

json.loads(json_path.read_text())
```

**Justificación**: la API pathlib es una API de archivos moderna orientada a objetos que elimina el texto repetitivo. El uso de `.read_text()` / `.read_bytes()` también garantiza que los archivos se cierren correctamente.

### Si el método no usa `self`, debería ser una función.

Si un método de clase no usa `self`, debería ser una función simple (definida fuera de la clase).

**Justificación**: queda explícito para el lector que la función no tiene efectos secundarios ni entradas/salidas ocultas:

```python
x = f(y)  # Clear inputs/outputs

x = self.f(y)  # Does f depend on additional hidden variables ? Is it stateful ?
```

## Importaciones diferidas en Python

Importamos de forma diferida los módulos grandes como TensorFlow. Las importaciones diferidas difieren la importación real del módulo hasta el primer uso del módulo. Por lo tanto, los usuarios que no necesiten este módulo grande nunca lo importarán.

```python
from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf
# After this statement, TensorFlow is not imported yet

...

features = tfds.features.Image(dtype=tf.uint8)
# After using it (`tf.uint8`), TensorFlow is now imported
```

En el fondo, la [clase `LazyModule`](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/utils/lazy_imports_utils.py) actúa como una fábrica, que solo importará el módulo cuando se acceda a un atributo (`__getattr__`).

También se puede usar de forma conveniente con un administrador de contexto:

```python
from tensorflow_datasets.core.utils.lazy_imports_utils import lazy_imports

with lazy_imports(error_callback=..., success_callback=...):
  import some_big_module
```
