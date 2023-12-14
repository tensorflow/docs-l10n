# Consejos de rendimiento

Este documento proporciona consejos de rendimiento específicos de TensorFlow Datasets (TFDS). Tenga en cuenta que TFDS proporciona conjuntos de datos como objetos `tf.data.Dataset`, por lo que aún aplican los consejos de la [guía de `tf.data`](https://www.tensorflow.org/guide/data_performance#optimize_performance).

## Conjuntos de datos de prueba comparativa

Use `tfds.benchmark(ds)` para realizar al prueba comparativa de cualquier objeto `tf.data.Dataset`.

Asegúrese de indicar el `batch_size=` para normalizar los resultados (por ejemplo, 100 iter/seg -&gt; 3200 ex/seg). Esto funciona con cualquier iterable (por ejemplo, `tfds.benchmark(tfds.as_numpy(ds))`).

```python
ds = tfds.load('mnist', split='train').batch(32).prefetch()
# Display some benchmark statistics
tfds.benchmark(ds, batch_size=32)
# Second iteration is much faster, due to auto-caching
tfds.benchmark(ds, batch_size=32)
```

## Conjuntos de datos pequeños (menos de 1 GB)

Todos los conjuntos de datos TFDS almacenan los datos en el disco en formato [`TFRecord`](https://www.tensorflow.org/tutorials/load_data/tfrecord). Para los conjuntos de datos pequeños (por ejemplo, MNIST, CIFAR-10/-100), la lectura de `.tfrecord` puede suponer una sobrecarga significativa.

Mientras que esos conjuntos de datos entren en la memoria, es posible mejorar significativamente el rendimiento al almacenar en caché o precargar el conjunto de datos. Tenga en cuenta que TFDS almacena en caché conjuntos de datos pequeños de forma automática (puede ver los detalles en la siguiente sección).

### Almacenar en caché el conjunto de datos

A continuación se muestra un ejemplo de una canalización de datos que almacena en caché el conjunto de datos de forma explícita después de normalizar las imágenes.

```python
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


ds, ds_info = tfds.load(
    'mnist',
    split='train',
    as_supervised=True,  # returns `(img, label)` instead of dict(image=, ...)
    with_info=True,
)
# Applying normalization before `ds.cache()` to re-use it.
# Note: Random transformations (e.g. images augmentations) should be applied
# after both `ds.cache()` (to avoid caching randomness) and `ds.batch()` (for
# vectorization [1]).
ds = ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.cache()
# For true randomness, we set the shuffle buffer to the full dataset size.
ds = ds.shuffle(ds_info.splits['train'].num_examples)
# Batch after shuffling to get unique batches at each epoch.
ds = ds.batch(128)
ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
```

- [[1] Asignación de vectorización](https://www.tensorflow.org/guide/data_performance#vectorizing_mapping)

Al iterar en este conjunto de datos, la segunda iteración será mucho más rápida que la primera gracias al almacenamiento en caché.

### Almacenamiento en caché automático

De forma predeterminada, TFDS almacena en caché automáticamente (con `ds.cache()`) conjuntos de datos que cumplen con las siguientes restricciones:

- El tamaño total del conjunto de datos (todas las divisiones) está definido y es &lt; 250 MiB
- `shuffle_files` está deshabilitado o solo se lee una partición

Es posible desactivar el almacenamiento en caché automático al pasar `try_autocaching=False` a `tfds.ReadConfig` en `tfds.load`. Consulte la documentación del catálogo de conjuntos de datos para ver si un conjunto de datos específico usa el caché automático.

### Cargar todos los datos como un solo Tensor

Si su conjunto de datos entra en la memoria, también puede cargar todo el conjunto de datos como un arreglo único Tensor o NumPy. Esto es posible si se configura `batch_size=-1` para agrupar todos los ejemplos en un solo `tf.Tensor`. Luego se usa `tfds.as_numpy` para la conversión de `tf.Tensor` a `np.array`.

```python
(img_train, label_train), (img_test, label_test) = tfds.as_numpy(tfds.load(
    'mnist',
    split=['train', 'test'],
    batch_size=-1,
    as_supervised=True,
))
```

## Conjuntos de datos grandes

Los conjuntos de datos grandes están particionados (divididos en varios archivos) y normalmente no entran en la memoria, por lo que no deben almacenarse en caché.

### Aleatorización y entrenamiento

Durante el entrenamiento, es importante aleatorizar bien los datos; los datos mal aleatorizados pueden causar que el entranamiento sea menos preciso.

Aparte de usar `ds.shuffle` para aleatorizar registros, también debe configurar `shuffle_files=True` para obtener un buen comportamiento de aleatorización para los conjuntos de datos más grandes que están divididos en varios archivos. De lo contrario, las épocas leerán las particiones en el mismo orden y, por lo tanto, los datos no serán realmente aleatorios.

```python
ds = tfds.load('imagenet2012', split='train', shuffle_files=True)
```

Además, cuando `shuffle_files=True`, TFDS deshabilita [`options.deterministic`](https://www.tensorflow.org/api_docs/python/tf/data/Options#deterministic), lo que puede aumentar ligeramente el rendimiento. Para obtener una mezcla aleatoria determinista, se puede desactivar esta función con `tfds.ReadConfig`: ya sea al configurar `read_config.shuffle_seed` o sobrescribir `read_config.options.deterministic`.

### Particionar datos automáticamente entre los trabajadores (TF)

Cuando se entrena con varios trabajadores, puede usar el argumento `input_context` de `tfds.ReadConfig`, así cada trabajador leerá un subconjunto de datos.

```python
input_context = tf.distribute.InputContext(
    input_pipeline_id=1,  # Worker id
    num_input_pipelines=4,  # Total number of workers
)
read_config = tfds.ReadConfig(
    input_context=input_context,
)
ds = tfds.load('dataset', split='train', read_config=read_config)
```

Esto es complementario a la API de subdivisón. Primero, se aplica la API de subdivisión: `train[:50%]` se convierte en una lista de archivos para leer. Luego, se aplica una operación `ds.shard()` en esos archivos. Por ejemplo, cuando se usa `train[:50%]` con `num_input_pipelines=2`, cada uno de los 2 trabajadores leerá 1/4 de los datos.

Cuando `shuffle_files=True`, los archivos se aleatorizan en un trabajador, pero no entre los trabajadores. Cada trabajador leerá el mismo subconjunto de archivos entre épocas.

Nota: Cuando se usa `tf.distribute.Strategy`, se puede crear automáticamente el `input_context` con [distribuir_datasets_from_function](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy#distribute_datasets_from_function)

### Particione sus datos automáticamente entre los trabajadores (Jax)

Con Jax, puede usar la API `tfds.split_for_jax_process` o `tfds.even_splits` para distribuir sus datos entre los trabajadores. Consulte la [guía de la API de división](https://www.tensorflow.org/datasets/splits).

```python
split = tfds.split_for_jax_process('train', drop_remainder=True)
ds = tfds.load('my_dataset', split=split)
```

`tfds.split_for_jax_process` es un alias simple para:

```python
# The current `process_index` loads only `1 / process_count` of the data.
splits = tfds.even_splits('train', n=jax.process_count(), drop_remainder=True)
split = splits[jax.process_index()]
```

### Decodificación de imágenes más rápida

De forma predeterminada, TFDS decodifica imágenes automáticamente. Sin embargo, hay casos en los que puede resultar más eficaz omitir la decodificación de la imagen con `tfds.decode.SkipDecoding` y aplicar la operación `tf.io.decode_image` de forma manual:

- Al filtrar ejemplos (con `tf.data.Dataset.filter`), para decodificar imágenes después de filtrar los ejemplos.
- Al recortar imágenes, para usar la operación `tf.image.decode_and_crop_jpeg` fusionada.

El código de ambos ejemplos está disponible en la [guía de decodificación](https://www.tensorflow.org/datasets/decode#usage_examples).

### Omitir funciones que no se usan

Si solo usa un subconjunto de funciones, es posible omitir por completo algunas funciones. Si su conjunto de datos tiene muchas funciones que no se usan, se puede mejorar significativamente el rendimiento si no se decodifican. Consulte https://www.tensorflow.org/datasets/decode#only_decode_a_sub-set_of_the_features.
