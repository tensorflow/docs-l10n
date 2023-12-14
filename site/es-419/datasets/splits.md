# Divisiones y segmento

Todos los conjuntos de datos TFDS exponen varias divisiones de datos (por ejemplo `'train'`, `'test'`) que se pueden explorar en el [catálogo](https://www.tensorflow.org/datasets/catalog/overview). Se puede usar cualquier cadena alfabética como nombre de la división, excepto `all` (que es un término reservado que corresponde a la unión de todas las divisiones, ver más abajo).

Además de las divisiones de conjuntos de datos "oficiales", TFDS permite seleccionar segmento(s) de divisione(s) y varias combinaciones.

## API de segmentación

Las instrucciones de segmentación se especifican en `tfds.load` o `tfds.DatasetBuilder.as_dataset` mediante el kwarg `split=`.

```python
ds = tfds.load('my_dataset', split='train[:75%]')
```

```python
builder = tfds.builder('my_dataset')
ds = builder.as_dataset(split='test+train[:75%]')
```

La división puede ser:

- **Nombres de división simples** (una cadena como `'train'`, `'test'`, ...): todos los ejemplos dentro de la división seleccionada.
- **Segmentos**: Los segmentos tienen la misma semántica que [la notación de segmentos de Python](https://docs.python.org/3/library/stdtypes.html#common-sequence-operations). Los segmentos pueden ser:
    - **Absolutos** (`'train[123:450]'`, `train[:4000]`): (consulte la nota a continuación para conocer las advertencias sobre el orden de la lectura)
    - **Porcentuales** (`'train[:75%]'`, `'train[25%:75%]'`): divide todos los datos en segmentos iguales. Si los datos no son divisibles uniformemente, algún porcentaje podría contener ejemplos adicionales. Se admiten porcentajes fraccionarios.
    - **Particionados** (`train[:4shard]`, `train[4shard]`): seleccione todos los ejemplos en la partición solicitada. (consulte `info.splits['train'].num_shards` para obtener la cantidad de particiones de la división)
- **Unión de divisiones** (`'train+test'`, `'train[:25%]+test'`): Las divisiones estarán intercaladas.
- **Todo el conjunto de datos** (`'all'`): `'all'` es un nombre de división especial correspondiente a la unión de todas las divisiones (equivalente a `'train+test+...'`).
- **Lista de divisiones** (`['train', 'test']`): Se devuelven varios `tf.data.Dataset` por separado:

```python
# Returns both train and test split separately
train_ds, test_ds = tfds.load('mnist', split=['train', 'test[:50%]'])
```

Nota: Debido a que los segmentos están [intercalados](https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=nightly#interleave), no se garantiza que el orden sea coherente entre las subdivisiones. En otras palabras, leer `test[0:100]` seguido de `test[100:200]` puede generar ejemplos en un orden diferente que leer `test[:200]`. Consulte [la guía de determinismo](https://www.tensorflow.org/datasets/determinism#determinism_when_reading) para comprender el orden en que TFDS lee los ejemplos.

## `tfds.even_splits` y entrenamiento con hospedaje múltiple

`tfds.even_splits` genera una lista de subdivisiones no superpuestas del mismo tamaño.

```python
# Divide the dataset into 3 even parts, each containing 1/3 of the data
split0, split1, split2 = tfds.even_splits('train', n=3)

ds = tfds.load('my_dataset', split=split2)
```

Esto puede resultar particularmente útil cuando se entrena en un entorno distribuido, donde cada host debe recibir una porción de los datos originales.

Con `Jax`, esto se puede simplificar aún más al usar `tfds.split_for_jax_process`:

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

`tfds.even_splits`, `tfds.split_for_jax_process` acepta cualquier valor dividido como entrada (por ejemplo `'train[75%:]+test'`)

## División y metadatos

Es posible obtener información adicional sobre las divisiones/subdivisiones (`num_examples`, `file_instructions`,...) con la [información del conjunto de datos](https://www.tensorflow.org/datasets/overview#access_the_dataset_metadata):

```python
builder = tfds.builder('my_dataset')
builder.info.splits['train'].num_examples  # 10_000
builder.info.splits['train[:75%]'].num_examples  # 7_500 (also works with slices)
builder.info.splits.keys()  # ['train', 'test']
```

## Validación cruzada

Ejemplos de validación cruzada de 10 iteraciones con la API de cadena:

```python
vals_ds = tfds.load('mnist', split=[
    f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)
])
trains_ds = tfds.load('mnist', split=[
    f'train[:{k}%]+train[{k+10}%:]' for k in range(0, 100, 10)
])
```

Cada uno de los conjuntos de datos de validación será del 10%: `[0%:10%]`, `[10%:20%]`, ..., `[90%:100%]`. Y cada uno de los conjuntos de datos de entrenamiento será el 90% complementario: `[10%:100%]` (para un conjunto de validación correspondiente de `[0%:10%]`), `[0%:10%]

- [20%:100%]`(for a validation set of `[10%:20%]`),...

## `tfds.core.ReadInstruction` y redondeo

En lugar de `str`, es posible pasar divisiones como `tfds.core.ReadInstruction`:

Por ejemplo, `split = 'train[50%:75%] + test'` es equivalente a:

```python
split = (
    tfds.core.ReadInstruction(
        'train',
        from_=50,
        to=75,
        unit='%',
    )
    + tfds.core.ReadInstruction('test')
)
ds = tfds.load('my_dataset', split=split)
```

`unit` puede ser:

- `abs`: división absoluta
- `%`: división porcentual
- `shard`: división particionada

`tfds.ReadInstruction` también tiene un argumento de redondeo. Si el número de ejemplos en el conjunto de datos no se divide en partes iguales:

- `rounding='closest'` (predeterminado): los ejemplos restantes se distribuyen entre el porcentaje, por lo que algún porcentaje puede contener ejemplos adicionales.
- `rounding='pct1_dropremainder'`: los ejemplos restantes se eliminan, pero esto garantiza que todos los porcentajes contengan exactamente la misma cantidad de ejemplos (por ejemplo: `len(5%) == 5 * len(1%)`).

### Reproducibilidad y determinismo

Durante la generación, para una versión determinada del conjunto de datos, TFDS garantiza que los ejemplos se ordenen de manera determinista en el disco. Por lo tanto, generar el conjunto de datos dos veces (en 2 computadoras diferentes) no cambiará el orden del ejemplo.

De manera similar, la API de subdivisión siempre seleccionará el mismo `set` de ejemplos, independientemente de la plataforma, arquitectura, etc. Esto significa `set('train[:20%]') == set('train[:10%]') + set('train[10%:20%]')`.

Sin embargo, es posible que el orden en que se leen los ejemplos **no** sea determinista. Esto depende de otros parámetros (por ejemplo, si `shuffle_files=True`).
