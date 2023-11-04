# <a>Analizar el rendimiento de <code>tf.data</code> con el TF Profiler</a>

## Visión general

Esta guía asume que está familiarizado con el [Profiler](https://www.tensorflow.org/guide/profiler) y [`tf.data`](https://www.tensorflow.org/guide/data) de TensorFlow. Su finalidad es dar instrucciones paso a paso con ejemplos para ayudar a los usuarios a diagnosticar y solucionar problemas de rendimiento de la canalización de entrada.

Para empezar, recabe un perfil de su trabajo TensorFlow. Las instrucciones sobre cómo hacerlo están disponibles para [CPUs/GPUs](https://www.tensorflow.org/guide/profiler#collect_performance_data) y [Cloud TPUs](https://cloud.google.com/tpu/docs/cloud-tpu-tools#capture_profile).

![Trace Viewer de TensorFlow](images/data_performance_analysis/trace_viewer.png "The trace viewer page of the TensorFlow Profiler")

El flujo de trabajo de análisis a continuación se centra en la herramienta Trace Viewer en el perfilador. Esta herramienta despliega una línea de tiempo que muestra la duración de las operaciones ejecutadas por su programa TensorFlow, y le permite identificar qué operaciones tardan más en ejecutarse. Para más información sobre el Trace Viewer, consulte [esta sección](https://www.tensorflow.org/guide/profiler#trace_viewer) de la guía del Perfilador TF. En general, `tf.data` los eventos aparecen en la línea de tiempo de la CPU anfitriona.

## Flujo de trabajo del análisis

*Siga el flujo de trabajo que se indica a continuación. Si tiene comentarios que nos ayuden a mejorarlo, [cree una incidencia en github](https://github.com/tensorflow/tensorflow/issues/new/choose) con la etiqueta "comp:data".*

### 1. ¿Su canalización `tf.data` produce datos con suficiente rapidez?

Empiece por averiguar si la canalización de entrada es el cuello de botella de su programa TensorFlow.

Para ello, busque ops de `IteratorGetNext::DoCompute` en el visor de seguimiento. En general, se espera verlas al comienzo de un paso. Estos fragmentos representan el tiempo que tarda su canalización de entrada en producir un lote de elementos cuando se le solicita. Si está usando keras o iterando sobre su conjunto de datos en una `tf.function`, estos deberían encontrarse en los hilos `tf_data_iterator_get_next`.

Tenga en cuenta que si está usando una [estrategia de distribución](https://www.tensorflow.org/guide/distributed_training), es posible que vea eventos `IteratorGetNextAsOptional::DoCompute` en lugar de `IteratorGetNext::DoCompute`(a partir de TF 2.3).

![imagen](images/data_performance_analysis/get_next_fast.png "If your IteratorGetNext::DoCompute calls return quickly, `tf.data` is not your bottleneck.")

**Si las llamadas regresan rápido (&lt;= 50 us),** esto significa que sus datos están disponibles cuando se solicitan. La canalización de entrada no es su cuello de botella; consulte la [Guía del perfilador](https://www.tensorflow.org/guide/profiler) para obtener consejos más genéricos sobre el análisis del rendimiento.

![imagen](images/data_performance_analysis/get_next_slow.png "If your IteratorGetNext::DoCompute calls return slowly, `tf.data` is not producing data quickly enough.")

**Si las llamadas regresan lentamente,** `tf.data` es incapaz de mantener el ritmo de las peticiones del consumidor. Continúe con la sección siguiente.

### 2. ¿Está preextrayendo datos?

La mejor práctica para el rendimiento de la canalización de entrada es insertar una transformación `tf.data.Dataset.prefetch` al final de su canalización `tf.data`. Esta transformación superpone el cálculo de preprocesamiento de la canalización de entrada con el siguiente paso del cálculo del modelo y es necesaria para un rendimiento óptimo de la canalización de entrada al entrenar su modelo. Si está preextrayendo datos, debería ver un trozo `Iterator::Prefetch` en el mismo hilo que la operación `IteratorGetNext::DoCompute`.

![imagen](images/data_performance_analysis/prefetch.png "If you're prefetching data, you should see a `Iterator::Prefetch` slice in the same stack as the `IteratorGetNext::DoCompute` op.")

**Si no tiene un `prefetch` al final de su canalización**, debería añadir uno. Para más información sobre las recomendaciones de rendimiento de `tf.data`, consulte la guía de rendimiento de [tf.data](https://www.tensorflow.org/guide/data_performance#prefetching).

**Si ya está preextrayendo datos**, y la canalización de entrada sigue siendo su cuello de botella, continúe con la siguiente sección para analizar más a fondo el rendimiento.

### 3. ¿Está registrando una alta utilización de la CPU?

`tf.data` logra un alto rendimiento intentando usar lo mejor posible los recursos disponibles. En general, incluso cuando ejecuta su modelo en un acelerador como una GPU o TPU, las canalizaciones `tf.data` se ejecutan en la CPU. Puede comprobar su utilización con herramientas como [sar](https://linux.die.net/man/1/sar) y [htop](https://en.wikipedia.org/wiki/Htop), o en la [consola de monitoreo en la nube](https://cloud.google.com/monitoring/docs/monitoring_in_console) si está ejecutando en GCP.

**Si su utilización es baja,** esto sugiere que su canalización de entrada puede no estar aprovechando al máximo la CPU del host. Consulte la [guía de rendimiento de tf.data](https://www.tensorflow.org/guide/data_performance) para conocer las mejores prácticas. Si ha aplicado las mejores prácticas y la utilización y el rendimiento siguen siendo bajos, prosiga con [Análisis de cuellos de botella](#4_bottleneck_analysis) a continuación.

**Si su utilización se está acercando al límite de recursos**, para mejorar aún más el rendimiento, debe mejorar la eficacia de su canal de entrada (por ejemplo, evitando el cálculo innecesario) o aliviar la carga de cálculo.

Puede mejorar la eficiencia de su canal de entrada evitando los cálculos innecesarios en `tf.data`. Puede hacerlo insertando una transformación [`tf.data.Dataset.cache`](https://www.tensorflow.org/guide/data_performance#caching) después del trabajo de cálculo intensivo si sus datos caben en la memoria; esto reduce el cálculo a costa de un mayor uso de la memoria. Además, desactivar el paralelismo intra-operativo en `tf.data` tiene el potencial de aumentar la eficiencia en &gt; 10%, y puede hacerse estableciendo la siguiente opción en su canalización de entrada:

```python
dataset = ...
options = tf.data.Options()
options.experimental_threading.max_intra_op_parallelism = 1
dataset = dataset.with_options(options)
```

### 4. Análisis de cuellos de botella

La siguiente sección explica cómo leer los eventos `tf.data` en el visor de seguimiento para comprender dónde se encuentra el cuello de botella y las posibles estrategias de mitigación.

#### Comprender los eventos `tf.data` en el Perfilador

Cada evento `tf.data` del Perfilador tiene el nombre `Iterador::<Dataset>`, donde `<Dataset>` es el nombre de la fuente o transformación del conjunto de datos. Cada evento también tiene el nombre largo `Iterator::<Dataset_1>::...::<Dataset_n>`, que puede ver haciendo clic en el evento `tf.data`. En el nombre largo, `<Dataset_n>` coincide con `<Dataset>` del nombre (corto), y los demás conjuntos de datos del nombre largo representan transformaciones posteriores.

![imagen](images/data_performance_analysis/map_long_name.png "tf.data.Dataset.range(10).map(lambda x: x).repeat(2).batch(5)")

Por ejemplo, la captura de pantalla anterior se generó a partir del siguiente código:

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.map(lambda x: x)
dataset = dataset.repeat(2)
dataset = dataset.batch(5)
```

Aquí, el evento `Iterator::Map` tiene el nombre largo `Iterator::BatchV2::FiniteRepeat::Map`. Tenga en cuenta que el nombre de los conjuntos de datos puede diferir ligeramente de la API de Python (por ejemplo, FiniteRepeat en lugar de Repeat), pero debería ser lo suficientemente intuitivo como para parsearlo.

##### Transformaciones síncronas y asíncronas

Para las transformaciones síncronas `tf.data` (como `Batch` y `Map`), verá los eventos de las transformaciones ascendentes en el mismo hilo. En el ejemplo anterior, como todas las transformaciones usadas son síncronas, todos los eventos aparecen en el mismo hilo.

Para las transformaciones asíncronas (como `Prefetch`, `ParallelMap`, `ParallelInterleave` y `MapAndBatch`) los eventos de las transformaciones ascendentes estarán en un hilo diferente. En estos casos, el "nombre largo" puede ayudarle a identificar a qué transformación de la cadena de suministro corresponde un evento.

![imagen](images/data_performance_analysis/async_long_name.png "tf.data.Dataset.range(10).map(lambda x: x).repeat(2).batch(5).prefetch(1)")

Por ejemplo, la captura de pantalla anterior se generó a partir del siguiente código:

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.map(lambda x: x)
dataset = dataset.repeat(2)
dataset = dataset.batch(5)
dataset = dataset.prefetch(1)
```

Aquí, los eventos `Iterator::Prefetch` están en los hilos `tf_data_iterator_get_next`. Dado que `Prefetch` es asíncrono, sus eventos de entrada (`BatchV2`) estarán en un hilo diferente, y pueden localizarse buscando el nombre largo `Iterator::Prefetch::BatchV2`. En este caso, se encuentran en el hilo `tf_data_iterator_resource`. Por su nombre largo, se puede deducir que `BatchV2` es ascendente de `Prefetch`. Además, el `parent_id` del evento `BatchV2` coincidirá con el ID del evento `Prefetch`.

#### Identificar el cuello de botella

En general, para identificar el cuello de botella en su canalización de entrada, recórrala desde la transformación más externa hasta el origen. Empezando por la transformación final de su canalización, recurra a las transformaciones anteriores hasta que encuentre una transformación lenta o llegue a un conjunto de datos fuente, como `TFRecord`. En el ejemplo anterior, empezaría por `Prefetch`, luego recorrería ascendentemente hasta `BatchV2`, `FiniteRepeat`, `Map`, y finalmente `Range`.

En general, una transformación lenta corresponde a aquella cuyos eventos son largos, pero sus eventos de entrada son cortos. Algunos ejemplos son los siguientes.

Tenga en cuenta que la transformación final (más externa) en la mayoría de las canalizaciones de entrada del host es el evento `Iterator::Model`. La transformación Model es introducida automáticamente por el runtime `tf.data` y se usa para instrumentar y autoajustar el rendimiento de la canalización de entrada.

Si su trabajo está usando una [estrategia de distribución](https://www.tensorflow.org/guide/distributed_training), el visor de seguimiento contendrá eventos adicionales correspondientes a la canalización de entrada del dispositivo. La transformación más externa de la canalización del dispositivo (anidada bajo `IteratorGetNextOp::DoCompute` o `IteratorGetNextAsOptionalOp::DoCompute`) será un evento `Iterator::Prefetch` con un evento `Iterator::Generator` ascendente. Puede encontrar la canalización host correspondiente buscando eventos `Iterator::Model`.

##### Ejemplo 1

![imagen](images/data_performance_analysis/example_1_cropped.png "Example 1")

La captura de pantalla anterior se genera a partir de la siguiente canalización de entrada:

```python
dataset = tf.data.TFRecordDataset(filename)
dataset = dataset.map(parse_record)
dataset = dataset.batch(32)
dataset = dataset.repeat()
```

En la captura de pantalla, observe que (1) los eventos de `Iterator::Map` son largos, pero (2) sus eventos de entrada (`Iterator::FlatMap`) retornan rápidamente. Esto sugiere que la transformación secuencial Map es el cuello de botella.

Observe que en la captura de pantalla, el evento `InstantiatedCapturedFunction::Run` corresponde al tiempo que tarda en ejecutarse la función mapear.

##### Ejemplo 2

![imagen](https://github.com/tensorflow/docs-l10n/blob/master/site/es-419/guide/images/data_performance_analysis/example_2_cropped.png?raw=true)

La captura de pantalla anterior se genera a partir de la siguiente canalización de entrada:

```python
dataset = tf.data.TFRecordDataset(filename)
dataset = dataset.map(parse_record, num_parallel_calls=2)
dataset = dataset.batch(32)
dataset = dataset.repeat()
```

Este ejemplo es similar al anterior, pero usa ParallelMap en lugar de Map. Aquí observamos que (1) los eventos `Iterator::ParallelMap` son largos, pero (2) sus eventos de entrada `Iterator::FlatMap` (que están en un hilo diferente, ya que ParallelMap es asíncrono) son cortos. Esto sugiere que la transformación ParallelMap es el cuello de botella.

#### Resolver el cuello de botella

##### Conjuntos de datos de origen

Si ha identificado una fuente de conjuntos de datos como el cuello de botella, como leer de archivos TFRecord, puede mejorar el rendimiento paralelizando la extracción de datos. Para ello, asegúrese de que sus datos están fragmentados en varios archivos y use `tf.data.Dataset.interleave` con el parámetro `num_parallel_calls` en el valor `tf.data.AUTOTUNE`. Si el determinismo no es importante para su programa, puede mejorar aún más el rendimiento configurando el indicador `deterministic=False` en `tf.data.Dataset.interleave` a partir de TF 2.2. Por ejemplo, si está leyendo de TFRecords, puede hacer lo siguiente:

```python
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.interleave(tf.data.TFRecordDataset,
  num_parallel_calls=tf.data.AUTOTUNE,
  deterministic=False)
```

Observe que los archivos fragmentados deben ser razonablemente grandes para amortizar la sobrecarga de abrir un archivo. Para más detalles sobre la extracción paralela de datos, consulte [esta sección](https://www.tensorflow.org/guide/data_performance#parallelizing_data_extraction) de la guía de rendimiento de `tf.data`.

##### Conjuntos de datos de transformación

Si ha identificado una transformación `tf.data` intermedia como el cuello de botella, puede solucionarla paralelizando la transformación o [almacenando en caché el cálculo](https://www.tensorflow.org/guide/data_performance#caching) si sus datos caben en memoria y es apropiado. Algunas transformaciones como `Map` tienen homólogos paralelos; la guía de rendimiento <a href="https://www.tensorflow.org/guide/data_performance#parallelizing_data_transformation" data-md-type="link">`tf.data` demuestra</a> cómo paralelizarlos. Otras transformaciones, como `Filter`, `Unbatch`, y `Batch` son inherentemente secuenciales; puede paralelizarlas introduciendo "paralelismo externo". Por ejemplo, supongamos que su canalización de entrada tiene inicialmente el siguiente aspecto, con `Batch` como cuello de botella:

```python
filenames = tf.data.Dataset.list_files(file_path, shuffle=is_training)
dataset = filenames_to_dataset(filenames)
dataset = dataset.batch(batch_size)
```

Puede introducir "paralelismo externo" ejecutando múltiples copias de la canalización de entrada sobre entradas fragmentadas y combinando los resultados:

```python
filenames = tf.data.Dataset.list_files(file_path, shuffle=is_training)

def make_dataset(shard_index):
  filenames = filenames.shard(NUM_SHARDS, shard_index)
  dataset = filenames_to_dataset(filenames)
  Return dataset.batch(batch_size)

indices = tf.data.Dataset.range(NUM_SHARDS)
dataset = indices.interleave(make_dataset,
                             num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

## Recursos adicionales

- [Guía de rendimiento de tf.data](https://www.tensorflow.org/guide/data_performance) sobre cómo escribir canalizaciones de entrada de `tf.data` de rendimiento alto
- [Vídeo "Dentro de TensorFlow": mejores prácticas para `tf.data`](https://www.youtube.com/watch?v=ZnukSLKEw34)
- [Guía de Perfilador](https://www.tensorflow.org/guide/profiler)
- [Tutorial del Perfilador con colab](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)
