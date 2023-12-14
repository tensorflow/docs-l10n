# El componente de canalización Transform TFX

El componente de canalización Transform TFX ejecuta una ingeniería de características en tf.Examples emitidos desde un componente [ExampleGen](examplegen.md), utilizando un esquema de datos creado por un componente [SchemaGen](schemagen.md), y emite tanto un SavedModel como estadísticas sobre datos previos y posteriores a la transformación. Cuando se ejecuta, SavedModel acepta tf.Examples emitidos desde un componente ExampleGen y emite los datos de características transformados.

- Consume: tf.Examples de un componente de ExampleGen y un esquema de datos de un componente de SchemaGen.
- Emite: un SavedModel para un componente Trainer, estadísticas previas y posteriores a la transformación.

## Cómo configurar un componente Transform

Una vez que haya escrito su `preprocessing_fn`, debe definirse en un módulo de Python que luego se proporciona al componente Transform como entrada. Transform cargará este módulo y, luego, encontrará y usará la función denominada `preprocessing_fn` para construir la canalización de preprocesamiento.

```
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(_taxi_transform_module_file))
```

Además, es posible que desee proporcionar opciones para el cálculo de estadísticas previas o posteriores a la transformación basadas en [TFDV](tfdv.md). Para hacerlo, defina un `stats_options_updater_fn` dentro del mismo módulo.

## Transform y TensorFlow Transform

Transform hace un uso extensivo de [TensorFlow Transform](tft.md) para ejecutar ingeniería de funciones en su conjunto de datos. TensorFlow Transform es una gran herramienta para transformar datos de características antes de enviarlos a su modelo y como parte del proceso de entrenamiento. Las transformaciones de características comunes incluyen las siguientes:

- **Incorporación**: convertir características dispersas (como los ID de números enteros producidos por un vocabulario) en características densas al encontrar una asignación significativa desde el espacio de alta dimensión al espacio de baja dimensión. Consulte la [unidad de incorporaciones en el curso intensivo de aprendizaje automático](https://developers.google.com/machine-learning/crash-course/embedding) para acceder a una introducción a las incorporaciones.
- **Generación de vocabulario**: convertir cadenas u otras características no numéricas en números enteros al crear un vocabulario que asigne cada valor único a un número de identificación.
- **Normalización de valores**: transformar las características numéricas para que todas estén dentro de un rango similar.
- **Creación de depósitos**: convertir características de valor continuo en características categóricas asignando valores a cubos discretos.
- **Enriquecimiento de característica de texto**: producción de características a partir de datos sin procesar como tokens, n-gramas, entidades, sentimientos, etc., para enriquecer el conjunto de características.

TensorFlow Transform ofrece compatibilidad para estos y muchos otros tipos de transformaciones:

- Genere automáticamente un vocabulario a partir de sus datos más recientes.

- Ejecute transformaciones arbitrarias en sus datos antes de enviarlos a su modelo. TensorFlow Transform crea transformaciones en el grafo de TensorFlow para su modelo, de modo que las mismas transformaciones se realicen en el momento del entrenamiento y la inferencia. Puede definir transformaciones que hagan referencia a propiedades globales de los datos, como el valor máximo de una característica en todas las instancias de entrenamiento.

Puede transformar sus datos como quiera antes de ejecutar TFX. Pero si lo hace dentro de TensorFlow Transform, las transformaciones se convierten en parte del grafo de TensorFlow. Este enfoque ayuda a evitar el sesgo entrenamiento/servicio.

Las transformaciones dentro de su código de modelado usan FeatureColumns. Con FeatureColumns, puede definir agrupaciones, integraciones que emplean vocabularios predefinidos o cualquier otra transformación que se pueda definir sin consultar los datos.

Por el contrario, TensorFlow Transform se diseñó para transformaciones que requieren un paso completo sobre los datos para calcular valores que no se conocen de antemano. Por ejemplo, la generación de vocabulario requiere un paso completo sobre los datos.

Nota: Estos cálculos se implementan en [Apache Beam](https://beam.apache.org/) a nivel interno.

Además de usar Apache Beam para calcular valores, TensorFlow Transform permite a los usuarios incorporar estos valores en un grafo de TensorFlow, que luego se puede cargar en el grafo de entrenamiento. Por ejemplo, al normalizar características, la función `tft.scale_to_z_score` calculará la media y la desviación estándar de una característica, y también una representación, en un grafo de TensorFlow, de la función que resta la media y divide por la desviación estándar. Al emitir un grafo de TensorFlow, no solo estadísticas, TensorFlow Transform simplifica el proceso de creación de su canalización de preprocesamiento.

Dado que el preprocesamiento se expresa como un grafo, se puede llevar a cabo en el servidor y se garantiza que será coherente entre el entrenamiento y el servicio. Esta coherencia elimina una fuente de sesgo entrenamiento/servicio.

TensorFlow Transform permite a los usuarios especificar su canalización de preprocesamiento utilizando el código de TensorFlow. Esto significa que una canalización se construye de la misma manera que un grafo de TensorFlow. Si solo se usaran operaciones de TensorFlow en este grafo, la canalización sería un mapa puro que acepta lotes de entrada y devuelve lotes de salida. Una canalización de este tipo sería equivalente a colocar este grafo dentro de su `input_fn` cuando use la API `tf.Estimator`. Para especificar operaciones de paso completo, como el cálculo de cuantiles, TensorFlow Transform proporciona funciones especiales llamadas `analyzers` que parecen operaciones de TensorFlow, pero que en realidad especifican un cálculo diferido que realizará Apache Beam y la salida se insertará en el grafo como una constante. Mientras que una operación ordinaria de TensorFlow tomará un solo lote como entrada, hará algunos cálculos solo en ese lote y emitirá un lote, `analyzer` ejecutará una reducción global (implementada en Apache Beam) en todos los lotes y devolverá el resultado.

Si se combinan las operaciones comunes de TensorFlow y los analizadores de TensorFlow Transform, los usuarios pueden crear canalizaciones complejas para preprocesar sus datos. Por ejemplo, la función `tft.scale_to_z_score` toma un tensor de entrada y devuelve ese tensor normalizado para tener una media de `0` y una varianza de `1`. Para esto, llame a los analizadores `mean` y `var` a nivel interno, que generarán efectivamente constantes en el grafo iguales a la media y la varianza del tensor de entrada. Luego utilizará las operaciones de TensorFlow para restar la media y dividirla por la desviación estándar.

## `preprocessing_fn` de TensorFlow Transform

El componente TFX Transform simplifica el uso de Transform al manejar las llamadas de la API relacionadas con la lectura y escritura de datos, y al escribir el SavedModel de salida en el disco. Como usuario de TFX, solo tiene que definir una única función llamada `preprocessing_fn`. En `preprocessing_fn`, se definen una serie de funciones que manipulan el dictado de entrada de los tensores para producir el dictado de salida de los tensores. Puede encontrar funciones ayudantes como scale_to_0_1 y Compute_and_apply_vocabulary en la [API de TensorFlow Transform](/tfx/transform/api_docs/python/tft) o usar funciones comunes de TensorFlow como se muestra a continuación.

```python
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}
  for key in _DENSE_FLOAT_FEATURE_KEYS:
    # If sparse make it dense, setting nan's to 0 or '', and apply zscore.
    outputs[_transformed_name(key)] = transform.scale_to_z_score(
        _fill_in_missing(inputs[key]))

  for key in _VOCAB_FEATURE_KEYS:
    # Build a vocabulary for this feature.
    outputs[_transformed_name(
        key)] = transform.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]),
            top_k=_VOCAB_SIZE,
            num_oov_buckets=_OOV_SIZE)

  for key in _BUCKET_FEATURE_KEYS:
    outputs[_transformed_name(key)] = transform.bucketize(
        _fill_in_missing(inputs[key]), _FEATURE_BUCKET_COUNT)

  for key in _CATEGORICAL_FEATURE_KEYS:
    outputs[_transformed_name(key)] = _fill_in_missing(inputs[key])

  # Was this passenger a big tipper?
  taxi_fare = _fill_in_missing(inputs[_FARE_KEY])
  tips = _fill_in_missing(inputs[_LABEL_KEY])
  outputs[_transformed_name(_LABEL_KEY)] = tf.where(
      tf.is_nan(taxi_fare),
      tf.cast(tf.zeros_like(taxi_fare), tf.int64),
      # Test if the tip was > 20% of the fare.
      tf.cast(
          tf.greater(tips, tf.multiply(taxi_fare, tf.constant(0.2))), tf.int64))

  return outputs
```

### Explicación de las entradas a preprocessing_fn

`preprocessing_fn` describe una serie de operaciones en tensores (es decir, `Tensor`, `SparseTensor` o `RaggedTensor`). Para definir `preprocessing_fn` correctamente es necesario comprender cómo se representan los datos como tensores. La entrada a `preprocessing_fn` está determinada por el esquema. Un [proto `Schema`](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto#L72) finalmente se convierte en una "especificación de característica" (a veces denominada "especificación de parseo") que se usa para el parseo de datos; consulte más detalles sobre la lógica de conversión [aquí](https://github.com/tensorflow/metadata/blob/master/tfx_bsl/docs/schema_interpretation.md).

## Cómo usar TensorFlow Transform para manejar etiquetas de cadenas

Por lo general, uno quiere aprovechar TensorFlow Transform para generar un vocabulario y aplicar ese vocabulario para convertir cadenas en números enteros. Al seguir este flujo de trabajo, la `input_fn` construida en el modelo generará la cadena entera. Sin embargo, las etiquetas son una excepción, porque para que el modelo pueda asignar las etiquetas de salida (enteros) a cadenas, el modelo necesita una `input_fn` para generar una etiqueta de cadena, junto con una lista de posibles valores de la etiqueta. Por ejemplo, si las etiquetas son `cat` y `dog`, entonces la salida de `input_fn` deberían ser estas cadenas sin formato, y las claves `["cat", "dog"]` deben pasarse al estimador como parámetro (consulte los detalles a continuación).

Con el fin de manejar la asignación de etiquetas de cadena a enteros, deberá usar TensorFlow Transform para generar un vocabulario. Lo demostramos en el siguiente fragmento de código:

```python
def _preprocessing_fn(inputs):
  """Preprocess input features into transformed features."""

  ...


  education = inputs[features.RAW_LABEL_KEY]
  _ = tft.vocabulary(education, vocab_filename=features.RAW_LABEL_KEY)

  ...
```

La función de preprocesamiento anteriormente mencionada toma la característica de entrada sin formato (que también se devolverá como parte de la salida de la función de preprocesamiento) y llama a `tft.vocabulary` en ella. Esto, como resultado, genera un vocabulario para `education` al que se puede acceder desde el modelo.

En el ejemplo también se muestra cómo transformar una etiqueta y luego generar un vocabulario para la etiqueta transformada. En particular, se toma la etiqueta sin procesar `education` y se convierten todas las etiquetas excepto las 5 principales (por frecuencia) a `UNKNOWN`, sin convertir la etiqueta a un número entero.

En el código del modelo, el clasificador debe recibir el vocabulario generado por `tft.vocabulary` como argumento `label_vocabulary`. Para hacerlo, primero se lee este vocabulario como una lista con una función ayudante. Esto se muestra en el siguiente fragmento de código. Tenga en cuenta que el código de ejemplo usa la etiqueta transformada descrita anteriormente, pero aquí mostramos el código para usar la etiqueta sin procesar.

```python
def create_estimator(pipeline_inputs, hparams):

  ...

  tf_transform_output = trainer_util.TFTransformOutput(
      pipeline_inputs.transform_dir)

  # vocabulary_by_name() returns a Python list.
  label_vocabulary = tf_transform_output.vocabulary_by_name(
      features.RAW_LABEL_KEY)

  return tf.contrib.learn.DNNLinearCombinedClassifier(
      ...
      n_classes=len(label_vocab),
      label_vocabulary=label_vocab,
      ...)
```

## Cómo configurar estadísticas previas y posteriores a la transformación

Como se mencionó anteriormente, el componente Transform invoca TFDV para calcular estadísticas previas y posteriores a la transformación. TFDV toma como entrada un objeto [StatsOptions](https://github.com/tensorflow/data-validation/blob/master/tensorflow_data_validation/statistics/stats_options.py) opcional. Es posible que los usuarios deseen configurar este objeto para habilitar ciertas estadísticas adicionales (por ejemplo, estadísticas de PNL) o para establecer umbrales que se validen (por ejemplo, frecuencia de token mínima/máxima). Para hacerlo, defina `stats_options_updater_fn` en el archivo del módulo.

```python
def stats_options_updater_fn(stats_type, stats_options):
  ...
  if stats_type == stats_options_util.StatsType.PRE_TRANSFORM:
    # Update stats_options to modify pre-transform statistics computation.
    # Most constraints are specified in the schema which can be accessed
    # via stats_options.schema.
  if stats_type == stats_options_util.StatsType.POST_TRANSFORM
    # Update stats_options to modify post-transform statistics computation.
    # Most constraints are specified in the schema which can be accessed
    # via stats_options.schema.
  return stats_options
```

Las estadísticas posteriores a la transformación a menudo se benefician del conocimiento del vocabulario que se usa para preprocesar una característica. El nombre del vocabulario se asigna a StatsOptions (y, por tanto, a TFDV) para cada vocabulario generado por TFT. Además, se pueden agregar asignaciones para vocabularios creados externamente (i) si se modifica directamente el diccionario `vocab_paths` dentro de StatsOptions o (ii) si se usa `tft.annotate_asset`.
