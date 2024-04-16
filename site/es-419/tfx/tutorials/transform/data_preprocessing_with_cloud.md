# Preprocesamiento de datos para aprendizaje automático con Google Cloud

En este tutorial se muestra cómo usar [TensorFlow Transform](https://github.com/tensorflow/transform){: .external} (la biblioteca `tf.Transform`) para implementar el preprocesamiento de datos para el aprendizaje automático (ML). La biblioteca `tf.Transform` de TensorFlow le permite definir transformaciones de datos tanto a nivel de instancia como de paso completo a través de canalizaciones de preprocesamiento de datos. Estas canalizaciones se ejecutan de manera eficiente con [Apache Beam](https://beam.apache.org/){: .external} y crean como subproductos un grafo de TensorFlow para aplicar las mismas transformaciones durante la predicción que cuando se sirve el modelo.

En este tutorial se ofrece un ejemplo de un extremo a otro con [Dataflow](https://cloud.google.com/dataflow/docs){: .external } como ejecutor para Apache Beam. Se supone que está familiarizado con [BigQuery](https://cloud.google.com/bigquery/docs){: .external }, Dataflow, [Vertex AI](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform){: .external } y la API TensorFlow [Keras](https://www.tensorflow.org/guide/keras/overview). También se supone que tiene cierta experiencia en el uso de Jupyter Notebooks y [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction){: .external }.

En este tutorial también se supone que está familiarizado con los conceptos de tipos, desafíos y opciones de preprocesamiento en Google Cloud, como se describe en [Preprocesamiento de datos para aprendizaje automático: opciones y recomendaciones](../../guide/tft_bestpractices).

## Objetivos

- Implemente la canalización de Apache Beam con ayuda de la biblioteca `tf.Transform`.
- Ejecute la canalización en Dataflow.
- Implemente el modelo de TensorFlow con la biblioteca `tf.Transform`.
- Entrene y use el modelo para predicciones.

## Costos

Este tutorial usa los siguientes componentes sujetos a facturación de Google Cloud:

- [Vertex AI](https://cloud.google.com/vertex-ai/pricing){: .external}
- [Cloud Storage](https://cloud.google.com/storage/pricing){: .external}
- [BigQuery](https://cloud.google.com/bigquery/pricing){: .external}
- [Dataflow](https://cloud.google.com/dataflow/pricing){: .external}

<!-- This doc uses plain text cost information because the pricing calculator is pre-configured -->

Para calcular el costo de ejecución de este tutorial, suponiendo que use todos los recursos durante un día completo, use la [calculadora de precios](/products/calculator/#id=fad4d8-dd68-45b8-954e-5a56a5d148) preconfigurada {: .external }.

## Antes de empezar

1. En la consola de Google Cloud, en la página de selección de proyectos, seleccione o [cree un proyecto de Google Cloud](https://cloud.google.com/resource-manager/docs/creating-managing-projects).

Nota: Si no planea conservar los recursos que cree en este procedimiento, cree un proyecto en lugar de seleccionar un proyecto existente. Después de finalizar estos pasos, puede eliminar el proyecto, lo que elimina todos los recursos asociados con el proyecto.

[Vaya al selector de proyectos](https://console.cloud.google.com/projectselector2/home/dashboard){: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

1. Asegúrese de que la facturación esté habilitada para su proyecto de Cloud. Aprenda a [verificar si la facturación está habilitada en un proyecto](https://cloud.google.com/billing/docs/how-to/verify-billing-enabled).

2. Habilite las API de Dataflow, Vertex AI y Notebooks. [Habilite las API](https://console.cloud.google.com/flows/enableapi?apiid=dataflow,aiplatform.googleapis.com,notebooks.googleapis.com){: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

## Blocs de notas Jupyter para esta solución

Los siguientes blocs de notas Jupyter muestran el ejemplo de implementación:

- [El bloc de notas 1](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/blogs/babyweight_tft/babyweight_tft_keras_01.ipynb){: .external } cubre el preprocesamiento de datos. Los detalles se proporcionan más adelante en la sección [Implementación de la canalización de Apache Beam](#implement-the-apache-beam-pipeline).
- [El bloc de notas 2](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/blogs/babyweight_tft/babyweight_tft_keras_02.ipynb){: .external } cubre el entrenamiento de modelos. Los detalles se proporcionan en la sección [Implementación del modelo de TensorFlow](#implement-the-tensorflow-model) más adelante.

En las siguientes secciones, clonará estos blocs de notas y luego los ejecutará para aprender cómo funciona el ejemplo de implementación.

## Lance una instancia de blocs de notas administrada por el usuario

1. En la consola de Google Cloud, vaya a la página **Vertex AI Workbench**.

    [Vaya a Workbench](https://console.cloud.google.com/ai-platform/notebooks/list/instances){: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

2. En la pestaña **Blocs de notas administrados por el usuario**, haga clic en **+Nuevo bloc de notas**.

3. Seleccione **TensorFlow Enterprise 2.8 (con LTS) sin GPU** para el tipo de instancia.

4. Haga clic en **Crear**.

Después de crear el bloc de notas, espere a que el proxy de JupyterLab termine de inicializarse. Cuando esté listo, se muestra **Abrir JupyterLab** junto al nombre del bloc de notas.

## Clone el bloc de notas

1. En la **pestaña Blocs de notas administrados por el usuario**, junto al nombre del bloc de notas, haga clic en **Abrir JupyterLab**. La interfaz de JupyterLab se abre en una nueva pestaña.

    Si JupyterLab muestra un cuadro de diálogo **Compilación recomendada**, haga clic en **Cancelar** para rechazar la compilación sugerida.

2. En la pestaña **Iniciador**, haga clic en **Terminal**.

3. En la ventana de la terminal, clone el bloc de notas:

    ```sh
    git clone https://github.com/GoogleCloudPlatform/training-data-analyst
    ```

## Implemente la canalización Apache Beam

En esta sección y en la siguiente [Ejecute la canalización en Dataflow](#run-the-pipeline-in-dataflow){: track-type="solution" track-name="internalLink" track-metadata-position="body" } brindan una descripción general y un contexto para el Bloc de notas 1. El bloc de notas ofrece un ejemplo práctico que describe cómo utilizar la biblioteca `tf.Transform` para preprocesar datos. En este ejemplo se usa el conjunto de datos Natality, que se usa para predecir el peso de los bebés en función de varios datos de entrada. Los datos se almacenan en la tabla pública de [natalidad](https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=samples&t=natality&page=table&_ga=2.267763789.2122871960.1676620306-376763843.1676620306){: target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" } en BigQuery.

### Ejecute el bloc de notas 1

1. En la interfaz de JupyterLab, haga clic en **Archivo &gt; Abrir desde ruta** y luego ingrese la siguiente ruta:

    ```sh
    training-data-analyst/blogs/babyweight_tft/babyweight_tft_keras_01.ipynb
    ```

2. Haga clic en **Editar &gt; Borrar todas las salidas**.

3. En la sección **Instalar paquetes requeridos**, ejecute la primera celda para ejecutar el comando `pip install apache-beam`.

    La última parte de la salida es la siguiente:

    ```none{:.devsite-disable-click-to-copy}
    Successfully installed ...
    ```

    Puede ignorar los errores de dependencia en la salida. No es necesario reiniciar el núcleo todavía.

4. Ejecute la segunda celda para ejecutar el comando `pip install tensorflow-transform`. La última parte de la salida es la siguiente:

    ```none{:.devsite-disable-click-to-copy}
    Successfully installed ...
    Note: you may need to restart the kernel to use updated packages.
    ```

    Puede ignorar los errores de dependencia en la salida.

5. Haga clic en **Núcleo&gt; Reiniciar núcleo** .

6. Ejecute las celdas en las secciones **Confirmar los paquetes instalados** y **Crear setup.py para instalar paquetes en contenedores de Dataflow**.

7. En la sección **Establecer marcas globales**, junto a `PROJECT` y `BUCKET`, reemplace `your-project` con su ID de proyecto en Cloud y luego ejecute la celda.

8. Ejecute todas las celdas restantes hasta la última celda del bloc de notas. Para obtener información sobre qué hacer en cada celda, consulte las instrucciones en el bloc de notas.

### Descripción general de la canalización

En el ejemplo del bloc de notas, Dataflow ejecuta la canalización `tf.Transform` a escala para preparar los datos y producir los artefactos de transformación. Las secciones posteriores de este documento describen las funciones que ejecutan en cada paso del proceso. Los pasos generales del proceso son los siguientes:

1. Leer datos de entrenamiento de BigQuery.
2. Analizar y transformar datos de entrenamiento con ayuda de la biblioteca `tf.Transform`.
3. Escribir datos de entrenamiento transformados en Cloud Storage en el formato [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord){: target="external" class="external" track-type="solution" track-name="externalLink" track-metadata-position="body" }.
4. Leer datos de evaluación de BigQuery.
5. Transformar los datos de evaluación con el grafo `transform_fn` que se produce en el paso 2.
6. Escribir datos de entrenamiento transformados en Cloud Storage en formato TFRecord.
7. Escribir artefactos de transformación en Cloud Storage que se usarán más adelante para crear y exportar el modelo.

El siguiente ejemplo muestra el código Python para la canalización general. En las próximas secciones se proporcionan explicaciones y listados de códigos para cada paso.

```py{:.devsite-disable-click-to-copy}
def run_transformation_pipeline(args):

    pipeline_options = beam.pipeline.PipelineOptions(flags=[], **args)

    runner = args['runner']
    data_size = args['data_size']
    transformed_data_location = args['transformed_data_location']
    transform_artefact_location = args['transform_artefact_location']
    temporary_dir = args['temporary_dir']
    debug = args['debug']

    # Instantiate the pipeline
    with beam.Pipeline(runner, options=pipeline_options) as pipeline:
        with impl.Context(temporary_dir):

            # Preprocess train data
            step = 'train'
            # Read raw train data from BigQuery
            raw_train_dataset = read_from_bq(pipeline, step, data_size)
            # Analyze and transform raw_train_dataset
            transformed_train_dataset, transform_fn = analyze_and_transform(raw_train_dataset, step)
            # Write transformed train data to sink as tfrecords
            write_tfrecords(transformed_train_dataset, transformed_data_location, step)

            # Preprocess evaluation data
            step = 'eval'
            # Read raw eval data from BigQuery
            raw_eval_dataset = read_from_bq(pipeline, step, data_size)
            # Transform eval data based on produced transform_fn
            transformed_eval_dataset = transform(raw_eval_dataset, transform_fn, step)
            # Write transformed eval data to sink as tfrecords
            write_tfrecords(transformed_eval_dataset, transformed_data_location, step)

            # Write transformation artefacts
            write_transform_artefacts(transform_fn, transform_artefact_location)

            # (Optional) for debugging, write transformed data as text
            step = 'debug'
            # Write transformed train data as text if debug enabled
            if debug == True:
                write_text(transformed_train_dataset, transformed_data_location, step)
```

### Leer datos de entrenamiento sin procesar de BigQuery{: id="read_raw_training_data"}

El primer paso consiste en leer los datos de entrenamiento sin procesar de BigQuery mediante la función `read_from_bq`. Esta función devuelve un objeto `raw_dataset` extraído de BigQuery. Pasa un valor `data_size` y pasa un valor `step` de `train` o `eval`. La consulta de origen de BigQuery se construye mediante la función `get_source_query`, como se muestra en el siguiente ejemplo:

```py{:.devsite-disable-click-to-copy}
def read_from_bq(pipeline, step, data_size):

    source_query = get_source_query(step, data_size)
    raw_data = (
        pipeline
        | '{} - Read Data from BigQuery'.format(step) >> beam.io.Read(
                           beam.io.BigQuerySource(query=source_query, use_standard_sql=True))
        | '{} - Clean up Data'.format(step) >> beam.Map(prep_bq_row)
    )

    raw_metadata = create_raw_metadata()
    raw_dataset = (raw_data, raw_metadata)
    return raw_dataset
```

Antes de realizar el preprocesamiento `tf.Transform`, es posible que deba realizar el procesamiento típico basado en Apache Beam, incluido el procesamiento de mapas, filtros, grupos y ventanas. En el ejemplo, el código limpia los registros leídos de BigQuery mediante el método `beam.Map(prep_bq_row)`, donde `prep_bq_row` es una función personalizada. Esta función personalizada convierte el código numérico de una característica categórica en etiquetas legibles por humanos.

Además, para usar la biblioteca `tf.Transform` para analizar y transformar el objeto `raw_data` extraído de BigQuery, debe crear un objeto `raw_dataset`, que es una tupla de objetos `raw_data` y `raw_metadata`. El objeto `raw_metadata` se crea con ayuda de la función `create_raw_metadata`, de la siguiente manera:

```py{:.devsite-disable-click-to-copy}
CATEGORICAL_FEATURE_NAMES = ['is_male', 'mother_race']
NUMERIC_FEATURE_NAMES = ['mother_age', 'plurality', 'gestation_weeks']
TARGET_FEATURE_NAME = 'weight_pounds'

def create_raw_metadata():

    feature_spec = dict(
        [(name, tf.io.FixedLenFeature([], tf.string)) for name in CATEGORICAL_FEATURE_NAMES] +
        [(name, tf.io.FixedLenFeature([], tf.float32)) for name in NUMERIC_FEATURE_NAMES] +
        [(TARGET_FEATURE_NAME, tf.io.FixedLenFeature([], tf.float32))])

    raw_metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(feature_spec))

    return raw_metadata
```

Cuando ejecuta la celda en el bloc de notas que sigue inmediatamente a la celda que define este método, se muestra el contenido del objeto `raw_metadata.schema`. Incluye las siguientes columnas:

- `gestation_weeks` (tipo: `FLOAT`)
- `is_male` (tipo: `BYTES`)
- `mother_age` (tipo: `FLOAT`)
- `mother_race` (tipo: `BYTES`)
- `plurality` (tipo: `FLOAT`)
- `weight_pounds` (tipo: `FLOAT`)

### Transforme los datos de entrenamiento sin procesar

Imagine que desea aplicar transformaciones de preprocesamiento típicas a las características sin procesar de entrada de los datos de entrenamiento para prepararlos para ML. Estas transformaciones incluyen operaciones de paso completo y de nivel de instancia, como se muestra en la siguiente tabla:

<table>
<thead>
  <tr>
    <th>Característica de entrada</th>
    <th>Transformación</th>
    <th>Estadísticas necesarias</th>
    <th>Tipo</th>
    <th>Característica de salida</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><code>weight_pound</code></td>
    <td>Ninguna</td>
    <td>Ninguna</td>
    <td>NA</td>
    <td><code>weight_pound</code></td>
  </tr>
  <tr>
    <td><code>mother_age</code></td>
    <td>Normalización</td>
    <td>media, var</td>
    <td>paso completo</td>
    <td><code>mother_age_normalized</code></td>
  </tr>
  <tr>
    <td><code>mother_age</code></td>
    <td>Creación de depósitos de igual tamaño</td>
    <td>cuantiles</td>
    <td>paso completo</td>
    <td><code>mother_age_bucketized</code></td>
  </tr>
  <tr>
    <td><code>mother_age</code></td>
    <td>Cálculo del registro</td>
    <td>Ninguna</td>
    <td>nivel de instancia</td>
    <td>
        <code>mother_age_log</code>
    </td>
  </tr>
  <tr>
    <td><code>plurality</code></td>
    <td>Indicación de si se trata de bebés únicos o múltiples</td>
    <td>Ninguna</td>
    <td>nivel de instancia</td>
    <td><code>is_multiple</code></td>
  </tr>
  <tr>
    <td><code>is_multiple</code></td>
    <td>Conversión de valores nominales a índice numérico</td>
    <td>vocab</td>
    <td>paso completo</td>
    <td><code>is_multiple_index</code></td>
  </tr>
  <tr>
    <td><code>gestation_weeks</code></td>
    <td>Escala entre 0 y 1</td>
    <td>mín, máx</td>
    <td>paso completo</td>
    <td><code>gestation_weeks_scaled</code></td>
  </tr>
  <tr>
    <td><code>mother_race</code></td>
    <td>Conversión de valores nominales a índice numérico</td>
    <td>vocab</td>
    <td>paso completo</td>
    <td><code>mother_race_index</code></td>
  </tr>
  <tr>
    <td><code>is_male</code></td>
    <td>Conversión de valores nominales a índice numérico</td>
    <td>vocab</td>
    <td>paso completo</td>
    <td><code>is_male_index</code></td>
  </tr>
</tbody>
</table>

Estas transformaciones se implementan en una función `preprocess_fn`, que espera un diccionario de tensores (`input_features`) y devuelve un diccionario de características procesadas (`output_features`).

El siguiente código muestra la implementación de la función `preprocess_fn`, con las API de transformación de paso completo `tf.Transform` (con el prefijo `tft.`) y las operacioes de TensorFlow a nivel de instancia (con el prefijo `tf.`):

```py{:.devsite-disable-click-to-copy}
def preprocess_fn(input_features):

    output_features = {}

    # target feature
    output_features['weight_pounds'] = input_features['weight_pounds']

    # normalization
    output_features['mother_age_normalized'] = tft.scale_to_z_score(input_features['mother_age'])

    # scaling
    output_features['gestation_weeks_scaled'] =  tft.scale_to_0_1(input_features['gestation_weeks'])

    # bucketization based on quantiles
    output_features['mother_age_bucketized'] = tft.bucketize(input_features['mother_age'], num_buckets=5)

    # you can compute new features based on custom formulas
    output_features['mother_age_log'] = tf.math.log(input_features['mother_age'])

    # or create flags/indicators
    is_multiple = tf.as_string(input_features['plurality'] > tf.constant(1.0))

    # convert categorical features to indexed vocab
    output_features['mother_race_index'] = tft.compute_and_apply_vocabulary(input_features['mother_race'], vocab_filename='mother_race')
    output_features['is_male_index'] = tft.compute_and_apply_vocabulary(input_features['is_male'], vocab_filename='is_male')
    output_features['is_multiple_index'] = tft.compute_and_apply_vocabulary(is_multiple, vocab_filename='is_multiple')

    return output_features
```

El [marco](https://github.com/tensorflow/transform){: .external } `tf.Transform` tiene varias otras transformaciones además de las del ejemplo anterior, incluidas las que se enumeran en la siguiente tabla:

<table>
<thead>
  <tr>
  <th>Transformación</th>
  <th>Se aplica a</th>
  <th>Descripción</th>
  </tr>
</thead>
<tbody>
    <tr>
    <td><code>scale_by_min_max</code></td>
    <td>Características numéricas</td>
    <td>       Escala una columna numérica al rango [ <code>output_min</code>, <code>output_max</code>]</td>
  </tr>
  <tr>
    <td><code>scale_to_0_1</code></td>
    <td>Características numéricas</td>
    <td>       Devuelve una columna que es la columna de entrada escalada para tener un rango [ <code>0</code>, <code>1</code>]</td>
  </tr>
  <tr>
    <td><code>scale_to_z_score</code></td>
    <td>Características numéricas</td>
    <td>Devuelve una columna estandarizada con media 0 y varianza 1</td>
  </tr>
  <tr>
    <td><code>tfidf</code></td>
    <td>Características de texto</td>
    <td>       Asigna los términos en <i>x</i> a su frecuencia de términos * frecuencia inversa del documento</td>
  </tr>
  <tr>
    <td><code>compute_and_apply_vocabulary</code></td>
    <td>Características categóricas</td>
    <td>       Genera un vocabulario para una característica categórica y lo asigna a un número entero con este vocabulario</td>
  </tr>
  <tr>
    <td><code>ngrams</code></td>
    <td>Características de texto</td>
    <td>Crea un <code>SparseTensor</code> de n-gramas</td>
  </tr>
  <tr>
    <td><code>hash_strings</code></td>
    <td>Características categóricas</td>
    <td>Convierte cadenas en cubos</td>
  </tr>
  <tr>
    <td><code>pca</code></td>
    <td>Características numéricas</td>
    <td>Calcula PCA en el conjunto de datos mediante covarianza sesgada</td>
  </tr>
  <tr>
    <td><code>bucketize</code></td>
    <td>Características numéricas</td>
    <td>       Devuelve una columna agrupada en cubos de igual tamaño (basada en cuantiles), con un índice de cubo asignado a cada entrada</td>
  </tr>
</tbody>
</table>

Para aplicar las transformaciones implementadas en la función `preprocess_fn` al objeto `raw_train_dataset` que se produjo en el paso anterior de la canalización, se utiliza el método `AnalyzeAndTransformDataset`. Este método espera el objeto `raw_dataset` como entrada, aplica la función `preprocess_fn` y produce el objeto `transformed_dataset` y el grafo `transform_fn`. El siguiente código ilustra este procesamiento:

```py{:.devsite-disable-click-to-copy}
def analyze_and_transform(raw_dataset, step):

    transformed_dataset, transform_fn = (
        raw_dataset
        | '{} - Analyze & Transform'.format(step) >> tft_beam.AnalyzeAndTransformDataset(
            preprocess_fn, output_record_batches=True)
    )

    return transformed_dataset, transform_fn
```

Las transformaciones se aplican a los datos sin procesar en dos fases: la fase de análisis y la fase de transformación. En la Figura 3 más adelante en este documento se muestra cómo el método `AnalyzeAndTransformDataset` se descompone en el método `AnalyzeDataset` y el método `TransformDataset`.

#### La fase de análisis

En la fase de análisis, los datos de entrenamiento sin procesar se analizan en un proceso de paso completo para calcular las estadísticas necesarias para las transformaciones. Esto incluye el cálculo de la media, la varianza, el mínimo, el máximo, los cuantiles y el vocabulario. El proceso de análisis espera un conjunto de datos sin procesar (datos sin procesar más metadatos sin procesar) y produce dos salidas:

- `transform_fn`: un grafo de TensorFlow que contiene las estadísticas calculadas de la fase de análisis y la lógica de transformación (que usa las estadísticas) como operaciones a nivel de instancia. Como se analiza más adelante en [Guarde el grafo](#save_the_graph){: track-type="solution" track-name="internalLink" track-metadata-position="body" }, el grafo `transform_fn` se guarda para adjuntarlo a la función `serving_fn` del modelo. Esto hace posible aplicar la misma transformación a los puntos de datos de predicción en línea.
- `transform_metadata`: un objeto que describe el esquema esperado de los datos después de la transformación.

La fase de análisis se ilustra en el siguiente diagrama, figura 1:

<figure id="tf-transform-analyze-phase">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-tf-transform-analyze-phase.svg"
    alt="The tf.Transform analyze phase.">
  <figcaption><b>Figure 1.</b> The <code>tf.Transform</code> analyze phase.</figcaption>
</figure>

Los [analizadores](https://github.com/tensorflow/transform/blob/master/tensorflow_transform/beam/analyzer_impls.py){: target="github" class="external" track-type="solution" track-name="gitHubLink" track-metadata-position="body" } `tf.Transform` incluyen `min`, `max`, `sum`, `size`, `mean`, `var`, `covariance`, `quantiles`, `vocabulary` y `pca`.

#### La fase de transformación

En la fase de transformación, el grafo `transform_fn` que se produce en la fase de análisis permite transformar los datos de entrenamiento sin procesar en un proceso a nivel de instancia para producir los datos de entrenamiento transformados. Los datos de entrenamiento transformados se emparejan con los metadatos transformados (que se producen en la fase de análisis) para generar el conjunto de datos `transformed_train_dataset`.

La fase de transformación se ilustra en el siguiente diagrama, figura 2:

<figure id="tf-transform-transform-phase">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-tf-transform-transform-phase.svg"
    alt="The tf.Transform transform phase.">
  <figcaption><b>Figure 2.</b> The <code>tf.Transform</code> transform phase.</figcaption>
</figure>

Para preprocesar las características, llama a las transformaciones `tensorflow_transform` requeridas (importadas como `tft` en el código) en su implementación de la función `preprocess_fn`. Por ejemplo, cuando llama a las operaciones `tft.scale_to_z_score`, la biblioteca `tf.Transform` traduce esta llamada a función en analizadores de media y varianza, calcula las estadísticas en la fase de análisis y luego aplica estas estadísticas para normalizar la característica numérica en la fase de transformación. Todo esto se hace automáticamente cuando se llama al método `AnalyzeAndTransformDataset(preprocess_fn)`.

La entidad `transformed_metadata.schema` producida por esta llamada incluye las siguientes columnas:

- `gestation_weeks_scaled` (tipo: `FLOAT`)
- `is_male_index` (tipo: `INT`, is_categorical: `True`)
- `is_multiple_index` (tipo: `INT`, is_categorical: `True`)
- `mother_age_bucketized` (tipo: `INT`, is_categorical: `True`)
- `mother_age_log` (tipo: `FLOAT`)
- `mother_age_normalized` (tipo: `FLOAT`)
- `mother_race_index` (tipo: `INT`, is_categorical: `True`)
- `weight_pounds` (tipo: `FLOAT`)

Como se explica en [Operaciones de preprocesamiento](data-preprocessing-for-ml-with-tf-transform-pt1#preprocessing_operations) en la primera parte de esta serie, la transformación de características convierte características categóricas en una representación numérica. Después de la transformación, las características categóricas se representan mediante valores enteros. En la entidad `transformed_metadata.schema`, la marca `is_categorical` para columnas de tipo `INT` indica si la columna representa una característica categórica o una característica numérica verdadera.

### Escriba datos de entrenamiento transformados{: id="step_3_write_transformed_training_data"}

Después de preprocesar los datos de entrenamiento con la función `preprocess_fn` a través de las fases de análisis y transformación, puede escribir los datos en un receptor para usarlos para entrenar el modelo de TensorFlow. Cuando se ejecuta la canalización de Apache Beam con Dataflow, el receptor es Cloud Storage. De lo contrario, el receptor es el disco local. Aunque puede escribir los datos como un archivo CSV de archivos formateados de ancho fijo, el formato de archivo recomendado para los conjuntos de datos de TensorFlow es el formato TFRecord. Este es un formato binario simple orientado a registros que consta de mensajes de búfer de protocolo `tf.train.Example`.

Cada registro `tf.train.Example` contiene una o más caracteristicas. Estos se convierten en tensores cuando se introducen en el modelo para su entrenamiento. El siguiente código escribe el conjunto de datos transformado en archivos TFRecord en la ubicación especificada:

```py{:.devsite-disable-click-to-copy}
def write_tfrecords(transformed_dataset, location, step):
    from tfx_bsl.coders import example_coder

    transformed_data, transformed_metadata = transformed_dataset
    (
        transformed_data
        | '{} - Encode Transformed Data'.format(step) >> beam.FlatMapTuple(
                            lambda batch, _: example_coder.RecordBatchToExamples(batch))
        | '{} - Write Transformed Data'.format(step) >> beam.io.WriteToTFRecord(
                            file_path_prefix=os.path.join(location,'{}'.format(step)),
                            file_name_suffix='.tfrecords')
    )
```

### Lea, transforme y escriba datos de evaluación

Después de transformar los datos de entrenamiento y producir el grafo `transform_fn`, puede usarlo para transformar los datos de evaluación. Primero, lee y limpia los datos de evaluación de BigQuery mediante la función `read_from_bq` descrita anteriormente en [Lea datos de entrenamiento sin procesar de BigQuery](#read-raw-training-data-from-bigquery){: track-type="solution" track-name="internalLink" track-metadata-position="body" } y pasando un valor de `eval` para el parámetro `step`. Luego, utiliza el siguiente código para transformar el conjunto de datos de evaluación sin procesar (`raw_dataset`) al formato transformado esperado (`transformed_dataset`):

```py{:.devsite-disable-click-to-copy}
def transform(raw_dataset, transform_fn, step):

    transformed_dataset = (
        (raw_dataset, transform_fn)
        | '{} - Transform'.format(step) >> tft_beam.TransformDataset(output_record_batches=True)
    )

    return transformed_dataset
```

Cuando transforma los datos de evaluación, solo se aplican operaciones a nivel de instancia, utilizando tanto la lógica del grafo `transform_fn` como las estadísticas calculadas a partir de la fase de análisis en los datos de entrenamiento. En otras palabras, no se analizan los datos de evaluación en forma completa para calcular nuevas estadísticas, como la media y la varianza para la normalización de puntuación z de características numéricas en los datos de evaluación. En su lugar, utiliza las estadísticas calculadas a partir de los datos de entrenamiento para transformar los datos de evaluación a nivel de instancia.

Por lo tanto, se usa el método `AnalyzeAndTransform` en el contexto de datos de entrenamiento para calcular las estadísticas y transformar los datos. Al mismo tiempo, se utiliza el método `TransformDataset` en el contexto de transformar datos de evaluación para transformar solo los datos a partir de las estadísticas calculadas en los datos de entrenamiento.

Luego, escribe los datos en un receptor (Cloud Storage o disco local, según el ejecutor) en el formato TFRecord para evaluar el modelo de TensorFlow durante el proceso de entrenamiento. Para hacer esto, use la función `write_tfrecords` que se analiza en [Escriba datos de entrenamiento transformados](#step_3_write_transformed_training_data){: track-type="solution" track-name="internalLink" track-metadata-position="body" }. En el siguiente diagrama, figura 3, se muestra cómo se usa el grafo `transform_fn` que se produce en la fase de análisis de los datos de entrenamiento para transformar los datos de evaluación.

<figure id="transform-eval-data-using-transform-fn">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-transforming-eval-data-using-transform_fn.svg"
    alt="Transforming evaluation data using the transform_fn graph.">
  <figcaption><b>Figure 3.</b> Transforming evaluation data using the <code>transform_fn</code> graph.</figcaption>
</figure>

### Guarde el grafo

Un último paso en el proceso de preprocesamiento `tf.Transform` consiste en almacenar los artefactos, incluido el grafo `transform_fn` que se produce en la fase de análisis de los datos de entrenamiento. El código para almacenar los artefactos se muestra en la siguiente función `write_transform_artefacts`:

```py{:.devsite-disable-click-to-copy}
def write_transform_artefacts(transform_fn, location):

    (
        transform_fn
        | 'Write Transform Artifacts' >> transform_fn_io.WriteTransformFn(location)
    )
```

Estos artefactos se utilizarán más adelante para entrenar modelos y exportarlos para su servicio. También se producen los siguientes artefactos, como se muestra en la siguiente sección:

- `saved_model.pb`: representa el grafo de TensorFlow que incluye la lógica de transformación (el grafo `transform_fn`), que se adjuntará a la interfaz de servicio del modelo para transformar los puntos de datos sin procesar al formato transformado.
- `variables`: incluye las estadísticas calculadas durante la fase de análisis de los datos de entrenamiento y se usa en la lógica de transformación en el artefacto `saved_model.pb`.
- `assets`: incluye archivos de vocabulario, uno para cada característica categórica procesada con el método `compute_and_apply_vocabulary`, que se usará durante la publicación para convertir un valor nominal bruto de entrada en un índice numérico.
- `transformed_metadata`: un directorio que contiene el archivo `schema.json` que describe el esquema de los datos transformados.

## Ejecute la canalización en Dataflow{:#run_the_pipeline_in_dataflow}

Después de definir la canalización `tf.Transform`, ejecute la canalización mediante Dataflow. En el siguiente diagrama, figura 4, se muestra el grafo de ejecución de Dataflow de la canalización `tf.Transform` descrita en el ejemplo.

<figure id="dataflow-execution-graph">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-dataflow-execution-graph.png"
    alt="Dataflow execution graph of the tf.Transform pipeline." class="screenshot">
  <figcaption><b>Figure 4.</b> Dataflow execution graph
     of the <code>tf.Transform</code> pipeline.</figcaption>
</figure>

Después de ejecutar la canalización de Dataflow para preprocesar los datos de entrenamiento y evaluación, puede explorar los objetos producidos en Cloud Storage al ejecutar la última celda del bloc de notas. Los fragmentos de código de esta sección muestran los resultados, donde <var><code>YOUR_BUCKET_NAME</code></var> es el nombre de su cubo de Cloud Storage.

Los datos de entrenamiento y evaluación transformados en formato TFRecord se almacenan en la siguiente ubicación:

```none{:.devsite-disable-click-to-copy}
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed
```

Los artefactos de transformación se producen en la siguiente ubicación:

```none{:.devsite-disable-click-to-copy}
gs://YOUR_BUCKET_NAME/babyweight_tft/transform
```

La siguiente lista es la salida de la canalización y muestra los objetos y artefactos de datos producidos:

```none{:.devsite-disable-click-to-copy}
transformed data:
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed/eval-00000-of-00001.tfrecords
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed/train-00000-of-00002.tfrecords
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed/train-00001-of-00002.tfrecords

transformed metadata:
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transformed_metadata/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transformed_metadata/asset_map
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transformed_metadata/schema.pbtxt

transform artefact:
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/saved_model.pb
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/variables/

transform assets:
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/is_male
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/is_multiple
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/mother_race
```

## Implementar el modelo de TensorFlow{: id="implementing_the_tensorflow_model"}

Esta sección y la siguiente, [Entrene y use el modelo para predicciones](#train_and_use_the_model_for_predictions){: track-type="solution" track-name="internalLink" track-metadata-position="body" }, brindan una descripción general y un contexto para Bloc de notas 2. El bloc de notas proporciona un modelo de ML de ejemplo que permite predecir el peso de los bebés. En este ejemplo, se implementa un modelo de TensorFlow que usa la API de Keras. El modelo usa los datos y artefactos producidos por la canalización de preprocesamiento `tf.Transform` que se explicó anteriormente.

### Ejecute el bloc de notas 2

1. En la interfaz de JupyterLab, haga clic en **Archivo &gt; Abrir desde ruta** y luego ingrese la siguiente ruta:

    ```sh
    training-data-analyst/blogs/babyweight_tft/babyweight_tft_keras_02.ipynb
    ```

2. Haga clic en **Editar &gt; Borrar todas las salidas**.

3. En la sección **Instalar paquetes requeridos**, ejecute la primera celda para ejecutar el comando `pip install tensorflow-transform`.

    La última parte de la salida es la siguiente:

    ```none{:.devsite-disable-click-to-copy}
    Successfully installed ...
    Note: you may need to restart the kernel to use updated packages.
    ```

    Puede ignorar los errores de dependencia en la salida.

4. En el menú **Núcleo**, seleccione **Reiniciar núcleo**.

5. Ejecute las celdas en las secciones **Confirmar los paquetes instalados** y **Crear setup.py para instalar paquetes en contenedores de Dataflow**.

6. En la sección **Establecer marcas globales**, junto a `PROJECT` y `BUCKET`, reemplace <code>your-project</code> con su ID de proyecto en Cloud y luego ejecute la celda.

7. Ejecute todas las celdas restantes hasta la última celda del bloc de notas. Para obtener información sobre qué hacer en cada celda, consulte las instrucciones en el bloc de notas.

### Descripción general de la creación del modelo

Los pasos para crear el modelo son los siguientes:

1. Cree columnas de características y use la información del esquema que se almacena en el directorio `transformed_metadata`.
2. Cree el modelo amplio y profundo con la API de Keras y use las columnas de características como entrada al modelo.
3. Cree la función `tfrecords_input_fn` para leer y analizar los datos de entrenamiento y evaluación con los artefactos de transformación.
4. Entrene y evalúe el modelo.
5. Defina una función `serving_fn` que tenga adjunto el grafo `transform_fn` para exportar el modelo entrenado.
6. Inspeccione el modelo exportado con ayuda de la herramienta [`saved_model_cli`](https://www.tensorflow.org/guide/saved_model).
7. Use el modelo exportado para predicciones.

En este documento no se explica cómo compilar el modelo, por lo que no se analiza en detalle cómo se compiló o entrenó el modelo. Sin embargo, las siguientes secciones muestran cómo la información almacenada en el directorio `transform_metadata`, que se produce mediante el proceso `tf.Transform`, se usa para crear las columnas de características del modelo. En el documento también se muestra cómo el grafo `transform_fn`, que también se produce mediante el proceso `tf.Transform`, se usa en la función `serving_fn` cuando el modelo se exporta para su servicio.

### Use los artefactos de transformación generados en el entrenamiento de modelos

Cuando entrena el modelo de TensorFlow, usa los objetos `train` y `eval` transformados que se produjeron en el paso anterior de procesamiento de datos. Estos objetos se almacenan como archivos fragmentados en formato TFRecord. La información del esquema en el directorio `transformed_metadata` generado en el paso anterior puede ser útil para analizar los datos (objetos `tf.train.Example`) para alimentar el modelo para entrenamiento y evaluación.

#### Parsee los datos

Como lee archivos en formato TFRecord para alimentar el modelo con datos de entrenamiento y evaluación, necesita analizar cada objeto `tf.train.Example` en los archivos para crear un diccionario de características (tensores). Esto garantiza que las características se asignen a la capa de entrada del modelo mediante las columnas de características, que actúan como interfaz de evaluación y entrenamiento del modelo. Para analizar los datos, usa el objeto `TFTransformOutput` que se crea a partir de los artefactos generados en el paso anterior:

1. Cree un objeto `TFTransformOutput` a partir de los artefactos que se generan y guardan en el paso de preprocesamiento anterior, como se describe en la sección [Guarde el grafo](#save_the_graph){: track-type="solution" track-name="internalLink" track-metadata-position="body" }:

    ```py
    tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
    ```

2. Extraiga un objeto `feature_spec` del objeto `TFTransformOutput`:

    ```py
    tf_transform_output.transformed_feature_spec()
    ```

3. Use el objeto `feature_spec` para especificar las características contenidas en el objeto `tf.train.Example` como en la función `tfrecords_input_fn`:

    ```py
    def tfrecords_input_fn(files_name_pattern, batch_size=512):

        tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
        TARGET_FEATURE_NAME = 'weight_pounds'

        batched_dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=files_name_pattern,
            batch_size=batch_size,
            features=tf_transform_output.transformed_feature_spec(),
            reader=tf.data.TFRecordDataset,
            label_key=TARGET_FEATURE_NAME,
            shuffle=True).prefetch(tf.data.experimental.AUTOTUNE)

        return batched_dataset
    ```

#### Cree las columnas de características

La canalización produce la información del esquema en el directorio `transformed_metadata` que describe el esquema de los datos transformados que espera el modelo para entrenamiento y evaluación. El esquema contiene el nombre de la característica y el tipo de datos, como los siguientes:

- `gestation_weeks_scaled` (tipo: `FLOAT`)
- `is_male_index` (tipo: `INT`, is_categorical: `True`)
- `is_multiple_index` (tipo: `INT`, is_categorical: `True`)
- `mother_age_bucketized` (tipo: `INT`, is_categorical: `True`)
- `mother_age_log` (tipo: `FLOAT`)
- `mother_age_normalized` (tipo: `FLOAT`)
- `mother_race_index` (tipo: `INT`, is_categorical: `True`)
- `weight_pounds` (tipo: `FLOAT`)

Para ver esta información, use los siguientes comandos:

```sh
transformed_metadata = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR).transformed_metadata
transformed_metadata.schema
```

El siguiente código muestra cómo se usa el nombre de la característica para crear columnas de características:

```py
def create_wide_and_deep_feature_columns():

    deep_feature_columns = []
    wide_feature_columns = []
    inputs = {}
    categorical_columns = {}

    # Select features you've checked from the metadata
    # Categorical features are associated with the vocabulary size (starting from 0)
    numeric_features = ['mother_age_log', 'mother_age_normalized', 'gestation_weeks_scaled']
    categorical_features = [('is_male_index', 1), ('is_multiple_index', 1),
                            ('mother_age_bucketized', 4), ('mother_race_index', 10)]

    for feature in numeric_features:
        deep_feature_columns.append(tf.feature_column.numeric_column(feature))
        inputs[feature] = layers.Input(shape=(), name=feature, dtype='float32')

    for feature, vocab_size in categorical_features:
        categorical_columns[feature] = (
            tf.feature_column.categorical_column_with_identity(feature, num_buckets=vocab_size+1))
        wide_feature_columns.append(tf.feature_column.indicator_column(categorical_columns[feature]))
        inputs[feature] = layers.Input(shape=(), name=feature, dtype='int64')

    mother_race_X_mother_age_bucketized = tf.feature_column.crossed_column(
        [categorical_columns['mother_age_bucketized'],
         categorical_columns['mother_race_index']],  55)
    wide_feature_columns.append(tf.feature_column.indicator_column(mother_race_X_mother_age_bucketized))

    mother_race_X_mother_age_bucketized_embedded = tf.feature_column.embedding_column(
        mother_race_X_mother_age_bucketized, 5)
    deep_feature_columns.append(mother_race_X_mother_age_bucketized_embedded)

    return wide_feature_columns, deep_feature_columns, inputs
```

El código crea una columna `tf.feature_column.numeric_column` para características numéricas y una columna `tf.feature_column.categorical_column_with_identity` para características categóricas.

También puede crear columnas de características extendidas, como se describe en la [Opción C: TensorFlow](/architecture/data-preprocessing-for-ml-with-tf-transform-pt1#option_c_tensorflow){: track-type="solution" track-name="internalLink" track-metadata-position="body" } en la primera parte de esta serie. En el ejemplo que se usa para esta serie, se crea una nueva característica, `mother_race_X_mother_age_bucketized`, al cruzar las características `mother_race` y `mother_age_bucketized` con la columna de características `tf.feature_column.crossed_column`. La representación densa y de baja dimensión de esta característica cruzada se crea mediante el uso de la columna de características `tf.feature_column.embedding_column`.

En el siguiente diagrama, figura 5, se muestran los datos transformados y cómo se utilizan los metadatos transformados para definir y entrenar el modelo de TensorFlow:

<figure id="training-tf-with-transformed-data">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-training-tf-model-with-transformed-data.svg"
    alt="Training the TensorFlow model with transformed data.">
  <figcaption><b>Figure 5.</b> Training the TensorFlow model with
    the transformed data.</figcaption>
</figure>

### Exporte el modelo para servir la predicción

Después de entrenar el modelo de TensorFlow con la API de Keras, exporta el modelo entrenado como un objeto SavedModel, para que pueda servir nuevos puntos de datos para la predicción. Cuando exporta el modelo, debe definir su interfaz, es decir, el esquema de características de entrada que se espera durante el servicio. Este esquema de características de entrada se define en la función `serving_fn`, como se muestra en el siguiente código:

```py{:.devsite-disable-click-to-copy}
def export_serving_model(model, output_dir):

    tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
    # The layer has to be saved to the model for Keras tracking purposes.
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serveing_fn(uid, is_male, mother_race, mother_age, plurality, gestation_weeks):
        features = {
            'is_male': is_male,
            'mother_race': mother_race,
            'mother_age': mother_age,
            'plurality': plurality,
            'gestation_weeks': gestation_weeks
        }
        transformed_features = model.tft_layer(features)
        outputs = model(transformed_features)
        # The prediction results have multiple elements in general.
        # But we need only the first element in our case.
        outputs = tf.map_fn(lambda item: item[0], outputs)

        return {'uid': uid, 'weight': outputs}

    concrete_serving_fn = serveing_fn.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string, name='uid'),
        tf.TensorSpec(shape=[None], dtype=tf.string, name='is_male'),
        tf.TensorSpec(shape=[None], dtype=tf.string, name='mother_race'),
        tf.TensorSpec(shape=[None], dtype=tf.float32, name='mother_age'),
        tf.TensorSpec(shape=[None], dtype=tf.float32, name='plurality'),
        tf.TensorSpec(shape=[None], dtype=tf.float32, name='gestation_weeks')
    )
    signatures = {'serving_default': concrete_serving_fn}

    model.save(output_dir, save_format='tf', signatures=signatures)
```

Durante el servicio, el modelo espera los puntos de datos en su forma sin procesar (es decir, características sin procesar antes de las transformaciones). Por lo tanto, la función `serving_fn` recibe las características sin procesar y las almacena en un objeto `features` como un diccionario de Python. Sin embargo, como se analizó anteriormente, el modelo entrenado espera los puntos de datos en el esquema transformado. Para convertir las características sin procesar en los objetos `transformed_features` que espera la interfaz del modelo, aplique el grafo `transform_fn` guardado al objeto `features` con los siguientes pasos:

1. Cree el objeto `TFTransformOutput` a partir de los artefactos generados y guardados en el paso de preprocesamiento anterior:

    ```py
    tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
    ```

2. Cree un objeto `TransformFeaturesLayer` a partir del objeto `TFTransformOutput`:

    ```py
    model.tft_layer = tf_transform_output.transform_features_layer()
    ```

3. Aplique el grafo `transform_fn` a partir del objeto `TransformFeaturesLayer`:

    ```py
    transformed_features = model.tft_layer(features)
    ```

En el siguiente diagrama, figura 6, se ilustra el paso final de exportación de un modelo para servir:

<figure id="exporting-model-for-serving-with-transform_fn">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-exporting-model-for-serving-with-transform_fn.svg"
    alt="Exporting the model for serving with the transform_fn graph attached.">
  <figcaption><b>Figure 6.</b> Exporting the model for serving with the
    <code>transform_fn</code> graph attached.</figcaption>
</figure>

## Entrene y use el modelo para predicciones

Puedes entrenar el modelo a nivel local si ejecuta las celdas del bloc de notas. Para ver ejemplos de cómo empaquetar el código y entrenar su modelo a escala con ayuda de Vertex AI Training, consulte los ejemplos y las guías en el repositorio de GitHub de Google Cloud [cloudml-samples](https://github.com/GoogleCloudPlatform/cloudml-samples){: .external }.

Cuando inspeccione el objeto SavedModel exportado con la herramienta `saved_model_cli`, verá que los elementos `inputs` de la definición de firma `signature_def` incluyen las características sin procesar, como se muestra en el siguiente ejemplo:

```py{:.devsite-disable-click-to-copy}
signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['gestation_weeks'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: serving_default_gestation_weeks:0
    inputs['is_male'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: serving_default_is_male:0
    inputs['mother_age'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: serving_default_mother_age:0
    inputs['mother_race'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: serving_default_mother_race:0
    inputs['plurality'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: serving_default_plurality:0
    inputs['uid'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: serving_default_uid:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['uid'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: StatefulPartitionedCall_6:0
    outputs['weight'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: StatefulPartitionedCall_6:1
  Method name is: tensorflow/serving/predict
```

Las celdas restantes del bloc de notas le muestran cómo usar el modelo exportado para una predicción local y cómo implementar el modelo como un microservicio usando Vertex AI Prediction. Es importante resaltar que el punto de datos de entrada (muestra) está en el esquema sin formato en ambos casos.

## Limpieza

Para evitar incurrir en cargos adicionales en su cuenta de Google Cloud por los recursos utilizados en este tutorial, elimine el proyecto que contiene los recursos.

### Elimine el proyecto

  <aside class="caution">     <strong>Advertencia</strong>: Eliminar un proyecto tiene los siguientes efectos:<ul>
<li> <strong>Se elimina todo lo que hay en el proyecto.</strong> Si usó un proyecto existente para este tutorial, cuando lo elimine, también eliminará cualquier otro trabajo que haya ejecutado en el proyecto.</li>
<li> <strong>Los ID de proyectos personalizados se pierden.</strong> Es posible que cuando creó este proyecto haya creado un ID de proyecto personalizado que desee usar más adelante. Para conservar las URL que utilizan el ID del proyecto, como la URL <code translate="no" dir="ltr">appspot.com</code>, elimine los recursos seleccionados dentro del proyecto en lugar de eliminar todo el proyecto.</li>
</ul>
<p> Si planea explorar varios tutoriales e inicios rápidos, reutilizar proyectos puede ayudarlo a evitar que exceda los límites de cuota del proyecto.</p></aside>


1. En la consola de Google Cloud, vaya a la página **Administrar recursos**.

    [Vaya a Administrar recursos](https://console.cloud.google.com/iam-admin/projects){: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

2. En la lista de proyectos, seleccione el proyecto que desea eliminar y, luego, haga clic en **Eliminar**.

3. En el cuadro de diálogo, escriba el ID del proyecto y luego haga clic en **Cerrar** para eliminar el proyecto.

## Siguientes pasos

- Para conocer los conceptos, los desafíos y las opciones del preprocesamiento de datos para el aprendizaje automático en Google Cloud, consulte el primer artículo de esta serie, [Preprocesamiento de datos para aprendizaje automático: opciones y recomendaciones](../guide/tft_bestpractices).
- Para obtener más información sobre cómo implementar, empaquetar y ejecutar una canalización tf.Transform en Dataflow, consulte el ejemplo [Cómo predecir ingresos con Census Dataset](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/census/tftransformestimator){: .external }.
- Haga la especialización en ML de Coursera con [TensorFlow en Google Cloud](https://www.coursera.org/specializations/machine-learning-tensorflow-gcp){: .external }.
- Obtenga información sobre las prácticas recomendadas para la ingeniería de ML en [Reglas de ML](https://developers.google.com/machine-learning/guides/rules-of-ml/){: .external }.
- Para obtener más arquitecturas de referencia, diagramas y mejores prácticas, explore el [Centro de arquitectura de Cloud](https://cloud.google.com/architecture).
