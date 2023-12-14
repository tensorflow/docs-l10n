<devsite-mathjax config="TeX-AMS-MML_SVG"></devsite-mathjax>

# Preprocesamiento de datos para aprendizaje automático: opciones y recomendaciones

Este documento es el primero de una serie de dos partes donde se explora el tema de la ingeniería de datos y la ingeniería de características para el aprendizaje automático (ML), centrándose en las tareas de aprendizaje supervisado. En esta primera parte se analizan las prácticas recomendadas para el preprocesamiento de datos en una canalización de ML en Google Cloud. El documento se centra en el uso de las bibliotecas de código abierto TensorFlow y [TensorFlow Transform](https://github.com/tensorflow/transform) {: target="github" class="external" track-type="solution" track-name="gitHubLink" track-metadata-position="body"} (`tf.Transform`) para preparar datos, entrenar el modelo y servir el modelo para predicción. Este documento destaca los desafíos del preprocesamiento de datos para ML y describe las opciones y los escenarios para realizar la transformación de datos en Google Cloud de manera efectiva.

En este documento se asume que el lector está familiarizado con [BigQuery](https://cloud.google.com/bigquery/docs) {: .external }, [Dataflow](https://cloud.google.com/dataflow/docs) {: .external }, [Vertex AI](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform) {: .external } y la API [Keras](https://www.tensorflow.org/guide/keras/overview) de TensorFlow.

En l segundo documento, [Preprocesamiento de datos para ML con Google Cloud](../tutorials/transform/data_preprocessing_with_cloud), se proporciona un tutorial paso a paso sobre cómo implementar una canalización de `tf.Transform`.

## Introducción

El aprendizaje automático le ayuda a encontrar automáticamente patrones complejos y potencialmente útiles en los datos. Estos patrones se condensan en un modelo de ML que luego se puede aplicar a nuevos puntos de datos, un proceso denominado *predicción* o *inferencia*.

La compilación de un modelo de ML es un proceso de varios pasos. Cada paso presenta sus propios desafíos técnicos y conceptuales. Esta serie de dos partes se centra en las tareas de aprendizaje supervisado y el proceso de selección, transformación y aumento de los datos de origen para crear poderosas señales predictivas para la variable objetivo. Estas operaciones combinan el conocimiento del dominio con técnicas de ciencia de datos. Las operaciones son la esencia de la [ingeniería de características](https://developers.google.com/machine-learning/glossary/#feature_engineering) {: .external }.

El tamaño de los conjuntos de datos de entrenamiento para modelos de aprendizaje automático del mundo real puede equivaler o ser mayor que un terabyte (TB) o incluso superarlo. Por lo tanto, se necesitan marcos de procesamiento de datos a gran escala para poder procesar estos conjuntos de datos de manera eficiente y distribuida. Cuando usa un modelo de ML para hacer predicciones, se deben aplicar las mismas transformaciones que se usaron para los datos de entrenamiento en los nuevos puntos de datos. Al aplicar las mismas transformaciones, se presenta el conjunto de datos en vivo al modelo de ML de la manera que el modelo espera.

En este documento se analizan estos desafíos para diferentes niveles de granularidad de las operaciones de ingeniería de características: agregaciones a nivel de instancia, de paso completo y de ventana de tiempo. Además, en este documento se describen las opciones y los escenarios para realizar la transformación de datos para ML en Google Cloud.

Por último, en este documento se proporciona una descripción general de [TensorFlow Transform](https://github.com/tensorflow/transform) {: .external } (`tf.Transform`), una biblioteca para TensorFlow que le permite definir la transformación de datos a nivel de instancia y de paso completo a través de canalizaciones de preprocesamiento de datos. Estas canalizaciones se ejecutan con [Apache Beam](https://beam.apache.org/) {: .external } y crean artefactos que le permiten aplicar las mismas transformaciones durante la predicción que cuando se sirve el modelo.

## Preprocesamiento de datos para ML

En esta sección se presentan las operaciones de preprocesamiento de datos y las etapas de preparación de los datos. También se analizan los tipos de operaciones de preprocesamiento y su granularidad.

### Comparación entre ingeniería de datos e ingeniería de características

El preprocesamiento de datos para ML implica tanto ingeniería de datos como ingeniería de características. La ingeniería de datos es el proceso mediante el cual se convierten *datos sin procesar* en *datos preparados*. Luego, la ingeniería de características ajusta los datos preparados para crear las características que espera el modelo de ML. Estos términos tienen los siguientes significados:

**Datos sin procesar** (o simplemente **datos**): los datos en su forma fuente, sin ninguna preparación previa para ML. En este contexto, los datos pueden estar en su forma original (en un lago de datos) o en forma transformada (en un almacén de datos). Los datos transformados que se encuentran en un almacén de datos pueden haberse convertido de su forma original sin procesar para utilizarlos en análisis. Sin embargo, en este contexto, los *datos sin procesar* significan que los datos no se han preparado específicamente para su tarea de aprendizaje automático. Los datos también se consideran datos sin procesar si se envían desde sistemas de transmisión que eventualmente llaman a modelos de aprendizaje automático para hacer predicciones.

**Datos preparados**: el conjunto de datos listo para su tarea de aprendizaje automático: las fuentes de datos se han analizado, unido y puesto en forma tabular. Los datos preparados se agregan y se resumen con la granularidad adecuada; por ejemplo, cada fila del conjunto de datos representa un cliente único y cada columna representa información resumida del cliente, como el total gastado en las últimas seis semanas. En una tabla de datos preparada, se eliminaron las columnas irrelevantes y se filtraron los registros no válidos. Para tareas de aprendizaje supervisadas, la característica objetivo está presente.

**Características de ingeniería**: el conjunto de datos con las características ajustadas que espera el modelo, es decir, características que se crean al ejecutar determinadas operaciones específicas de ML en las columnas del conjunto de datos preparado y al crear nuevas características para el modelo durante el entrenamiento y la predicción, como se describe más adelante en [Operaciones de preprocesamiento](#preprocessing_operations). Ejemplos de estas operaciones incluyen escalar columnas numéricas a un valor entre 0 y 1, recortar valores y características categóricas de [codificación única](https://developers.google.com/machine-learning/glossary/#one-hot_encoding) {: .external }.

En el siguiente diagrama, figura 1, se muestran los pasos que intervienen en la preparación de datos preprocesados:

<figure id="data-flow-raw-prepared-engineered-features">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-data-preprocessing-flow.svg"
    alt="Flow diagram showing raw data moving to prepared data moving to engineered features.">
  <figcaption><b>Figure 1.</b> The flow of data from raw data to prepared data to engineered
features to machine learning.</figcaption>
</figure>

En la práctica, los datos de la misma fuente suelen encontrarse en diferentes etapas de preparación. Por ejemplo, un campo de una tabla en su almacén de datos podría usarse directamente como una característica de ingeniería. Al mismo tiempo, es posible que otro campo de la misma tabla deba pasar por transformaciones antes de convertirse en una característica de ingeniería. De manera similar, las operaciones de ingeniería de datos y de ingeniería de características podrían combinarse en el mismo paso de preprocesamiento de datos.

### Operaciones de preprocesamiento

El preprocesamiento de datos incluye varias operaciones. Cada operación está diseñada para contribuir a que el ML compile mejores modelos predictivos. Los detalles de estas operaciones de preprocesamiento están fuera del alcance de este documento, pero algunas operaciones se describen brevemente en esta sección.

Para datos estructurados, las operaciones de preprocesamiento de datos incluyen lo siguiente:

- **Limpieza de datos:** eliminar o corregir registros que tienen valores corruptos o no válidos de los datos sin procesar y eliminar registros a los que les faltan una gran cantidad de columnas.
- **Selección y partición de instancias:** selección de puntos de datos del conjunto de datos de entrada para crear [conjuntos de entrenamiento, evaluación (validación) y prueba](https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets) {: .external}. Este proceso incluye técnicas de muestreo aleatorio repetible, sobremuestreo de clases minoritarias y partición estratificada.
- **Ajuste de características:** mejorar la calidad de una característica para ML, lo que incluye escalar y normalizar valores numéricos, imputar valores faltantes, recortar valores atípicos y ajustar valores que tienen distribuciones sesgadas.
- **Transformación de características:** convertir una característica numérica en una característica categórica (mediante [la creación de depósitos](https://developers.google.com/machine-learning/glossary/#bucketing) {: .external }) y convertir características categóricas en una representación numérica (mediante codificación única, [aprendizaje con recuentos](https://dl.acm.org/doi/10.1145/3326937.3341260) {: .external }, incorporaciones de características dispersas, etc. .). Algunos modelos funcionan sólo con características numéricas o categóricas, mientras que otros pueden manejar características de tipo mixto. Incluso cuando los modelos manejan ambos tipos, pueden beneficiarse de diferentes representaciones (numéricas y categóricas) de la misma característica.
- **Extracción de características:** reducir la cantidad de características mediante la creación de representaciones de datos más potentes y de menor dimensión a partir de la aplicación de técnicas como [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) {: .external }, [incorporación](https://developers.google.com/machine-learning/glossary/#embeddings), extracción {: .external } y [hash](https://medium.com/value-stream-design/introducing-one-of-the-best-hacks-in-machine-learning-the-hashing-trick-bf6a9c8af18f) {: .external }.
- **Selección de características:** seleccionar un subconjunto de las características de entrada para entrenar el modelo e ignorar las irrelevantes o redundantes, utilizando [métodos de filtro o envoltorio](https://en.wikipedia.org/wiki/Feature_selection) {: .external}. La selección de características también puede implicar simplemente eliminar características si a las características les falta una gran cantidad de valores.
- **Construcción de características:** creación de nuevas características mediante el uso de técnicas típicas, como [la expansión polinomial](https://en.wikipedia.org/wiki/Polynomial_expansion) {: .external } (mediante el uso de funciones matemáticas univariadas) o [el cruce de características](https://developers.google.com/machine-learning/glossary/#feature_cross) {: .external } (para capturar interacciones de características). Las características también se pueden construir utilizando la lógica empresarial del dominio del caso de uso de ML.

Cuando se trabaja con datos no estructurados (por ejemplo, imágenes, audio o documentos de texto), el aprendizaje profundo reemplaza la ingeniería de características basada en el conocimiento del dominio incorporándola a la arquitectura del modelo. Una [capa convolucional](https://developers.google.com/machine-learning/glossary/#convolutional_layer) {: .external } es un preprocesador de automático. Construir la arquitectura del modelo correcto requiere cierto conocimiento empírico de los datos. Además, se necesita cierta cantidad de preprocesamiento, como el siguiente:

- Para documentos de texto: [derivación y lematización](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) {: .external }, cálculo [TF-IDF](https://en.wikipedia.org/wiki/Tf%e2%80%93idf) {: .external } y extracción de [n-gram](https://en.wikipedia.org/wiki/N-gram) {: .external }, búsqueda de incorporación.
- Para imágenes: recorte, cambio de tamaño, corte, desenfoque gaussiano y filtros canarios.
- Para todo tipo de datos (incluidos texto e imágenes): [transferencia de aprendizaje](https://developers.google.com/machine-learning/glossary/#transfer_learning) {: .external }, que trata todas las capas, excepto las últimas, del modelo completamente entrenado como un paso de ingeniería de características.

### Granularidad de preprocesamiento

En esta sección se analiza la granularidad de los tipos de transformaciones de datos. Se muestra por qué esta perspectiva es fundamental a la hora de preparar nuevos puntos de datos para predicciones mediante transformaciones que se aplican a los datos de entrenamiento.

Las operaciones de preprocesamiento y transformación se pueden clasificar de la siguiente manera, en función de la granularidad de la operación:

- **Transformaciones a nivel de instancia durante el entrenamiento y la predicción**. Estas son transformaciones sencillas, donde solo se necesitan valores de la misma instancia para la transformación. Por ejemplo, las transformaciones a nivel de instancia pueden incluir el recorte del valor de una característica hasta algún umbral, la expansión polinomial de otra característica, la multiplicación de dos características o la comparación de dos características para crear un indicador booleano.

    Estas transformaciones deben aplicarse de manera idéntica durante el entrenamiento y la predicción, porque el modelo se entrenará en las características transformadas, no en los valores de entrada sin procesar. Si los datos no se transforman de manera idéntica, entonces el modelo se comporta mal porque se le presentan datos que tienen una distribución de valores para la que no fue entrenado. Para obtener más información, consulte el debate sobre el sesgo de entrenamiento-servicio en la sección [Desafíos de preprocesamiento](#preprocessing_challenges).

- **Transformaciones de paso completo durante el entrenamiento, pero transformaciones a nivel de instancia durante la predicción**. En este escenario, las transformaciones tienen estado porque usan algunas estadísticas previamente calculadas para ejecutar la transformación. Durante el entrenamiento, se analiza todo el conjunto de datos de entrenamiento para calcular cantidades como mínimo, máximo, media y varianza para transformar los datos de entrenamiento, los datos de evaluación y los datos nuevos en el momento de la predicción.

    Por ejemplo, para normalizar una característica numérica para el entrenamiento, se calcula su media (μ) y su desviación estándar (σ) en todos los datos de entrenamiento. Este cálculo se denomina operación *de paso completo* (o *de análisis*). Cuando sirve el modelo para predicción, el valor de un nuevo punto de datos se normaliza para evitar un sesgo entre entrenamiento y servicio. Por lo tanto, los valores μ y σ que se calculan durante el entrenamiento se usan para ajustar el valor de la característica, que es la siguiente operación simple *a nivel de instancia*:

    <div> $$ value_{scaled} = (value_{raw} - \mu) \div \sigma $$</div>

    Las transformaciones de paso completo incluyen lo siguiente:

    - Características numéricas de escala MinMax con valores *mínimo* y *máximo* calculados a partir del conjunto de datos de entrenamiento.
    - Características numéricas de escala estándar (normalización de puntuación z) con valores de μ y σ calculados en el conjunto de datos de entrenamiento.
    - Creación en depósitos de características numéricas mediante cuantiles.
    - Imputación de valores faltantes a partir de la mediana (características numéricas) o la moda (características categóricas).
    - Conversión de cadenas (valores nominales) en enteros (índices) mediante la extracción de todos los valores distintos (vocabulario) de una característica categórica de entrada.
    - Conteo de la aparición de un término (valor de característica) en todos los documentos (instancias) para calcular TF-IDF.
    - Cálculo del ACP de las características de entrada para proyectar los datos en un espacio de menor dimensión (con características linealmente dependientes).

    Debe usar solo los datos de entrenamiento para calcular estadísticas como μ, σ, *min* y *max*. Si agrega los datos de prueba y evaluación para estas operaciones, está [filtrando información](https://towardsdatascience.com/data-leakage-in-machine-learning-10bdd3eec742) {: .external } de los datos de evaluación y prueba para entrenar el modelo. Hacerlo afecta la confiabilidad de los resultados de la prueba y la evaluación. Para asegurarse de aplicar una transformación coherente a todos los conjuntos de datos, use las mismas estadísticas calculadas a partir de los datos de entrenamiento para transformar los datos de prueba y evaluación.

- **Agregaciones históricas durante el entrenamiento y la predicción**. Esto implica crear agregaciones, derivaciones e indicadores comerciales como señales de entrada para la tarea de predicción; por ejemplo, crear métricas de [actualidad, frecuencia y monetarias (RFM)](https://en.wikipedia.org/wiki/RFM_(market_research)) {: .external } para que los clientes compilen modelos de propensión. Estos tipos de características se pueden calcular previamente y almacenar en un almacén de características para usarlas durante el entrenamiento del modelo, la puntuación por lotes y el servicio de predicciones en línea. También puede aplicar ingeniería de características adicionales (por ejemplo, transformación y ajuste) en estas agregaciones antes del entrenamiento y la predicción.

- **Agregaciones históricas durante el entrenamiento, pero agregaciones en tiempo real durante la predicción**. Este enfoque implica crear una característica que resuma los valores en tiempo real a lo largo del tiempo. En este enfoque, las instancias que se agregarán se definen mediante cláusulas de ventana temporal. Por ejemplo, puede aplicar este enfoque si desea entrenar un modelo que estime el tiempo de viaje en taxi en función de las métricas de tráfico de la ruta en los últimos 5 minutos, en los últimos 10 minutos, en los últimos 30 minutos y en otros intervalos. También puede utilizar este enfoque para predecir la falla de una pieza del motor en función de la media móvil de los valores de temperatura y vibración calculados durante los últimos 3 minutos. Aunque estas agregaciones se pueden preparar fuera de línea para el entrenamiento, se calculan en tiempo real a partir de un flujo de datos durante el servicio.

    Más precisamente, cuando se preparan datos de entrenamiento, si el valor agregado no está en los datos sin procesar, el valor se crea durante la fase de ingeniería de datos. Los datos sin procesar generalmente se almacenan en una base de datos con un formato de `(entity, timestamp, value)`. En los ejemplos anteriores, `entity` es el identificador del segmento de ruta para las rutas de taxi y el identificador de la pieza del motor para el fallo del motor. Puede aplicar operaciones de ventanas para calcular `(entity, time_index, aggregated_value_over_time_window)` y usar las características de agregación como entrada para el entrenamiento del modelo.

    Cuando se sirve el modelo para predicción en tiempo real (en línea), el modelo espera características derivadas de los valores agregados como entrada. Por lo tanto, puede utilizar una tecnología de procesamiento de flujo como Apache Beam para calcular las agregaciones a partir de los puntos de datos en tiempo real transmitidos a su sistema. La tecnología de procesamiento de flujo agrega datos en tiempo real en función de las ventanas de tiempo a medida que llegan nuevos puntos de datos. También puede ejecutar ingeniería de características adicionales (por ejemplo, transformación y ajuste) en estas agregaciones antes del entrenamiento y la predicción.

## Canalización de aprendizaje automático en Google Cloud{: id="machine_learning_pipeline_on_gcp" }

En esta sección se analizan los componentes principales de una canalización típica de un extremo a otro para entrenar y servir modelos de ML de TensorFlow en Google Cloud mediante servicios administrados. También se analiza dónde se pueden implementar diferentes categorías de operaciones de preprocesamiento de datos y los desafíos comunes que podría enfrentar si se implementan dichas transformaciones. La sección [Cómo funciona tf.Transform](#how_tftransform_works) muestra cómo la biblioteca TensorFlow Transform nos ayuda a abordar estos desafíos.

### Arquitectura de alto nivel

En el siguiente diagrama, figura 2, se muestra una arquitectura de alto nivel de una canalización de ML típica para entrenar y servir modelos de TensorFlow. Las etiquetas A, B y C del diagrama se refieren a los diferentes lugares de la canalización donde se puede llevar adelante el preprocesamiento de datos. Los detalles sobre estos pasos se proporcionan en la siguiente sección.

<figure id="high-level-architecture-for-training-and-serving">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-ml-training-serving-architecture.svg"
    alt="Architecture diagram showing stages for processing data.">
  <figcaption><b>Figure 2.</b> High-level architecture for ML training and
    serving on Google Cloud.</figcaption>
</figure>

La canalización consta de los siguientes pasos:

1. Después de importar los datos sin procesar, los datos tabulares se almacenan en BigQuery y otros datos, como imágenes, audio y video, se almacenan en Cloud Storage. En la segunda parte de esta serie se usan datos tabulares almacenados en BigQuery como ejemplo.
2. La ingeniería de datos (preparación) y la ingeniería de características se ejecutan a escala con ayuda de Dataflow. Esta ejecución produce conjuntos de prueba, evaluación y entrenamiento listos para ML que se almacenan en Cloud Storage. Idealmente, estos conjuntos de datos se almacenan como archivos [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord), que es el formato optimizado para los cálculos de TensorFlow.
3. Se envía un [paquete trainer](https://cloud.google.com/vertex-ai/docs/training/create-python-pre-built-container) de modelos TensorFlow {: .external } a Vertex AI Training, que utiliza los datos preprocesados ​​de los pasos anteriores para entrenar el modelo. El resultado de este paso es un [SavedModel](https://www.tensorflow.org/guide/saved_model) TensorFlow entrenado que se exporta a Cloud Storage.
4. El modelo TensorFlow entrenado se implementa en Vertex AI Prediction como un servicio que tiene una API REST para que pueda usarse para predicciones en línea. El mismo modelo también se puede usar para trabajos de predicción por lotes.
5. Una vez que el modelo se implementa como API REST, las aplicaciones cliente y los sistemas internos pueden invocar la API enviando solicitudes con algunos puntos de datos y recibiendo respuestas del modelo con predicciones.
6. Para orquestar y automatizar esta canalización, puede usar [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction) {: .external } como programador para invocar los pasos de preparación de datos, entrenamiento de modelos e implementación de modelos.

También puede usar [Vertex AI Feature Store](https://cloud.google.com/vertex-ai/docs/featurestore/) {: .external } para almacenar características de entrada para hacer predicciones. Por ejemplo, puede crear periódicamente características diseñadas a partir de los datos sin procesar más recientes y almacenarlas en Vertex AI Feature Store. Las aplicaciones cliente obtienen las características de entrada requeridas de Vertex AI Feature Store y las envían al modelo para recibir predicciones.

### Dónde hacer el preprocesamiento

En la figura 2, las etiquetas A, B y C muestran que las operaciones de preprocesamiento de datos se pueden llevar a cabo en BigQuery, Dataflow o TensorFlow. En las siguientes secciones se describe cómo funciona cada una de estas opciones.

#### Opción A: BigQuery{: id="option_a_bigquery"}

Por lo general, la lógica se implementa en BigQuery para las siguientes operaciones:

- Muestreo: seleccionar aleatoriamente un subconjunto de los datos.
- Filtrado: eliminar instancias irrelevantes o no válidas.
- Partición: dividir los datos para producir conjuntos de entrenamiento, evaluación y prueba.

Los scripts SQL de BigQuery se pueden usar como consulta de origen para el proceso de preprocesamiento de Dataflow, que es el paso de procesamiento de datos en la figura 2. Por ejemplo, si se usa un sistema en Canadá y el almacén de datos tiene transacciones de todo el mundo, filtrar a la mejor manera de obtener datos de entrenamiento solo para Canadá es BigQuery. La ingeniería de características en BigQuery es simple y escalable, y admite la implementación de transformaciones de características de agregaciones históricas y a nivel de instancia.

Sin embargo, le recomendamos que use BigQuery para la ingeniería de características solo si usa su modelo para la predicción por lotes (puntuación) o si las características se calculan previamente en BigQuery, pero se almacenan en Vertex AI Feature Store para usarse durante la predicción en línea. Si planea implementar el modelo para predicciones en línea y no tiene la característica diseñada en una tienda de características en línea, debe replicar las operaciones de preprocesamiento de SQL para transformar los puntos de datos sin procesar que generan otros sistemas. En otras palabras, debe implementar la lógica dos veces: una vez en SQL para preprocesar los datos de entrenamiento en BigQuery y una segunda vez en la lógica de la aplicación que consume el modelo para preprocesar los puntos de datos en línea para la predicción.

Por ejemplo, si su aplicación cliente está escrita en Java, deberá volver a implementar la lógica en Java. Esto puede introducir errores debido a discrepancias en la implementación, como se describe en la sección de sesgo entre entrenamiento y servicio de [Desafíos de preprocesamiento](#preprocessing_challenges) más adelante en este documento. También supone una sobrecarga adicional mantener dos implementaciones diferentes. Siempre que cambie la lógica en SQL para preprocesar los datos de entrenamiento, deberá cambiar la implementación de Java en consecuencia para preprocesar los datos en el momento del servicio.

Si usa su modelo solo para la predicción por lotes (por ejemplo, usando [la predicción por lotes](https://cloud.google.com/vertex-ai/docs/predictions/get-batch-predictions) de Vertex AI {: .external }) y si sus datos para la puntuación provienen de BigQuery, puede implementar estas operaciones de preprocesamiento como parte del script SQL de BigQuery. En ese caso, puede utilizar el mismo script SQL de preprocesamiento para preparar datos de entrenamiento y puntuación.

Las transformaciones de paso completo con estado no son adecuadas para implementarse en BigQuery. Si usa BigQuery para transformaciones de paso completo, necesita tablas auxiliares para almacenar las cantidades necesarias para las transformaciones con estado, como medias y varianzas para escalar características numéricas. Además, la implementación de transformaciones de paso completo usando SQL en BigQuery crea una mayor complejidad en los scripts SQL y crea una dependencia compleja entre el entrenamiento y los scripts SQL de puntuación.

#### Opción B: Dataflow{: id="option_b_cloud_dataflow"}

Como se muestra en la figura 2, puede implementar operaciones de preprocesamiento computacionalmente costosas en Apache Beam y ejecutarlas a escala con ayuda de Dataflow. Dataflow es un servicio de escalado automático totalmente administrado que facilita el procesamiento de datos por lotes y en streaming. Cuando se usa Dataflow, también se pueden usar bibliotecas especializadas externas para el procesamiento de datos, lo que lo diferencia de BigQuery.

Dataflow puede ejecutar transformaciones a nivel de instancia y transformaciones de características de agregación históricas y en tiempo real. En particular, si sus modelos de aprendizaje automático esperan una característica de entrada como `total_number_of_clicks_last_90sec`, las [funciones de ventanas](https://beam.apache.org/documentation/programming-guide/#windowing) de Apache Beam {: .external } pueden calcular estas características en función de la agregación de los valores de las ventanas de tiempo de los datos de eventos (transmisión) en tiempo real (por ejemplo, eventos de clic). En la discusión anterior sobre la [granularidad de las transformaciones](#preprocessing_granularity), esto se denominó "Agregaciones históricas durante el entrenamiento, pero agregaciones en tiempo real durante la predicción".

En el siguiente diagrama, figura 3, se muestra la función de Dataflow en el procesamiento de datos de flujo para predicciones casi en tiempo real.

<figure id="high-level-architecture-for-stream-data">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-streaming-data-with-dataflow-architecture.svg"
    alt="Architecture for using stream data for prediction.">
  <figcaption><b>Figure 3.</b> High-level architecture using stream data
    for prediction in Dataflow.</figcaption>
</figure>

Como se muestra en la figura 3, durante el procesamiento, los eventos llamados *puntos de datos* se ingieren a [Pub/Sub](https://cloud.google.com/pubsub/docs) {: .external }. Dataflow consume estos puntos de datos, calcula características basadas en agregados a lo largo del tiempo y luego llama a la API del modelo de ML implementado para hacer predicciones. Luego, las predicciones se envían a una cola Pub/Sub saliente. Desde Pub/Sub, las predicciones pueden ser consumidas por sistemas posteriores, como monitoreo o control, o pueden insertarse (por ejemplo, como notificaciones) al cliente solicitante original. Las predicciones también se pueden almacenar en un almacén de datos de baja latencia como [Cloud Bigtable](https://cloud.google.com/bigtable/docs) {: .external } para recuperarlas en tiempo real. Cloud Bigtable también se puede usar para acumular y almacenar estas agregaciones en tiempo real para poder buscarlas cuando sea necesario para hacer predicciones.

La misma implementación de Apache Beam se puede usar para procesar por lotes datos de entrenamiento que provienen de un almacén de datos fuera de línea como BigQuery y procesar datos en tiempo real para servir predicciones en línea.

En otras arquitecturas típicas, como la arquitectura que se muestra en la figura 2, la aplicación cliente llama directamente a la API del modelo implementado para realizar predicciones en línea. En ese caso, si se implementan operaciones de preprocesamiento en Dataflow para preparar los datos de entrenamiento, las operaciones no se aplican a los datos de predicción que van directamente al modelo. Por lo tanto, transformaciones como estas deben integrarse en el modelo durante el servicio de predicciones en línea.

Dataflow se puede usar para ejecutar una transformación de paso completo, calculando las estadísticas requeridas a escala. Sin embargo, estas estadísticas deben almacenarse en algún lugar para usarse durante la predicción para transformar los puntos de datos de predicción. Al usar la biblioteca TensorFlow Transform (`tf.Transform`), puede incorporar directamente estas estadísticas en el modelo en lugar de almacenarlas en otro lugar. Este enfoque se explica más adelante en [Cómo funciona tf.Transform](#how_tftransform_works).

#### Opción C: TensorFlow{: id="option_c_tensorflow"}

Como se muestra en la figura 2, puede implementar operaciones de transformación y preprocesamiento de datos en el propio modelo de TensorFlow. Como se muestra en la figura, el preprocesamiento que implementa para entrenar el modelo de TensorFlow se convierte en una parte integral del modelo cuando el modelo se exporta e implementa para predicciones. Las transformaciones en el modelo de TensorFlow se pueden lograr de una de las siguientes maneras:

- Al implementar toda la lógica de transformación a nivel de instancia en la función `input_fn` y en la función `serving_fn`. La función `input_fn` prepara un conjunto de datos que usan la [API `tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) para entrenar un modelo. La función `serving_fn` recibe y prepara los datos para las predicciones.
- Al poner el código de transformación directamente en el modelo de TensorFlow mediante el uso de [capas de preprocesamiento de Keras](https://keras.io/guides/preprocessing_layers/) {: .external } o la [creación de capas personalizadas](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) {: .external }.

El código de lógica de transformación en la función [`serving_fn`](https://www.tensorflow.org/guide/saved_model#savedmodels_from_estimators) define la interfaz de servicio de su SavedModel para la predicción en línea. Si implementa las mismas transformaciones que se usaron para preparar los datos de entrenamiento en el código lógico de transformación de la función `serving_fn`, se garantiza que se apliquen las mismas transformaciones a los nuevos puntos de datos de predicción cuando se sirvan.

Sin embargo, debido a que el modelo de TensorFlow procesa cada punto de datos de forma independiente o en un lote pequeño, no se pueden calcular agregaciones a partir de todos los puntos de datos. Como resultado, las transformaciones de paso completo no se pueden implementar en su modelo de TensorFlow.

### Desafíos de preprocesamiento

Los siguientes son los principales desafíos de la implementación del preprocesamiento de datos:

- **"Sesgo entrenamiento-servicio"**. El [sesgo entre entrenamiento y servicio](https://developers.google.com/machine-learning/guides/rules-of-ml/#training-serving_skew) {: .external } hace referencia a una diferencia entre la efectividad (rendimiento predictivo) que se durante el entrenamiento y durante el servicio. Este sesgo puede deberse a una discrepancia entre la forma en que se manejan los datos en las canalizaciones de entrenamiento y de servicio. Por ejemplo, si su modelo está entrenado en una característica transformada logarítmicamente, pero se le presenta la característica sin procesar durante el servicio, es posible que el resultado de la predicción no sea preciso.

    Si las transformaciones se convierten en parte del modelo en sí, puede resultar sencillo manejar las transformaciones a nivel de instancia, como se describió anteriormente en la [Opción C: TensorFlow](#option_c_tensorflow). En ese caso, la interfaz de servicio del modelo (la función [`serving_fn`](https://www.tensorflow.org/guide/saved_model#savedmodels_from_estimators)) espera datos sin procesar, mientras que el modelo transforma internamente estos datos antes de calcular la salida. Las transformaciones son las mismas que se aplicaron en los puntos de datos sin procesar de entrenamiento y predicción.

- **Transformaciones de paso completo**. No se pueden implementar transformaciones de paso completo, como transformaciones de escalado y normalización, en un modelo de TensorFlow. En las transformaciones de paso completo, algunas estadísticas (por ejemplo, valores `max` y `min` para escalar características numéricas) se deben calcular de antemano en los datos de entrenamiento, como se describe en [Opción B: Dataflow](#option_b_dataflow). Luego, los valores deben almacenarse en algún lugar para usarse durante el servicio del modelo para que la predicción transforme los nuevos puntos de datos sin procesar como transformaciones a nivel de instancia, lo que evita el sesgo entre el entrenamiento y el servicio. Se puede usar la biblioteca TensorFlow Transform (`tf.Transform`) para incorporar directamente las estadísticas en su modelo de TensorFlow. Este enfoque se explica más adelante en [Cómo funciona tf.Transform](#how_tftransform_works).

- **Preparación anticipada de los datos para una mejor eficiencia del entrenamiento**. La implementación de transformaciones a nivel de instancia como parte del modelo puede degradar la eficiencia del proceso de entrenamiento. Esta degradación se produce porque las mismas transformaciones se aplican repetidamente a los mismos datos de entrenamiento en cada época. Imagine que tiene datos de entrenamiento sin procesar con 1000 características y aplica una combinación de transformaciones a nivel de instancia para generar 10 000 características. Si implementa estas transformaciones como parte de su modelo y luego alimenta el modelo con los datos de entrenamiento sin procesar, estas 10 000 operaciones se aplican *N* veces en cada instancia, donde *N* es el número de épocas. Además, si utiliza aceleradores (GPU o TPU), permanecen inactivos mientras la CPU ejecuta esas transformaciones, lo que no constituye un uso eficiente de sus costosos aceleradores.

    Lo ideal es que los datos de entrenamiento se transformen antes del entrenamiento, para lo que se emplea la técnica descrita en [Opción B: Dataflow](#option_b_dataflow), donde las 10 000 operaciones de transformación se aplican solo una vez en cada instancia de entrenamiento. Luego, los datos de entrenamiento transformados se presentan al modelo. No se aplican más transformaciones y los aceleradores están ocupados todo el tiempo. Además, usar Dataflow nos ayuda a preprocesar grandes cantidades de datos a escala mediante un servicio totalmente administrado.

    Preparar los datos de entrenamiento por adelantado puede mejorar la eficiencia del entrenamiento. Sin embargo, implementar la lógica de transformación fuera del modelo (los enfoques descritos en [la Opción A: BigQuery](#option_a_bigquery) o [la Opción B: Dataflow](#option_b_dataflow)) no resuelve el problema del sesgo entre el entrenamiento y el servicio. A menos que almacene la característica diseñada en el almacén de características para usarla tanto para el entrenamiento como para la predicción, la lógica de transformación debe implementarse en algún lugar para aplicarse a los nuevos puntos de datos que vienen para la predicción, porque la interfaz del modelo espera datos transformados. La biblioteca TensorFlow Transform (`tf.Transform`) puede ayudarle a solucionar este problema, como se describe en la siguiente sección.

## Cómo funciona tf.Transform{:#how_tftransform_works}

La biblioteca `tf.Transform` es útil para transformaciones que requieren un paso completo. El resultado de la biblioteca `tf.Transform` se exporta como un grafo de TensorFlow que representa la lógica de transformación a nivel de instancia y las estadísticas calculadas a partir de transformaciones de paso completo, que se utilizarán para el entrenamiento y el servicio. Usar el mismo grafo tanto para el entrenamiento como para el servicio puede evitar sesgos, porque se aplican las mismas transformaciones en ambas etapas. Además, la biblioteca `tf.Transform` se puede ejecutar a escala en una canalización de procesamiento por lotes en Dataflow para preparar los datos de entrenamiento por adelantado y mejorar la eficiencia de la entrenamiento.

En el siguiente diagrama, figura 4, se muestra cómo la biblioteca `tf.Transform` preprocesa y transforma datos para entrenamiento y predicción. El proceso se describe en las siguientes secciones.

<figure id="tf-Transform-preprocessing--transforming-data-for-training-and-prediction">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-tf-transform-behavior-flow.svg"
    alt="Diagram showing flow from raw data through tf.Transform to predictions.">
  <figcaption><b>Figure 4.</b> Behavior of <code>tf.Transform</code> for
    preprocessing and transforming data.</figcaption>
</figure>

### Transforme los datos de entrenamiento y evaluación

Los datos de entrenamiento sin procesar se preprocesan mediante la transformación implementada en las API `tf.Transform` Apache Beam y se ejecutan a escala en Dataflow. El preprocesamiento se produce en las siguientes fases:

- **Fase de análisis:** durante la fase de análisis, las estadísticas requeridas (como medias, varianzas y cuantiles) para las transformaciones con estado se calculan en los datos de entrenamiento con operaciones de paso completo. Esta fase produce un conjunto de artefactos de transformación, incluido el grafo `transform_fn`. El grafo `transform_fn` es un grafo de TensorFlow que tiene la lógica de transformación como operaciones a nivel de instancia. Incluye las estadísticas calculadas en la fase de análisis como constantes.
- **Fase de transformación:** durante la fase de transformación, el grafo `transform_fn` se aplica a los datos de entrenamiento sin procesar, donde las estadísticas calculadas se utilizan para procesar los registros de datos (por ejemplo, para escalar columnas numéricas) a nivel de instancia.

Un enfoque de dos fases como este aborda el [Desafío del preprocesamiento](#preprocessing_challenges) de ejecutar transformaciones de paso completo.

Cuando los datos de evaluación se preprocesan, solo se aplican operaciones a nivel de instancia, utilizando la lógica en el grafo `transform_fn` y las estadísticas calculadas a partir de la fase de análisis en los datos de entrenamiento. En otras palabras, no se analizan los datos de evaluación en forma completa para calcular nuevas estadísticas, como μ y σ, para normalizar las características numéricas en los datos de evaluación. En su lugar, utiliza las estadísticas calculadas a partir de los datos de entrenamiento para transformar los datos de evaluación a nivel de instancia.

Los datos de entrenamiento y evaluación transformados se preparan a escala con Dataflow, antes de usarlos para entrenar el modelo. Este proceso de preparación de datos por lotes aborda el [Desafío del preprocesamiento](#preprocessing_challenges) de preparar los datos por adelantado para mejorar la eficiencia del entrenamiento. Como se muestra en la figura 4, la interfaz interna del modelo espera características transformadas.

### Adjunte transformaciones al modelo exportado

Como se señaló, el grafo `transform_fn` producido por la canalización `tf.Transform` se almacena como un grafo de TensorFlow exportado. El grafo exportado consta de la lógica de transformación como operaciones a nivel de instancia y todas las estadísticas calculadas en las transformaciones de paso completo como constantes del grafo. Cuando el modelo entrenado se exporta para su servicio, el grafo `transform_fn` se adjunta al SavedModel como parte de su función `serving_fn`.

Mientras sirve el modelo para la predicción, la interfaz de servicio del modelo espera puntos de datos en formato sin procesar (es decir, antes de cualquier transformación). Sin embargo, la interfaz interna del modelo espera los datos en el formato transformado.

El grafo `transform_fn`, que ahora forma parte del modelo, aplica toda la lógica de preprocesamiento en el punto de datos entrante. Utiliza las constantes almacenadas (como μ y σ para normalizar las características numéricas) en la operación a nivel de instancia durante la predicción. Por lo tanto, el grafo `transform_fn` convierte el punto de datos sin procesar al formato transformado. El formato transformado es lo que espera la interfaz interna del modelo para producir predicción, como se muestra en la figura 4.

Este mecanismo resuelve el [Desafío de preprocesamiento](#preprocessing_challenges) del sesgo entre entrenamiento y servicio, porque la misma lógica (implementación) que se usa para transformar los datos de entrenamiento y evaluación se aplica para transformar los nuevos puntos de datos durante el servicio de predicción.

## Resumen de opciones de preprocesamiento

La siguiente tabla resume las opciones de preprocesamiento de datos que se analizan en este documento. En la tabla, "N/A" significa "no aplicable".

<table class="alternating-odd-rows">
<tbody>
<tr>
<th>   Opción de preprocesamiento de datos</th>
<th>   nivel de instancia<br> (transformaciones sin estado)</th>
<th>
  <p>     Paso completo durante el entrenamiento y nivel de instancia durante el servicio (transformaciones con estado)</p>
</th>
<th>
  <p>     Agregaciones en tiempo real (ventana) durante el entrenamiento y el servicio (transformaciones de transmisión)</p>
</th>
</tr>
<tr>
  <td>
    <p>       <b>BigQuery</b>          (SQL)</p>
  </td>
  <td>
    <p>       <b>Puntuación por lotes: OK</b>: se aplica la misma implementación de transformación a los datos durante el entrenamiento y la puntuación por lotes.</p>
    <p>       <b>Predicción en línea: no recomendada</b>: puede procesar datos de entrenamiento, pero esto da como resultado un sesgo en entrenamiento y servicio porque procesa los datos de servicio con diferentes herramientas.</p>
  </td>
  <td>
    <p>       <b>Puntuación por lotes: no recomendada</b>.</p>
    <p>       <b>Predicción en línea: no recomendada</b>.</p>
    <p>       Si bien se pueden usar estadísticas calculadas con BigQuery para transformaciones en línea o por lotes a nivel de instancia, no es fácil porque es necesario mantener un almacén de estadísticas que se completará durante el entrenamiento y se usará durante la predicción.</p>
  </td>
  <td>
    <p>       <b>Puntuación por lotes: N/A</b>: agregados como estos se calculan en función de eventos en tiempo real.</p>
    <p>       <b>Predicción en línea: no recomendada</b>: puede procesar datos de entrenamiento, pero esto da como resultado un sesgo en entrenamiento y servicio porque procesa los datos de servicio con diferentes herramientas.</p>
  </td>
</tr>
<tr>
  <td>
    <p>       <b>Dataflow</b> (Apache Beam)</p>
  </td>
  <td>
    <p>       <b>Puntuación por lotes: OK</b>: se aplica la misma implementación de transformación a los datos durante el entrenamiento y la puntuación por lotes.</p>
    <p>       <b>Predicción en línea: OK</b>, si los datos en el momento de la publicación provienen de Pub/Sub para ser consumidos por Dataflow. De lo contrario, se producirá un sesgo entre el entrenamiento y el servicio.</p>
  </td>
  <td>
    <p>       <b>Puntuación por lotes: no recomendada</b>.</p>
    <p>       <b>Predicción en línea: no recomendada</b>.</p>
    <p>       Si bien se pueden usar estadísticas calculadas con Dataflow para transformaciones en línea o por lotes a nivel de instancia, no es fácil porque es necesario mantener un almacén de estadísticas que se completará durante el entrenamiento y se usará durante la predicción.</p>
  </td>
  <td>
    <p>       <b>Puntuación por lotes: N/A</b>: agregados como estos se calculan en función de eventos en tiempo real.</p>
    <p>       <b>Predicción en línea: OK</b>: se aplica la misma transformación de Apache Beam a los datos durante el entrenamiento (lote) y el servicio (transmisión).</p>
  </td>
</tr>
<tr>
  <td>
    <p>       <b>Dataflow</b> (Apache Beam + TFT)</p>
  </td>
  <td>
    <p>       <b>Puntuación por lotes: OK</b>: se aplica la misma implementación de transformación a los datos durante el entrenamiento y la puntuación por lotes.</p>
    <p>       <b>Predicción en línea: recomendada</b>: evita el sesgo entre entrenamiento y servicio, y prepara los datos de entrenamiento por adelantado.</p>
  </td>
  <td>
    <p>       <b>Puntuación por lotes: recomendada</b>.</p>
    <p>       <b>Predicción en línea: recomendada</b>.</p>
    <p>       Se recomiendan ambos usos porque la lógica de transformación y las estadísticas calculadas durante el entrenamiento se almacenan como un grafo de TensorFlow que se adjunta al modelo exportado para su servicio.</p>
  </td>
  <td>
    <p>       <b>Puntuación por lotes: N/A</b>: agregados como estos se calculan en función de eventos en tiempo real.</p>
    <p>       <b>Predicción en línea: OK</b>: se aplica la misma transformación de Apache Beam a los datos durante el entrenamiento (lote) y el servicio (transmisión).</p>
  </td>
</tr>
<tr>
  <td>
    <p>       <b>TensorFlow</b> <sup>*</sup>       <br>       (<code>input_fn</code> y <code>serving_fn</code>)</p>
  </td>
  <td>
    <p>       <b>Puntuación por lotes: no recomendada</b>.</p>
    <p>       <b>Predicción en línea: no recomendada</b>.</p>
    <p>       Para que el entrenamiento sea eficiente en ambos casos, es mejor preparar los datos del entrenamiento por adelantado.</p>
  </td>
  <td>
    <p>       <b>Puntuación por lotes: no es posible</b>.</p>
    <p>       <b>Predicción en línea: no es posible</b>.</p>
  </td>
  <td>
    <p>       <b>Puntuación por lotes: N/A</b>: agregados como estos se calculan en función de eventos en tiempo real.</p>
<p>       <b>Predicción en línea: no es posible</b>.</p>
  </td>
</tr>
</tbody>
</table>

<sup>*</sup> Con TensorFlow, las transformaciones como el cruce, la incorporación y la codificación única deben realizarse de forma declarativa como columnas `feature_columns`.

## Siguientes pasos

- Para implementar una canalización de `tf.Transform` y ejecutarla con ayuda de Dataflow, lea la segunda parte de esta serie, [Preprocesamiento de datos para ML usando TensorFlow Transform](https://www.tensorflow.org/tfx/tutorials/transform/data_preprocessing_with_cloud).
- Haga la especialización en ML de Coursera con [TensorFlow en Google Cloud](https://www.coursera.org/specializations/machine-learning-tensorflow-gcp) {: .external }.
- Obtenga información sobre las prácticas recomendadas para la ingeniería de ML en [Reglas de ML](https://developers.google.com/machine-learning/guides/rules-of-ml/) {: .external }.

- Para obtener más arquitecturas de referencia, diagramas y mejores prácticas, explore las <a href="https://www.tensorflow.org/tfx/guide/solutions" track-type="tutorial" track-name="textLink" track-metadata-position="nextSteps">Soluciones en la nube de TFX</a>.
