# Cómo compilar una canalización de TFX a nivel local

TFX facilita la orquestación de su flujo de trabajo de aprendizaje automático (ML) como canalización para alcanzar los siguientes objetivos:

- Automatizar su proceso de aprendizaje automático, lo que le permite volver a entrenar, evaluar e implementar su modelo con regularidad.
- Crear canalizaciones de aprendizaje automático que incluyan un análisis profundo del rendimiento del modelo y la validación de modelos recién entrenados para garantizar el rendimiento y la confiabilidad.
- Monitorear los datos de entrenamiento para detectar anomalías y eliminar el sesgo entrenamiento-servicio
- Aumentar la velocidad de la experimentación al ejecutar una canalización con diferentes conjuntos de hiperparámetros.

Un proceso típico de desarrollo de canalizaciones empieza en una máquina local, con el análisis de datos y la configuración de los componentes, antes de que se implemente en producción. Esta guía describe dos formas de compilar una canalización a nivel local.

- Personalizar una plantilla de canalización de TFX para que se ajuste a las necesidades de su flujo de trabajo de aprendizaje automático. Las plantillas de canalización de TFX son flujos de trabajo prediseñados que demuestran las prácticas recomendadas mediante el uso de los componentes estándar de TFX.
- Compilar una canalización con TFX. En este caso de uso, se define una canalización sin partir de una plantilla.

Mientras se desarrolla su canalización, puede usar `LocalDagRunner` para ejecutarla. Luego, una vez que los componentes de la canalización se hayan definido y probado bien, se usará un orquestador de nivel de producción como Kubeflow o Airflow.

## Antes de empezar

TFX es un paquete de Python, por lo que deberá configurar un entorno de desarrollo de Python, como un entorno virtual o un contenedor Docker. Luego, escriba este código:

```bash
pip install tfx
```

Si no tiene experiencia con las canalizaciones de TFX, [obtenga más información sobre los conceptos básicos de las canalizaciones de TFX](understanding_tfx_pipelines) antes de continuar.

## Cómo compilar una canalización a partir de una plantilla

Las plantillas de canalización de TFX facilitan los primeros pasos en el desarrollo de canalizaciones, ya que ofrecen un conjunto prediseñado de definiciones de canalizaciones que puede personalizar en función de su caso de uso.

En las siguientes secciones se describe cómo crear una copia de una plantilla y personalizarla para que se adapte a sus necesidades.

### Cree una copia de la plantilla de canalización

1. Consulte la lista de plantillas de canalización de TFX disponibles:

    <pre class="devsite-click-to-copy devsite-terminal">
        tfx template list
        </pre>

2. Elija una plantilla de la lista

    <pre class="devsite-click-to-copy devsite-terminal">
        tfx template copy --model=&lt;var&gt;template&lt;/var&gt; --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
        --destination_path=&lt;var&gt;destination-path&lt;/var&gt;
        </pre>

    Reemplace lo siguiente:

    - <var>template</var>: el nombre de la plantilla que desea copiar.
    - <var>pipeline-name</var>: el nombre de la canalización que se creará.
    - <var>destination-path</var>: la ruta en la que se va a copiar la plantilla.

    Obtenga más información sobre el [comando `tfx template copy`](cli#copy).

3. Se ha creado una copia de la plantilla de canalización en la ruta que especificó.

Nota: En el resto de esta guía se asume que eligió la plantilla `penguin`.

### Explore la plantilla de canalización

Esta sección proporciona una descripción general del andamiaje creado por una plantilla.

1. Explore los directorios y archivos que se copiaron en el directorio raíz de su canalización

    - Un directorio de **canalización** con lo siguiente:

        - `pipeline.py`: define la canalización y enumera qué componentes se están usando
        - `configs.py`: contiene detalles de configuración, como de dónde provienen los datos o qué orquestador se está usando.

    - Un directorio de **datos**

        - Normalmente contiene un archivo `data.csv`, que es la fuente predeterminada para `ExampleGen`. Puede cambiar la fuente de datos en `configs.py`.

    - Un directorio de **modelos** con código de preprocesamiento e implementaciones de modelos.

    - La plantilla copia los ejecutores de DAG para el entorno local y Kubeflow.

    - Algunas plantillas también incluyen blocs de notas de Python para que pueda explorar sus datos y artefactos con Machine Learning MetaData.

2. Ejecute los siguientes comandos en su directorio de canalización:

    <pre class="devsite-click-to-copy devsite-terminal">
        tfx pipeline create --pipeline_path local_runner.py
        </pre>

    <pre class="devsite-click-to-copy devsite-terminal">
        tfx run create --pipeline_name &lt;var&gt;pipeline_name&lt;/var&gt;
        </pre>

    El comando usa `LocalDagRunner` para crear una ejecución de canalización, que agrega los siguientes directorios a su canalización:

    - Un directorio **tfx_metadata** que contiene el almacén ML Metadata que se usa localmente.
    - Un directorio **tfx_pipeline_output** que contiene las salidas de archivos de la canalización.

    Nota: `LocalDagRunner` es uno de tantos orquestadores compatibles con TFX. Es especialmente adecuado para ejecutar canalizaciones localmente para alcanzar iteraciones más rápidas, posiblemente con conjuntos de datos más pequeños. Quizás `LocalDagRunner` no sea adecuado para uso en producción, ya que se ejecuta en una sola máquina, lo que hace que sea más vulnerable a perder trabajo si el sistema deja de estar disponible. TFX también admite orquestadores como Apache Beam, Apache Airflow y Kubeflow Pipeline. Si usa TFX con otro orquestador, use el ejecutor de DAG más apropiado para ese orquestador.

    Nota: Al momento de escribir este artículo, `LocalDagRunner` se usa en la plantilla `penguin`, mientras que la plantilla `taxi` usa Apache Beam. Los archivos de configuración para la plantilla `taxi` están configurados para usar Beam y el comando de CLI es el mismo.

3. Abra el archivo `pipeline/configs.py` de su canalización y revise el contenido. Este script define las opciones de configuración usadas por la canalización y las funciones de los componentes. Aquí es donde especificaría datos como la ubicación de la fuente de datos o el número de pasos de entrenamiento en una ejecución.

4. Abra el archivo `pipeline/pipeline.py` de su canalización y revise el contenido. Este script crea la canalización de TFX. Inicialmente, la canalización contiene solo un componente `ExampleGen`.

    - Siga las instrucciones en los comentarios **TODO** en `pipeline.py` para agregar más pasos a la canalización.

5. Abra el archivo `local_runner.py` y revise el contenido. Este script crea una ejecución de canalización y especifica los *parámetros* de la ejecución, como `data_path` y `preprocessing_fn`.

6. Se revisó el andamiaje creado por la plantilla y se creó una ejecución de canalización a través de `LocalDagRunner`. A continuación, personalice la plantilla para que se ajuste a sus necesidades.

### Personalice su canalización

En esta sección se ofrece una descripción general de cómo comenzar a personalizar su plantilla.

1. Diseñe su canalización. El andamiaje que proporciona una plantilla le ayuda a implementar una canalización para datos tabulares a partir de los componentes estándar de TFX. Si está trasladando un flujo de trabajo de aprendizaje automático existente a una canalización, es posible que deba revisar su código para aprovechar al máximo los [componentes estándar de TFX](index#tfx_standard_components). Es posible que también deba crear [componentes personalizados](understanding_custom_components) que implementen características que sean exclusivas de su flujo de trabajo o que aún no sean compatibles con los componentes estándar de TFX.

2. Una vez que haya diseñado su canalización, personalícela de forma iterativa mediante el siguiente proceso. Comience desde el componente que ingiere datos en su canalización, que suele ser el componente `ExampleGen`.

    1. Personalice la canalización o un componente para que se ajuste a su caso de uso. Estas personalizaciones pueden incluir cambios como los siguientes:

        - Cambio de parámetros de la canalización.
        - Incorporación o eliminación de componentes en la canalización.
        - Reemplazo de la fuente de entrada de datos. Esta fuente de datos puede ser un archivo o consultas a servicios como BigQuery.
        - Cambio en la configuración de un componente en la canalización.
        - Cambio en la función de personalización de un componente.

    2. Ejecute el componente localmente con el script `local_runner.py` o con otro ejecutor de DAG apropiado en caso de que use otro orquestador. Si el script falla, depure el error y vuelva a intentar ejecutar el script.

    3. Una vez que esta personalización esté funcionando, pase a la siguiente personalización.

3. Al trabajar de forma iterativa, puede personalizar cada paso del flujo de trabajo de la plantilla para que se ajuste a sus necesidades.

## Compile una canalización personalizada

Use las siguientes instrucciones para obtener más información sobre cómo crear una canalización personalizada sin utilizar una plantilla.

1. Diseñe su canalización. Los componentes estándar de TFX brindan una funcionalidad comprobada para ayudarlo a implementar un flujo de trabajo de aprendizaje automático completo. Si está trasladando un flujo de trabajo de aprendizaje automático existente a una canalización, es posible que deba revisar su código para aprovechar al máximo los componentes estándar de TFX. Es posible que también deba crear [componentes personalizados](understanding_custom_components) que implementen características como el aumento de datos.

    - Obtenga más información sobre los [componentes estándar de TFX](index#tfx_standard_components).
    - Obtenga más información sobre los [componentes personalizados](understanding_custom_components).

2. Use el siguiente ejemplo para crear un archivo de script para definir su canalización. Esta guía hace referencia a este archivo como `my_pipeline.py`.

    <pre class="devsite-click-to-copy prettyprint">
        import os
        from typing import Optional, Text, List
        from absl import logging
        from ml_metadata.proto import metadata_store_pb2
        import tfx.v1 as tfx

        PIPELINE_NAME = 'my_pipeline'
        PIPELINE_ROOT = os.path.join('.', 'my_pipeline_output')
        METADATA_PATH = os.path.join('.', 'tfx_metadata', PIPELINE_NAME, 'metadata.db')
        ENABLE_CACHE = True

        def create_pipeline(
          pipeline_name: Text,
          pipeline_root:Text,
          enable_cache: bool,
          metadata_connection_config: Optional[
            metadata_store_pb2.ConnectionConfig] = None,
          beam_pipeline_args: Optional[List[Text]] = None
        ):
          components = []

          return tfx.dsl.Pipeline(
                pipeline_name=pipeline_name,
                pipeline_root=pipeline_root,
                components=components,
                enable_cache=enable_cache,
                metadata_connection_config=metadata_connection_config,
                beam_pipeline_args=beam_pipeline_args, &lt;!-- needed? --&gt;
            )

        def run_pipeline():
          my_pipeline = create_pipeline(
              pipeline_name=PIPELINE_NAME,
              pipeline_root=PIPELINE_ROOT,
              enable_cache=ENABLE_CACHE,
              metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH)
              )

          tfx.orchestration.LocalDagRunner().run(my_pipeline)

        if __name__ == '__main__':
          logging.set_verbosity(logging.INFO)
          run_pipeline()
        </pre>

    En los siguientes pasos, definirá su canalización en `create_pipeline` y ejecutará su canalización localmente usando el ejecutor local.

    Compile iterativamente su canalización mediante el siguiente proceso.

    1. Personalice la canalización o un componente para que se ajuste a su caso de uso. Estas personalizaciones pueden incluir cambios como los siguientes:

        - Cambio de parámetros de la canalización.
        - Incorporación o eliminación de componentes en la canalización.
        - Reemplazo de un archivo de entrada de datos.
        - Cambio en la configuración de un componente en la canalización.
        - Cambio en la función de personalización de un componente.

    2. Ejecute el componente localmente con el ejecutor local o con el script directamente. Si el script falla, depure el error y vuelva a intentar ejecutar el script.

    3. Una vez que esta personalización esté funcionando, pase a la siguiente personalización.

    Empiece por el primer nodo del flujo de trabajo de su canalización, normalmente, el primer nodo ingiere datos en su canalización.

3. Agregue el primer nodo de su flujo de trabajo a su canalización. En este ejemplo, la canalización usa el componente estándar `ExampleGen` para cargar un CSV desde un directorio en `./data`.

    <pre class="devsite-click-to-copy prettyprint">
        from tfx.components import CsvExampleGen

        DATA_PATH = os.path.join('.', 'data')

        def create_pipeline(
          pipeline_name: Text,
          pipeline_root:Text,
          data_path: Text,
          enable_cache: bool,
          metadata_connection_config: Optional[
            metadata_store_pb2.ConnectionConfig] = None,
          beam_pipeline_args: Optional[List[Text]] = None
        ):
          components = []

          example_gen = tfx.components.CsvExampleGen(input_base=data_path)
          components.append(example_gen)

          return tfx.dsl.Pipeline(
                pipeline_name=pipeline_name,
                pipeline_root=pipeline_root,
                components=components,
                enable_cache=enable_cache,
                metadata_connection_config=metadata_connection_config,
                beam_pipeline_args=beam_pipeline_args, &lt;!-- needed? --&gt;
            )

        def run_pipeline():
          my_pipeline = create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_path=DATA_PATH,
            enable_cache=ENABLE_CACHE,
            metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH)
            )

          tfx.orchestration.LocalDagRunner().run(my_pipeline)
        </pre>

    `CsvExampleGen` crea registros de ejemplo serializados usando los datos del CSV en la ruta de datos especificada. Configurando el parámetro `input_base` del componente `CsvExampleGen` con la raíz de datos.

4. Cree un directorio `data` en el mismo directorio que `my_pipeline.py`. Agregue un pequeño archivo CSV al directorio `data`.

5. Use el siguiente comando para ejecutar su script `my_pipeline.py`.

    <pre class="devsite-click-to-copy devsite-terminal">
        python my_pipeline.py
        </pre>

    El resultado debería ser similar al siguiente:

    <pre>
        INFO:absl:Component CsvExampleGen depends on [].
        INFO:absl:Component CsvExampleGen is scheduled.
        INFO:absl:Component CsvExampleGen is running.
        INFO:absl:Running driver for CsvExampleGen
        INFO:absl:MetadataStore with DB connection initialized
        INFO:absl:Running executor for CsvExampleGen
        INFO:absl:Generating examples.
        INFO:absl:Using 1 process(es) for Local pipeline execution.
        INFO:absl:Processing input csv data ./data/* to TFExample.
        WARNING:root:Couldn't find python-snappy so the implementation of _TFRecordUtil._masked_crc32c is not as fast as it could be.
        INFO:absl:Examples generated.
        INFO:absl:Running publisher for CsvExampleGen
        INFO:absl:MetadataStore with DB connection initialized
        INFO:absl:Component CsvExampleGen is finished.
        </pre>

6. Continúe agregando componentes de forma iterativa a su canalización.
