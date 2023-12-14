# Cómo compilar canalizaciones de TFX

Nota: Para obtener una vista conceptual de las canalizaciones de TFX, consulte [Explicación de las canalizaciones de TFX](understanding_tfx_pipelines).

Nota: ¿Quiere compilar su primera canalización antes de ahondar en los detalles? Comience a [compilar una canalización con ayuda de una plantilla](https://www.tensorflow.org/tfx/guide/build_local_pipeline#build_a_pipeline_using_a_template).

## Cómo usar la clase `Pipeline`

Las canalizaciones de TFX se definen con la [clase `Pipeline`](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/pipeline.py) {: .external }. En el siguiente ejemplo se muestra cómo se usa la clase `Pipeline`.

<pre class="devsite-click-to-copy prettyprint">
pipeline.Pipeline(
    pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;,
    pipeline_root=&lt;var&gt;pipeline-root&lt;/var&gt;,
    components=&lt;var&gt;components&lt;/var&gt;,
    enable_cache=&lt;var&gt;enable-cache&lt;/var&gt;,
    metadata_connection_config=&lt;var&gt;metadata-connection-config&lt;/var&gt;,
)
</pre>

Reemplace lo siguiente:

- <var>pipeline-name</var>: el nombre de esta canalización. El nombre de la canalización debe ser único.

    TFX usa el nombre de la canalización para consultar a ML Metadata en busca de artefactos de entrada de componentes. La reutilización de un nombre de canalización puede provocar comportamientos inesperados.

- <var>pipeline-root</var>: ruta raíz de las salidas de esta canalización. La ruta raíz debe ser la ruta completa a un directorio al que su orquestador tenga acceso de lectura y escritura. En tiempo de ejecución, TFX usa la raíz de la canalización para generar rutas de salida para los artefactos de los componentes. Este directorio puede ser local o estar en un sistema de archivos distribuido compatible, como Google Cloud Storage o HDFS.

- <var>components</var>: una lista de instancias de componentes que conforman el flujo de trabajo de esta canalización.

- <var>enable-cache</var>: (Opcional). Valor booleano que indica si esta canalización usa el almacenamiento en caché para acelerar su ejecución.

- <var>metadata-connection-config</var>: (Opcional). Una configuración de conexión para ML Metadata.

## Cómo definir el grafo de ejecución de componentes

Las instancias de componentes producen artefactos como salidas y normalmente dependen de los artefactos producidos por instancias de componentes ascendentes como entradas. La secuencia de ejecución de las instancias de componentes se determina mediante la creación de un grafo acíclico dirigido (DAG) de las dependencias de los artefactos.

Por ejemplo, el componente estándar `ExampleGen` puede ingerir datos de un archivo CSV y generar registros de ejemplo serializados. El componente estándar `StatisticsGen` acepta estos registros de ejemplo como entrada y produce estadísticas del conjunto de datos. En este ejemplo, la instancia de `StatisticsGen` debe seguir `ExampleGen` porque `SchemaGen` depende de la salida de `ExampleGen`.

### Dependencias basadas en tareas

Nota: Normalmente no se recomienda el uso de dependencias basadas en tareas. Definir el grafo de ejecución con dependencias de artefactos le permite aprovechar las funciones de almacenamiento en caché y seguimiento automático del linaje de artefactos de TFX.

También puede definir dependencias basadas en tareas si usa los métodos [`add_upstream_node` y `add_downstream_node`](https://github.com/tensorflow/tfx/blob/master/tfx/components/base/base_node.py) {: .external } de su componente. `add_upstream_node` le permite especificar que el componente actual debe ejecutarse después del componente especificado. `add_downstream_node` le permite especificar que el componente actual debe ejecutarse antes que el componente especificado.

## Plantillas de canalización

La forma más sencilla de configurar rápidamente una canalización y ver cómo encajan todas las piezas es mediante el uso de una plantilla. El uso de plantillas se aborda en [Cómo compilar una canalización de TFX a nivel local](build_local_pipeline).

## Almacenamiento en caché

El almacenamiento en caché de la canalización de TFX permite que la canalización omita los componentes que se ejecutaron con el mismo conjunto de entradas en una ejecución anterior de la canalización. Si el almacenamiento en caché está habilitado, la canalización intenta hacer coincidir la firma de cada componente, el componente y el conjunto de entradas, con una de las ejecuciones anteriores de componentes de esta canalización. Si hay coincidencia, la canalización usa las salidas de los componentes de la ejecución anterior. Si no hay coincidencia, se ejecuta el componente.

No use el almacenamiento en caché si su canalización usa componentes no deterministas. Por ejemplo, si crea un componente para crear un número aleatorio para su canalización, habilitar el caché hace que este componente se ejecute una vez. En este ejemplo, las ejecuciones posteriores usan el número aleatorio de la primera ejecución en lugar de generar un número aleatorio.
