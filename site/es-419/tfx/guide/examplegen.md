# El componente de canalización ExampleGen TFX

El componente de canalización ExampleGen TFX ingiere datos en las canalizaciones de TFX. Consume archivos/servicios externos para generar ejemplos que serán leídos por otros componentes de TFX. También proporciona una partición coherente y configurable, y selecciona aleatoriamente el conjunto de datos para alcanzar las prácticas recomendadas de ML.

- Consume: datos de fuentes de datos externas como CSV, `TFRecord`, Avro, Parquet y BigQuery.
- Emite: registros `tf.Example`, registros `tf.SequenceExample` o formato proto, según el formato de carga útil.

## ExampleGen y otros componentes

ExampleGen ofrece datos a componentes que utilizan la biblioteca [TensorFlow Data Validation](tfdv.md), como [SchemaGen](schemagen.md), [StatisticsGen](statsgen.md) y [ExampleValidator](exampleval.md). También proporciona datos a [Transform](transform.md), que usa la biblioteca [TensorFlow Transform](tft.md) y, en última instancia, a los objetivos de implementación durante la inferencia.

## Fuentes de datos y formatos

Actualmente, una instalación estándar de TFX incluye componentes completos de ExampleGen para estos formatos y fuentes de datos:

- [CSV](https://github.com/tensorflow/tfx/tree/master/tfx/components/example_gen/csv_example_gen)
- [tf.Record](https://github.com/tensorflow/tfx/tree/master/tfx/components/example_gen/import_example_gen)
- [BigQuery](https://github.com/tensorflow/tfx/tree/master/tfx/extensions/google_cloud_big_query/example_gen)

También hay ejecutores personalizados disponibles que permiten el desarrollo de componentes de ExampleGen para estas fuentes y formatos de datos:

- [Avro](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/custom_executors/avro_executor.py)
- [Parquet](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/custom_executors/parquet_executor.py)

Consulte los ejemplos de uso en el código fuente y [este debate](/tfx/guide/examplegen#custom_examplegen) para obtener más información sobre cómo usar y desarrollar ejecutores personalizados.

Nota: En la mayoría de los casos, es mejor heredar de `base_example_gen_executor` en lugar de `base_executor`. Por lo tanto, quizás sea conveniente seguir el ejemplo de Avro o Parquet en el código fuente de Executor.

Además, estas fuentes y formatos de datos están disponibles como ejemplos de [componentes personalizados](/tfx/guide/understanding_custom_components):

- [Presto](https://github.com/tensorflow/tfx/tree/master/tfx/examples/custom_components/presto_example_gen)

### Cómo ingerir formatos de datos compatibles con Apache Beam

Apache Beam admite la ingesta de datos de una [amplia gama de fuentes y formatos de datos](https://beam.apache.org/documentation/io/built-in/) ([consultar a continuación](#additional_data_formats)). Estas capacidades se pueden usar para crear componentes de ExampleGen personalizados para TFX, lo cual se demuestra con algunos componentes de ExampleGen existentes ([ver más abajo](#additional_data_formats)).

## Cómo usar un componente ExampleGen

Para las fuentes de datos admitidas (actualmente, archivos CSV, archivos TFRecord con `tf.Example`, `tf.SequenceExample` y formato proto, y resultados de consultas de BigQuery), el componente de canalización ExampleGen se puede usar directamente en la implementación y requiere poca personalización. Por ejemplo:

```python
example_gen = CsvExampleGen(input_base='data_root')
```

o, como se muestra a continuación, para importar TFRecord externo directamente con `tf.Example`:

```python
example_gen = ImportExampleGen(input_base=path_to_tfrecord_dir)
```

## Intervalo, versión y división

Un Span, o intervalo, es un conjunto de ejemplos de entrenamiento. Si sus datos persisten en un sistema de archivos, cada intervalo se puede almacenar en un directorio independiente. La semántica de un intervalo no está codificado en TFX; un intervalo puede corresponder a un día de datos, una hora de datos o cualquier otro conjunto que sea significativo para su tarea.

Cada intervalo puede contener múltiples versiones de datos. Para dar un ejemplo, si elimina algunos ejemplos de un intervalo para eliminar datos de mala calidad, esto podría resultar en una nueva versión de ese intervalo. De forma predeterminada, los componentes de TFX funcionan con la última versión dentro de un intervalo.

Cada versión dentro de un intervalo se puede subdividir en múltiples divisiones. El caso de uso más común para dividir un intervalo es dividirlo en datos de entrenamiento y evaluación.

![Intervalos y divisiones](images/spans_splits.png)

### División de entrada/salida personalizada

Nota: Esta característica solo está disponible a partir de TFX 0.14.

Para personalizar la relación de división de entrenamiento/evaluación que generará ExampleGen, configure `output_config` para el componente ExampleGen. Por ejemplo:

```python
# Input has a single split 'input_dir/*'.
# Output 2 splits: train:eval=3:1.
output = proto.Output(
             split_config=example_gen_pb2.SplitConfig(splits=[
                 proto.SplitConfig.Split(name='train', hash_buckets=3),
                 proto.SplitConfig.Split(name='eval', hash_buckets=1)
             ]))
example_gen = CsvExampleGen(input_base=input_dir, output_config=output)
```

Observe cómo se configuró `hash_buckets` en este ejemplo.

Para una fuente de entrada que ya se dividió, configure `input_config` para el componente ExampleGen:

```python

# Input train split is 'input_dir/train/*', eval split is 'input_dir/eval/*'.
# Output splits are generated one-to-one mapping from input splits.
input = proto.Input(splits=[
                example_gen_pb2.Input.Split(name='train', pattern='train/*'),
                example_gen_pb2.Input.Split(name='eval', pattern='eval/*')
            ])
example_gen = CsvExampleGen(input_base=input_dir, input_config=input)
```

Para la generación de ejemplos basada en archivos (por ejemplo, CsvExampleGen e ImportExampleGen), `pattern` es un patrón de archivo relativo global que se asigna a archivos de entrada con el directorio raíz proporcionado por la ruta base de entrada. Para generación de ejemplos basados ​​en consultas (por ejemplo, BigQueryExampleGen, PrestoExampleGen), `pattern` es una consulta SQL.

De forma predeterminada, todo el directorio base de entrada se trata como una única división de entrada, y la división de salida de entrenamiento y evaluación se genera con una proporción de 2:1.

Consulte [proto/example_gen.proto](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto) para conocer la configuración dividida de entrada y salida de ExampleGen. Y consulte la [guía de componentes de flujo descendente](#examplegen_downstream_components) para usar las divisiones personalizadas de flujo descendente.

#### Método de división

Cuando se usa el método de división `hash_buckets`, en lugar del registro completo, se puede usar una característica para particionar los ejemplos. Si hay una característica presente, ExampleGen usa una huella digital de esa característica como clave de partición.

Esta característica sirve para mantener una división estable con respecto a ciertas propiedades de los ejemplos: en ese caso, un usuario siempre se colocará en la misma división si se seleccionó "user_id" como nombre de la característica de partición.

La interpretación de lo que significa una "característica" y cómo hacer coincidir una "característica" con el nombre especificado depende de la implementación de ExampleGen y del tipo de ejemplos.

Para implementaciones de ExampleGen listas para usar:

- Si genera tf.Example, entonces una "característica" significa una entrada en tf.Example.features.feature.
- Si genera tf.SequenceExample, entonces una "característica" significa una entrada en tf.SequenceExample.context.feature.
- Solo se admiten características int64 y bytes.

En los siguientes casos, ExampleGen arroja errores de tiempo de ejecución:

- El nombre de la característica especificada no existe en el ejemplo.
- Característica vacía: `tf.train.Feature()`.
- Tipos de características no admitidas, por ejemplo, características flotantes.

Para generar la división de entrenamiento/evaluación en una característica de los ejemplos, configure `output_config` para el componente ExampleGen. Por ejemplo:

```python
# Input has a single split 'input_dir/*'.
# Output 2 splits based on 'user_id' features: train:eval=3:1.
output = proto.Output(
             split_config=proto.SplitConfig(splits=[
                 proto.SplitConfig.Split(name='train', hash_buckets=3),
                 proto.SplitConfig.Split(name='eval', hash_buckets=1)
             ],
             partition_feature_name='user_id'))
example_gen = CsvExampleGen(input_base=input_dir, output_config=output)
```

Observe cómo se configuró `partition_feature_name` en este ejemplo.

### Intervalo

Nota: Esta característica solo está disponible a partir de TFX 0.15.

El intervalo se puede recuperar mediante el uso de la especificación '{SPAN}' en el [patrón global de entrada](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto):

- Esta especificación coincide con dígitos y asigna los datos a los números SPAN relevantes. Por ejemplo, 'data_{SPAN}-*.tfrecord' recopilará archivos como 'data_12-a.tfrecord', 'date_12-b.tfrecord'.
- Opcionalmente, esta especificación se puede definir con el ancho de los números enteros cuando se asigna. Por ejemplo, 'data_{SPAN:2}.file' se asigna a archivos como 'data_02.file' y 'data_27.file' (como entradas para Span-2 y Span-27 respectivamente), pero no se asigna a 'data_1. file' ni 'data_123.file'.
- Cuando falta la especificación SPAN, se supone que siempre corresponde a Span '0'.
- Si se especifica SPAN, la canalización procesará el último intervalo y almacenará el número del intervalo en metadatos.

Por ejemplo, supongamos que hay datos de entrada:

- '/tmp/span-1/train/data'
- '/tmp/span-1/eval/data'
- '/tmp/span-2/train/data'
- '/tmp/span-2/eval/data'

y la configuración de entrada se muestra a continuación:

```python
splits {
  name: 'train'
  pattern: 'span-{SPAN}/train/*'
}
splits {
  name: 'eval'
  pattern: 'span-{SPAN}/eval/*'
}
```

al activar la canalización, procesará lo siguiente:

- '/tmp/span-2/train/data' como división de entrenamiento
- '/tmp/span-2/eval/data' como división de evaluación

con número de intervalo '2'. Si más adelante '/tmp/span-3/...' está listo, simplemente active la canalización nuevamente y recogerá el intervalo '3' para su procesamiento. A continuación, se muestra el ejemplo de código para usar la especificación de intervalo:

```python
input = proto.Input(splits=[
                proto.Input.Split(name='train',
                                            pattern='span-{SPAN}/train/*'),
                proto.Input.Split(name='eval',
                                            pattern='span-{SPAN}/eval/*')
            ])
example_gen = CsvExampleGen(input_base='/tmp', input_config=input)
```

Se puede recuperar un intervalo determinado con RangeConfig, que se detalla a continuación.

### Fecha

Nota: Esta característica solo está disponible a partir de TFX 0.24.0.

Si su fuente de datos está organizada en el sistema de archivos por fecha, TFX admite la asignación de fechas directamente para abarcar números. Hay tres especificaciones para representar la asignación de fechas a intervalos: {AAAA}, {MM} y {DD}:

- Las tres especificaciones deben estar presentes en el [patrón global de entrada](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto), si se especifica alguna:
- Se puede especificar exclusivamente la especificación {SPAN} o este conjunto de especificaciones de fecha.
- Se calcula una fecha del calendario con el año desde AAAA, el mes desde MM y el día del mes desde DD, luego el número de intervalo se calcula como el número de días desde la época Unix (es decir, 1970-01-01). Por ejemplo, 'log-{AAAA}{MM}{DD}.data' coincide con un archivo 'log-19700101.data' y lo consume como entrada para Span-0, y 'log-20170101.data' como entrada para Span-17167.
- Si se especifica este conjunto de especificaciones de fecha, la canalización procesará la última fecha y almacenará el número de intervalo correspondiente en metadatos.

Por ejemplo, supongamos que hay datos de entrada organizados por fecha del calendario:

- '/tmp/1970-01-02/train/data'
- '/tmp/1970-01-02/eval/data'
- '/tmp/1970-01-03/train/data'
- '/tmp/1970-01-03/eval/data'

y la configuración de entrada se muestra a continuación:

```python
splits {
  name: 'train'
  pattern: '{YYYY}-{MM}-{DD}/train/*'
}
splits {
  name: 'eval'
  pattern: '{YYYY}-{MM}-{DD}/eval/*'
}
```

al activar la canalización, procesará lo siguiente:

- '/tmp/1970-01-03/train/data' como división de entrenamiento
- '/tmp/1970-01-03/eval/data' como división de evaluación

con número de intervalo '2'. Si más adelante '/tmp/1970-01-04/...' está listo, simplemente active la canalización nuevamente y recogerá el intervalo '3' para su procesamiento. A continuación, se muestra el ejemplo de código para usar la especificación de fechas:

```python
input = proto.Input(splits=[
                proto.Input.Split(name='train',
                                            pattern='{YYYY}-{MM}-{DD}/train/*'),
                proto.Input.Split(name='eval',
                                            pattern='{YYYY}-{MM}-{DD}/eval/*')
            ])
example_gen = CsvExampleGen(input_base='/tmp', input_config=input)
```

### Versión

Nota: Esta característica solo está disponible a partir de TFX 0.24.0.

La versión se puede recuperar mediante el uso de la especificación '{VERSION}' en el [patrón global de entrada](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto):

- Esta especificación hace coincidir los dígitos y asigna los datos a los números de VERSION correspondientes bajo el SPAN. Tenga en cuenta que la especificación de versión se puede usar en combinación con la especificación de intervalo o de fecha.
- Esta especificación también se puede definir opcionalmente con el ancho de la misma manera que la especificación SPAN. Por ejemplo, 'span-{SPAN}/version-{VERSION:4}/data-*'.
- Cuando falta la especificación VERSION, la versión se establece en None.
- Si se especifican SPAN y VERSION, la canalización procesará la última versión para el último intervalo y almacenará el número de versión en metadatos.
- Si se especifica VERSION, pero no SPAN (o especificación de fecha), se generará un error.

Por ejemplo, supongamos que hay datos de entrada:

- '/tmp/span-1/ver-1/train/data'
- '/tmp/span-1/ver-1/eval/data'
- '/tmp/span-2/ver-1/train/data'
- '/tmp/span-2/ver-1/eval/data'
- '/tmp/span-2/ver-2/train/data'
- '/tmp/span-2/ver-2/eval/data'

y la configuración de entrada se muestra a continuación:

```python
splits {
  name: 'train'
  pattern: 'span-{SPAN}/ver-{VERSION}/train/*'
}
splits {
  name: 'eval'
  pattern: 'span-{SPAN}/ver-{VERSION}/eval/*'
}
```

al activar la canalización, procesará lo siguiente:

- '/tmp/span-2/ver-2/train/data' como división de entrenamiento
- '/tmp/span-2/ver-2/eval/data' como división de evaluación

con número de intervalo '2' y número de versión '2'. Si más adelante '/tmp/span-2/ver-3/...' está listo, simplemente active la canalización nuevamente y seleccionará el intervalo '2' y la versión '3' para su procesamiento. A continuación, se muestra el ejemplo de código para usar la especificación de versión:

```python
input = proto.Input(splits=[
                proto.Input.Split(name='train',
                                            pattern='span-{SPAN}/ver-{VERSION}/train/*'),
                proto.Input.Split(name='eval',
                                            pattern='span-{SPAN}/ver-{VERSION}/eval/*')
            ])
example_gen = CsvExampleGen(input_base='/tmp', input_config=input)
```

### Range Config

Nota: Esta característica solo está disponible a partir de TFX 0.24.0.

TFX admite la recuperación y el procesamiento de un intervalo específico en ExampleGen basado en archivos mediante la configuración de rangos, una configuración abstracta que se usa para describir rangos para diferentes entidades de TFX. Si desea recuperar un intervalo específico, configure `range_config` para un componente ExampleGen basado en archivos. Por ejemplo, supongamos que hay datos de entrada:

- '/tmp/span-01/train/data'
- '/tmp/span-01/eval/data'
- '/tmp/span-02/train/data'
- '/tmp/span-02/eval/data'

Para recuperar y procesar datos específicamente con intervalo '1', especificamos una configuración de rango además de la configuración de entrada. Tenga en cuenta que ExampleGen solo admite rangos estáticos de un solo intervalo (para especificar el procesamiento de intervalos individuales específicos). Por lo tanto, para StaticRange, start_span_number debe ser igual a end_span_number. Con ayuda del intervalo proporcionado y la información del ancho del intervalo (si se proporciona) para el relleno con ceros, ExampleGen reemplazará la especificación SPAN en los patrones de división proporcionados con el número de intervalo deseado. A continuación, se muestra un ejemplo de uso:

```python
# In cases where files have zero-padding, the width modifier in SPAN spec is
# required so TFX can correctly substitute spec with zero-padded span number.
input = proto.Input(splits=[
                proto.Input.Split(name='train',
                                            pattern='span-{SPAN:2}/train/*'),
                proto.Input.Split(name='eval',
                                            pattern='span-{SPAN:2}/eval/*')
            ])
# Specify the span number to be processed here using StaticRange.
range = proto.RangeConfig(
                static_range=proto.StaticRange(
                        start_span_number=1, end_span_number=1)
            )

# After substitution, the train and eval split patterns will be
# 'input_dir/span-01/train/*' and 'input_dir/span-01/eval/*', respectively.
example_gen = CsvExampleGen(input_base=input_dir, input_config=input,
                            range_config=range)
```

La configuración de rangos también se puede usar para procesar fechas específicas, si se usa la especificación de fecha en lugar de la especificación SPAN. Por ejemplo, supongamos que hay datos de entrada organizados por fecha del calendario:

- '/tmp/1970-01-02/train/data'
- '/tmp/1970-01-02/eval/data'
- '/tmp/1970-01-03/train/data'
- '/tmp/1970-01-03/eval/data'

Para recuperar y procesar datos específicamente el 2 de enero de 1970, hacemos lo siguiente:

```python
from  tfx.components.example_gen import utils

input = proto.Input(splits=[
                proto.Input.Split(name='train',
                                            pattern='{YYYY}-{MM}-{DD}/train/*'),
                proto.Input.Split(name='eval',
                                            pattern='{YYYY}-{MM}-{DD}/eval/*')
            ])
# Specify date to be converted to span number to be processed using StaticRange.
span = utils.date_to_span_number(1970, 1, 2)
range = proto.RangeConfig(
                static_range=range_config_pb2.StaticRange(
                        start_span_number=span, end_span_number=span)
            )

# After substitution, the train and eval split patterns will be
# 'input_dir/1970-01-02/train/*' and 'input_dir/1970-01-02/eval/*',
# respectively.
example_gen = CsvExampleGen(input_base=input_dir, input_config=input,
                            range_config=range)
```

## ExampleGen personalizado

Si los componentes ExampleGen disponibles actualmente no se ajustan a sus necesidades, puede crear un ExampleGen personalizado, que le permitirá leer desde diferentes fuentes de datos o en diferentes formatos de datos.

### Personalización de ExampleGen basado en archivos (experimental)

Primero, extienda BaseExampleGenExecutor con un Beam PTransform personalizado, que proporciona la conversión de su división de entrada de entrenamiento/evaluación a ejemplos de TF. Por ejemplo, el [ejecutor CsvExampleGen](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/csv_example_gen/executor.py) ofrece la conversión de una división CSV de entrada a ejemplos TF.

Luego, cree un componente con el ejecutor anterior, como se hizo en el [componente CsvExampleGen](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/csv_example_gen/component.py). Alternativamente, pase un ejecutor personalizado al componente estándar ExampleGen como se muestra a continuación.

```python
from tfx.components.base import executor_spec
from tfx.components.example_gen.csv_example_gen import executor

example_gen = FileBasedExampleGen(
    input_base=os.path.join(base_dir, 'data/simple'),
    custom_executor_spec=executor_spec.ExecutorClassSpec(executor.Executor))
```

Ahora también admitimos la lectura de archivos Avro y Parquet gracias a este [método](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/custom_executors/).

### Formatos de datos adicionales

Apache Beam admite la lectura de varios [formatos de datos adicionales](https://beam.apache.org/documentation/io/built-in/), a través de Beam I/O Transforms. Puede usar Beam I/O Transforms para crear componentes ExampleGen personalizados mediante el uso de un patrón similar al del [ejemplo de Avro.](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/custom_executors/avro_executor.py#L56)

```python
  return (pipeline
          | 'ReadFromAvro' >> beam.io.ReadFromAvro(avro_pattern)
          | 'ToTFExample' >> beam.Map(utils.dict_to_example))
```

Al momento de escribir este artículo, se incluyen los siguientes formatos y fuentes de datos actualmente admitidos para el SDK de Beam Python:

- Amazon S3
- Apache Avro
- Apache Hadoop
- Apache Kafka
- Apache Parquet
- Google Cloud BigQuery
- Google Cloud BigTable
- Google Cloud Datastore
- Google Cloud Pub/Sub
- Google Cloud Storage (GCS)
- MongoDB

Consulte los [documentos de Beam](https://beam.apache.org/documentation/io/built-in/) para obtener la lista más reciente.

### Personalización de ExampleGen basado en consultas (experimental)

En primer lugar, extienda BaseExampleGenExecutor con un Beam PTransform personalizado, que lea desde la fuente de datos externa. Luego, extienda QueryBasedExampleGen para crear un componente simple.

Esto puede requerir o no configuraciones de conexión adicionales. Por ejemplo, el [ejecutor BigQuery](https://github.com/tensorflow/tfx/blob/master/tfx/extensions/google_cloud_big_query/example_gen/executor.py) lee mediante un conector beam.io predeterminado, que abstrae los detalles de configuración de la conexión. El [ejecutor Presto](https://github.com/tensorflow/tfx/blob/master/tfx/examples/custom_components/presto_example_gen/presto_component/executor.py) requiere un Beam PTransform personalizado y un [protobuf de configuración de conexión personalizada](https://github.com/tensorflow/tfx/blob/master/tfx/examples/custom_components/presto_example_gen/proto/presto_config.proto) como entrada.

Si se requiere una configuración de conexión para un componente de ExampleGen personalizado, cree un nuevo protobuf y páselo a través de custom_config, que ahora es un parámetro de ejecución opcional. A continuación, se muestra un ejemplo de cómo usar un componente configurado.

```python
from tfx.examples.custom_components.presto_example_gen.proto import presto_config_pb2
from tfx.examples.custom_components.presto_example_gen.presto_component.component import PrestoExampleGen

presto_config = presto_config_pb2.PrestoConnConfig(host='localhost', port=8080)
example_gen = PrestoExampleGen(presto_config, query='SELECT * FROM chicago_taxi_trips')
```

## Componentes de flujo descendente de ExampleGen

Se admite la configuración dividida personalizada para componentes de flujo descendente.

### StatisticsGen

El comportamiento predeterminado consiste en generar estadísticas para todas las divisiones.

Para excluir cualquier división, configure `exclude_splits` para el componente StatisticsGen. Por ejemplo:

```python
# Exclude the 'eval' split.
statistics_gen = StatisticsGen(
             examples=example_gen.outputs['examples'],
             exclude_splits=['eval'])
```

### SchemaGen

El comportamiento predeterminado consiste en generar un esquema basado en todas las divisiones.

Para excluir cualquier división, establezca `exclude_splits` para el componente SchemaGen. Por ejemplo:

```python
# Exclude the 'eval' split.
schema_gen = SchemaGen(
             statistics=statistics_gen.outputs['statistics'],
             exclude_splits=['eval'])
```

### ExampleValidator

El comportamiento predeterminado consiste en validar las estadísticas de todas las divisiones en ejemplos de entrada con respecto a un esquema.

Para excluir cualquier división, establezca `exclude_splits` para el componente ExampleValidator. Por ejemplo:

```python
# Exclude the 'eval' split.
example_validator = ExampleValidator(
             statistics=statistics_gen.outputs['statistics'],
             schema=schema_gen.outputs['schema'],
             exclude_splits=['eval'])
```

### Transform

El comportamiento predeterminado consiste en analizar y producir los metadatos de la división 'entrenamiento' y transformar todas las divisiones.

Para especificar las divisiones de análisis y transformación, configure `splits_config` para el componente Transform. Por ejemplo:

```python
# Analyze the 'train' split and transform all splits.
transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=_taxi_module_file,
      splits_config=proto.SplitsConfig(analyze=['train'],
                                               transform=['train', 'eval']))
```

### Trainer y Tuner

El comportamiento predeterminado consiste en entrenar en la división 'entrenamiento' y evaluar en la división 'evaluación'.

Para especificar las divisiones de entrenamiento y evaluación, configure `train_args` y `eval_args` para el componente Trainer. Por ejemplo:

```python
# Train on the 'train' split and evaluate on the 'eval' split.
Trainer = Trainer(
      module_file=_taxi_module_file,
      examples=transform.outputs['transformed_examples'],
      schema=schema_gen.outputs['schema'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=proto.TrainArgs(splits=['train'], num_steps=10000),
      eval_args=proto.EvalArgs(splits=['eval'], num_steps=5000))
```

### Evaluator

El comportamiento predeterminado consiste en proporcionar métricas calculadas en la división 'evaluación'.

Para calcular estadísticas de evaluación en divisiones personalizadas, configure `example_splits` para el componente Evaluator. Por ejemplo:

```python
# Compute metrics on the 'eval1' split and the 'eval2' split.
evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      example_splits=['eval1', 'eval2'])
```

Si desea acceder a más información, consulte la [referencia de la API de CsvExampleGen](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/CsvExampleGen), la [implementación de la API de FileBasedExampleGen](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/component.py) y la [referencia de la API de ImportExampleGen](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/ImportExampleGen).
