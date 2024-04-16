# El componente de canalización StatisticsGen TFX

El componente de canalización StatisticsGen TFX genera estadísticas de características sobre datos de entrenamiento y servicio, que pueden ser utilizadas por otros componentes de canalización. StatisticsGen usa Beam para escalar a grandes conjuntos de datos.

- Consume: conjuntos de datos creados por un componente de canalización ExampleGen.
- Emite: estadísticas del conjunto de datos.

## StatisticsGen y TensorFlow Data Validation

StatisticsGen hace un uso extensivo de [TensorFlow Data Validation](tfdv.md) para validar sus datos de entrada.

## Cómo usar el componente StatsGen

Un componente de canalización StatsGen suele ser muy fácil de implementar y requiere muy poca personalización. El código típico se ve así:

```python
compute_eval_stats = StatisticsGen(
      examples=example_gen.outputs['examples'],
      name='compute-eval-stats'
      )
```

## Cómo usar el componente StatsGen con un esquema

Para la primera ejecución de una canalización, se usará el resultado de StatisticsGen para inferir un esquema. Sin embargo, en ejecuciones posteriores es posible que tenga un esquema seleccionado manualmente que contenga información adicional sobre su conjunto de datos. Al proporcionar este esquema a StatisticsGen, TFDV puede proporcionar estadísticas más útiles basadas en las propiedades declaradas de su conjunto de datos.

En esta configuración, invocará StatisticsGen con un esquema seleccionado que ha sido importado por un ImporterNode como este:

```python
user_schema_importer = Importer(
    source_uri=user_schema_dir, # directory containing only schema text proto
    artifact_type=standard_artifacts.Schema).with_id('schema_importer')

compute_eval_stats = StatisticsGen(
      examples=example_gen.outputs['examples'],
      schema=user_schema_importer.outputs['result'],
      name='compute-eval-stats'
      )
```

### Cómo crear un esquema seleccionado

`Schema` en TFX es una instancia del <a data-md-type="raw_html" href="https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto">proto de `Schema`</a> de metadatos de TensorFlow. Este se puede redactar en [formato de texto](https://googleapis.dev/python/protobuf/latest/google/protobuf/text_format.html) desde cero. Sin embargo, es más fácil usar el esquema inferido que produce `SchemaGen` como punto de partida. Una vez que se haya ejecutado el componente `SchemaGen`, el esquema se ubicará debajo de la raíz de la canalización en la siguiente ruta:

```
<pipeline_root>/SchemaGen/schema/<artifact_id>/schema.pbtxt
```

Donde `<artifact_id>` representa una ID única para esta versión del esquema en MLMD. Luego, este protocolo de esquema se puede modificar para comunicar información sobre el conjunto de datos que no se puede inferir de manera confiable, lo que hará que la salida de `StatisticsGen` sea más útil y la validación realizada en el componente [`ExampleValidator`](https://www.tensorflow.org/tfx/guide/exampleval) más estricta.

Hay más detalles disponibles en la [referencia de la API de StatisticsGen](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/StatisticsGen).
