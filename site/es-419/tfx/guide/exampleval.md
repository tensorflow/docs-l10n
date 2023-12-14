# El componente de canalización ExampleValidator TFX

El componente de canalización ExampleValidator identifica anomalías en el entrenamiento y el suministro de datos. Puede detectar diferentes clases de anomalías en los datos. Por ejemplo, puede ejecutar las siguientes acciones:

1. Llevar a cabo comprobaciones de validez mediante la comparación de estadísticas de datos con un esquema que codifica las expectativas del usuario.
2. Detectar el sesgo entrenamiento-servicio mediante la comparación de los datos de entrenamiento y servicio.
3. Detectar la desviación de datos mediante la observación de una serie de datos.
4. Llevar a cabo [validaciones personalizadas](https://github.com/tensorflow/data-validation/blob/master/g3doc/custom_data_validation.md) a partir de una configuración basada en SQL.

El componente de canalización ExampleValidator identifica cualquier anomalía en los datos de ejemplo a partir de la comparación de las estadísticas de datos calculadas por el componente de canalización StatisticsGen con un esquema. El esquema inferido codifica las propiedades que deberían satisfacer los datos de entrada y que el desarrollador puede modificar.

- Consume: un esquema de un componente SchemaGen y estadísticas de un componente StatisticsGen.
- Emite: resultados de validación

## ExampleValidator y TensorFlow Data Validation

ExampleValidator hace un uso extensivo de [TensorFlow Data Validation](tfdv.md) para validar sus datos de entrada.

## Cómo usar el componente ExampleValidator

Un componente de canalización ExampleValidator suele ser muy fácil de implementar y requiere poca personalización. El código típico se ve así:

```python
validate_stats = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema']
      )
```

Hay más detalles disponibles en la [referencia de la API de ExampleValidator](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/ExampleValidator).
