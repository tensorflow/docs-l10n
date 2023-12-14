# El componente de canalización SchemaGen TFX

Algunos componentes de TFX usan una descripción de sus datos de entrada conocida como *esquema*. El esquema es una instancia de [schema.proto](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto). Puede especificar tipos de datos para valores de características, si una característica debe estar presente en todos los ejemplos, rangos de valores permitidos y otras propiedades. Un componente de canalización SchemaGen generará automáticamente un esquema al inferir tipos, categorías y rangos a partir de los datos de entrenamiento.

- Consume: estadísticas de un componente StatisticsGen
- Emite: protocolo de esquema de datos

Aquí hay un extracto de un protocolo de esquema:

```proto
...
feature {
  name: "age"
  value_count {
    min: 1
    max: 1
  }
  type: FLOAT
  presence {
    min_fraction: 1
    min_count: 1
  }
}
feature {
  name: "capital-gain"
  value_count {
    min: 1
    max: 1
  }
  type: FLOAT
  presence {
    min_fraction: 1
    min_count: 1
  }
}
...
```

Las siguientes bibliotecas de TFX usan esquemas:

- TensorFlow Data Validation
- TensorFlow Transform
- TensorFlow Model Analysis

En una canalización típica de TFX, SchemaGen genera un esquema, que es consumido por los otros componentes de la canalización. Sin embargo, el esquema generado automáticamente es el mejor esfuerzo y solo intenta inferir propiedades básicas de los datos. Se espera que los desarrolladores lo revisen y hagan todos los cambios necesarios.

El esquema modificado se puede devolver a la canalización mediante el componente ImportSchemaGen. El componente SchemaGen para la generación del esquema inicial se puede eliminar y todos los componentes posteriores pueden usar la salida de ImportSchemaGen. También se recomienda agregar [ExampleValidator](https://www.tensorflow.org/tfx/guide/exampleval) con el esquema importado para examinar los datos de entrenamiento continuamente.

## SchemaGen y TensorFlow Data Validation

SchemaGen hace un uso extensivo de [TensorFlow Data Validation](tfdv.md) para inferir un esquema.

## Cómo usar el componente SchemaGen

### Para la generación inicial del esquema

Un componente de canalización de SchemaGen suele ser muy fácil de implementar y requiere muy poca personalización. El código típico se ve así:

```python
schema_gen = tfx.components.SchemaGen(
    statistics=stats_gen.outputs['statistics'])
```

Hay más detalles disponibles en la [referencia de la API de SchemaGen](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/SchemaGen).

### Para la importación del esquema revisado

Agregue el componente ImportSchemaGen a la canalización para incorporar la definición del esquema revisado a la canalización.

```python
schema_gen = tfx.components.ImportSchemaGen(
    schema_file='/some/path/schema.pbtxt')
```

El `schema_file` debe ser una ruta completa al archivo probuf de texto.

Hay más detalles disponibles en la [referencia de la API de ImportSchemaGen](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/ImportSchemaGen).
