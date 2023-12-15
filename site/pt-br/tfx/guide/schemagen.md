# O componente de pipeline SchemaGen TFX

Alguns componentes do TFX usam uma descrição dos seus dados de entrada, que é chamada de *esquema* (schema). O esquema é uma instância de [schema.proto](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto). Ele pode especificar os tipos de dados para valores de características, se uma característica deve estar presente em todos os exemplos, intervalos de valores permitidos e outras propriedades. Um componente de pipeline SchemaGen gerará um esquema automaticamente, inferindo tipos, categorias e intervalos dos dados de treinamento.

- Consome: estatísticas de um componente StatisticsGen
- Produz: Data schema proto

Aqui está o trecho de um schema proto:

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

As seguintes bibliotecas TFX usam o esquema:

- TensorFlow Data Validation
- TensorFlow Transform
- TensorFlow Model Analysis

Num pipeline TFX típico, o SchemaGen gera um esquema, que é consumido pelos outros componentes do pipeline. No entanto, o esquema gerado automaticamente é o melhor esforço e tenta apenas inferir propriedades básicas dos dados. Espera-se que os desenvolvedores o revisem e modifiquem conforme seja necessário.

O esquema modificado pode ser trazido de volta ao pipeline usando o componente ImportSchemaGen. O componente SchemaGen para a geração inicial do esquema pode ser removido e todos os componentes downstream podem usar a saída de ImportSchemaGen. Também é recomendado adicionar [ExampleValidator](https://www.tensorflow.org/tfx/guide/exampleval) usando o esquema importado para examinar os dados de treinamento continuamente.

## SchemaGen e TensorFlow Data Validation

O SchemaGen faz uso extensivo do [TensorFlow Data Validation](tfdv.md) para inferir um esquema.

## Usando o componente SchemaGen

### Para a geração inicial do esquema

Um componente de pipeline SchemaGen geralmente é muito fácil de implantar e requer pouca personalização. O código típico está mostrado a seguir:

```python
schema_gen = tfx.components.SchemaGen(
    statistics=stats_gen.outputs['statistics'])
```

Mais detalhes estão disponíveis na [Referência da API SchemaGen](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/SchemaGen).

### Para a importação do esquema revisado

Adicione o componente ImportSchemaGen ao pipeline para trazer a definição de esquema revisada para dentro do pipeline.

```python
schema_gen = tfx.components.ImportSchemaGen(
    schema_file='/some/path/schema.pbtxt')
```

O `schema_file` deve ser um caminho completo para o arquivo de texto protobuf.

Mais detalhes estão disponíveis na [referência da API ImportSchemaGen](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/ImportSchemaGen).
