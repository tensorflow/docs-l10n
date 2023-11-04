# O componente de pipeline ExampleValidator TFX

O componente de pipeline ExampleValidator identifica anomalias no treinamento e no fornecimento de dados. Ele pode detectar diferentes classes de anomalias nos dados. Por exemplo, ele pode:

1. realizar verificações de validade comparando estatísticas de dados com um esquema que codifica as expectativas do usuário.
2. detectar desvios no fornecimento de treinamento comparando dados de treinamento e fornecimento.
3. detectar desvios de dados observando uma série de dados.
4. executar [validações personalizadas](https://github.com/tensorflow/data-validation/blob/master/g3doc/custom_data_validation.md) usando uma configuração baseada em SQL.

O componente de pipeline ExampleValidator identifica quaisquer anomalias nos dados de exemplo comparando estatísticas de dados calculadas pelo componente de pipeline StatisticsGen contra um esquema. O esquema inferido codifica propriedades que se espera que os dados de entrada satisfaçam e podem ser modificados pelo desenvolvedor.

- Consome: o esquema de um componente SchemaGen e estatísticas de um componente StatisticsGen.
- Produz: resultados de validação

## ExampleValidator e TensorFlow Data Validation

O ExampleValidator faz uso extensivo do [TensorFlow Data Validation](tfdv.md) para validar seus dados de entrada.

## Usando o componente ExampleValidator

Um componente de pipeline ExampleValidator normalmente é muito fácil de implantar e requer pouca personalização. O código típico está mostrado a seguir:

```python
validate_stats = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema']
      )
```

Mais detalhes estão disponíveis na [Referência da API ExampleValidator](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/ExampleValidator).
