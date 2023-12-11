# O componente de pipeline StatisticsGen TFX

O componente de pipeline StatisticsGen TFX gera estatísticas de características sobre dados de treinamento e serviço, que podem ser usados ​​por outros componentes de pipeline. O StatisticsGen usa o Beam para escalar grandes datasets.

- Consome: datasets criados por um componente de pipeline ExampleGen.
- Produz: estatísticas do dataset.

## StatisticsGen e TensorFlow Data Validation

O StatisticsGen faz uso extensivo do [TensorFlow Data Validation](tfdv.md) para gerar estatísticas a partir do seu dataset.

## Usando o componente StatsGen

Um componente de pipeline StatisticsGen geralmente é muito fácil de implantar e requer pouca personalização. O código típico está mostrado a seguir:

```python
compute_eval_stats = StatisticsGen(
      examples=example_gen.outputs['examples'],
      name='compute-eval-stats'
      )
```

## Usando o componente StatsGen com um esquema

Para a primeira execução de um pipeline, a saída do StatisticsGen será usada para inferir um esquema. No entanto, em execuções subsequentes você poderá ter um esquema curado manualmente que contém informações adicionais sobre seu dataset. Ao fornecer este esquema ao StatisticsGen, o TFDV poderá fornecer estatísticas mais úteis com base nas propriedades declaradas do seu dataset.

Nesta configuração, você invocará o StatisticsGen com um esquema selecionado que foi importado por um ImporterNode da seguinte forma:

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

### Criando um esquema curado

`Schema`, no TFX, é uma instância de <a href="https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto" data-md-type="link">`Schema` proto</a> do TensorFlow Metadata. Ele pode ser escrito em [formato texto](https://googleapis.dev/python/protobuf/latest/google/protobuf/text_format.html) do zero. No entanto, é mais fácil usar o esquema inferido produzido pelo `SchemaGen` como ponto de partida. Depois que o componente `SchemaGen` for executado, o esquema estará localizado na raiz do pipeline no seguinte caminho:

```
<pipeline_root>/SchemaGen/schema/<artifact_id>/schema.pbtxt
```

Onde `<artifact_id>` representa um ID exclusivo para esta versão do esquema no MLMD. Este schema proto pode então ser modificado para comunicar informações sobre o dataset que não podem ser inferidas de forma confiável, o que deixará a saída do `StatisticsGen` mais útil e a validação realizada no componente [`ExampleValidator`](https://www.tensorflow.org/tfx/guide/exampleval) mais rigorosa.

Mais detalhes estão disponíveis na [Referência da API StatisticsGen](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/StatisticsGen).
