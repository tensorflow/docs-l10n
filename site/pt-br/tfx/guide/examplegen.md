# O componente de pipeline ExampleGen TFX

O componente de pipeline ExampleGen TFX consome dados em pipelines do TFX. Ele consome arquivos/serviços externos para gerar exemplos que serão lidos por outros componentes do TFX. Ele também fornece uma partição consistente e configurável e embaralha o dataset de acordo com as práticas recomendadas de ML.

- Consome: Dados de fontes de dados externas como CSV, `TFRecord`, Avro, Parquet e BigQuery.
- Produz: registros `tf.Example`, registros `tf.SequenceExample` ou formato proto, dependendo do formato da payload.

## ExampleGen e outros componentes

O ExampleGen fornece dados para componentes que fazem uso da biblioteca [TensorFlow Data Validation](tfdv.md), como [SchemaGen](schemagen.md), [StatisticsGen](statsgen.md) e [Example Validator](exampleval.md). Ele também fornece dados para [Transform](transform.md), que faz uso da biblioteca [TensorFlow Transform](tft.md) e, em última análise, para alvos de implantação durante a inferência.

## Fontes e formatos de dados

Atualmente, uma instalação padrão do TFX inclui componentes completos do ExampleGen para estas fontes e formatos de dados:

- [CSV](https://github.com/tensorflow/tfx/tree/master/tfx/components/example_gen/csv_example_gen)
- [tf.Record](https://github.com/tensorflow/tfx/tree/master/tfx/components/example_gen/import_example_gen)
- [BigQuery](https://github.com/tensorflow/tfx/tree/master/tfx/extensions/google_cloud_big_query/example_gen)

Também estão disponíveis executores personalizados que permitem o desenvolvimento de componentes ExampleGen para estas fontes e formatos de dados:

- [Avro](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/custom_executors/avro_executor.py)
- [Parquet](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/custom_executors/parquet_executor.py)

Consulte os exemplos de uso no código-fonte e [esta discussão](/tfx/guide/examplegen#custom_examplegen) para mais informações sobre como usar e desenvolver executores personalizados.

Observação: Na maioria dos casos, é melhor herdar de `base_example_gen_executor` em vez de `base_executor`. Portanto, pode ser aconselhável seguir o exemplo Avro ou Parquet no código-fonte do Executor.

Além disso, estas fontes e formatos de dados estão disponíveis como exemplos de [componentes personalizados](/tfx/guide/understanding_custom_components):

- [Presto](https://github.com/tensorflow/tfx/tree/master/tfx/examples/custom_components/presto_example_gen)

### Consumindo formatos de dados suportados pelo Apache Beam

O Apache Beam oferece suporte ao consumo de dados de uma [ampla variedade de fontes e formatos de dados](https://beam.apache.org/documentation/io/built-in/) ([veja abaixo](#additional_data_formats)). Esses recursos podem ser usados ​​para criar componentes Example Gen personalizados para o TFX, o que é demonstrado por alguns componentes ExampleGen existentes ([veja abaixo](#additional_data_formats)).

## Como usar um componente ExampleGen

Para fontes de dados suportadas (atualmente, arquivos CSV, arquivos TFRecord com `tf.Example`, `tf.SequenceExample` e formato proto e resultados de consultas do BigQuery), o componente de pipeline ExampleGen pode ser usado diretamente na implantação e requer pouca personalização. Por exemplo:

```python
example_gen = CsvExampleGen(input_base='data_root')
```

ou como abaixo para importar um TFRecord externo diretamente com `tf.Example`:

```python
example_gen = ImportExampleGen(input_base=path_to_tfrecord_dir)
```

## Span, Version e Split

Um Span é um agrupamento de exemplos de treinamento. Se seus dados são armazenados num sistema de arquivos, cada Span poderá ser armazenado num diretório separado. A semântica de um Span não é codificada no TFX; um Span pode corresponder a um dia de dados, uma hora de dados ou qualquer outro agrupamento que seja significativo para sua tarefa.

Cada Span pode conter múltiplas versões (Version) de dados. Para dar um exemplo, se você remover alguns exemplos de um Span para limpar dados de baixa qualidade, isso poderá resultar numa nova Version desse Span. Por padrão, os componentes do TFX operam na Version mais recente dentro de um Span.

Cada Version dentro de um Span pode ainda ser subdividida em múltiplas Splits (divisões). O caso de uso mais comum para dividir um Span é dividi-lo em dados de treinamento (train) e avaliação (eval).

![Spans e Splits](images/spans_splits.png)

### Divisão (split) personalizada de entrada/saída

Observação: este recurso só está disponível para versões posteriores ao TFX 0.14.

Para personalizar a proporção da divisão de treinamento/avaliação que o ExampleGen produzirá, defina o `output_config` para o componente ExampleGen. Por exemplo:

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

Observe como os `hash_buckets` foram definidos neste exemplo.

Para uma fonte de entrada que já foi dividida, defina `input_config` para o componente ExampleGen:

```python

# Input train split is 'input_dir/train/*', eval split is 'input_dir/eval/*'.
# Output splits are generated one-to-one mapping from input splits.
input = proto.Input(splits=[
                example_gen_pb2.Input.Split(name='train', pattern='train/*'),
                example_gen_pb2.Input.Split(name='eval', pattern='eval/*')
            ])
example_gen = CsvExampleGen(input_base=input_dir, input_config=input)
```

Para geração de exemplo baseada em arquivo (por exemplo, CsvExampleGen e ImportExampleGen), `pattern` é um padrão de arquivo glob relativo que mapeia para arquivos de entrada com diretório raiz fornecido pelo caminho base de entrada. Para geração de exemplos com base em consultas (por exemplo, BigQueryExampleGen, PrestoExampleGen), o `pattern` é uma consulta SQL.

Por padrão, todo o diretório base de entrada é tratado como uma única divisão de entrada, e a divisão de saída treinamento/avaliação é gerada com uma proporção de 2:1.

Consulte [proto/example_gen.proto](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto) para obter a configuração de divisão de entrada e saída do ExampleGen. E consulte o [guia de componentes downstream](#examplegen_downstream_components) para utilizar as divisões personalizadas downstream.

#### Método de divisão

Ao usar o método de divisão `hash_buckets`, em vez do registro inteiro, pode-se usar uma característica para particionar os exemplos. Se uma característica estiver presente, o ExampleGen usará uma impressão digital dessa característica como chave de partição.

Este característica pode ser usada para manter uma divisão estável com certas propriedades de exemplos: por exemplo, um usuário sempre será colocado na mesma divisão se "user_id" for selecionado como o nome da característica de partição.

A interpretação do que significa uma "característica" e como corresponder uma "característica" ao nome especificado depende da implementação do ExampleGen e do tipo dos exemplos.

Para implementações de ExampleGen prontas:

- Se gerar tf.Example, então uma "característica" significa uma entrada em tf.Example.features.feature.
- Se gerar tf.SequenceExample, então uma "característica" significa uma entrada em tf.SequenceExample.context.feature.
- Somente característica int64 e bytes são suportadas.

Nos seguintes casos, o ExampleGen gera erros de tempo de execução:

- O nome da característica especificada não existe no exemplo.
- Característica vazia: `tf.train.Feature()`.
- Tipos de característica não suportados, por exemplo, característica do tipo float.

Para gerar a divisão train/eval com base em uma característica nos exemplos, defina o `output_config` para o componente ExampleGen. Por exemplo:

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

Observe como o `partition_feature_name` foi definido neste exemplo.

### Span

Observação: este recurso só está disponível para versões posteriores ao TFX 0.15.

O span pode ser recuperado usando a especificação '{SPAN}' no [padrão de entrada glob](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto) :

- Esta especificação combina dígitos e mapeia os dados nos números SPAN relevantes. Por exemplo, 'data_{SPAN}-*.tfrecord' coletará arquivos como 'data_12-a.tfrecord', 'date_12-b.tfrecord'.
- Opcionalmente, esta especificação pode ser especificada com a largura dos inteiros quando mapeada. Por exemplo, 'data_{SPAN:2}.file' mapeia para arquivos como 'data_02.file' e 'data_27.file' (como entradas para Span-2 e Span-27 respectivamente), mas não mapeia para 'data_1. arquivo' nem 'data_123.file'.
- Quando a especificação SPAN está faltando, presume-se que seja sempre Span '0'.
- Se SPAN for especificado, o pipeline processará o intervalo mais recente e armazenará o número do span nos metadados.

Por exemplo, vamos supor que existam os seguintes dados de entrada:

- '/tmp/span-1/train/data'
- '/tmp/span-1/eval/data'
- '/tmp/span-2/train/data'
- '/tmp/span-2/eval/data'

e a configuração de entrada mostrada abaixo:

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

ao acionar o pipeline, ele processará:

- '/tmp/span-2/train/data' como uma divisão de treinamento (train)
- '/tmp/span-2/eval/data' como uma divisão de avaliação (eval)

com o número de span igual a '2'. Se mais tarde '/tmp/span-3/...' estiverem prontos, basta acionar o pipeline novamente e ele pegará o span '3' para processamento. Abaixo está mostrado um exemplo de código usando a especificação span:

```python
input = proto.Input(splits=[
                proto.Input.Split(name='train',
                                            pattern='span-{SPAN}/train/*'),
                proto.Input.Split(name='eval',
                                            pattern='span-{SPAN}/eval/*')
            ])
example_gen = CsvExampleGen(input_base='/tmp', input_config=input)
```

A recuperação de um determinado span pode ser feita com RangeConfig, detalhado a seguir.

### Date

Observação: este recurso só está disponível para versões posteriores ao TFX 0.24.0.

Se a sua fonte de dados estiver organizada no sistema de arquivos por data, o TFX oferece suporte ao mapeamento direto de datas para números de span. Existem três especificações para representar o mapeamento de datas para spans: {YYYY}, {MM} e {DD}:

- As três especificações devem estar presentes no [padrão de entrada glob](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto), se alguma for especificada:
- Ou a especificação {SPAN} ou este conjunto de especificações de data podem ser especificados exclusivamente.
- Uma data de calendário com o ano a partir de YYYY, o mês a partir de MM e o dia do mês a partir de DD é calculada, então o número do span é calculado como o número de dias desde a época Unix (ou seja, 1970-01-01). Por exemplo, 'log-{YYYY}{MM}{DD}.data' corresponde a um arquivo 'log-19700101.data' e o consome como entrada para Span-0, e 'log-20170101.data' como entrada para Span-17167.
- Se este conjunto de especificações de data for especificado, o pipeline processará a data mais recente e armazenará o número de span correspondente nos metadados.

Por exemplo, vamos supor que existam os seguintes dados de entrada organizados por data do calendário:

- '/tmp/1970-01-02/train/data'
- '/tmp/1970-01-02/eval/data'
- '/tmp/1970-01-03/train/data'
- '/tmp/1970-01-03/eval/data'

e a configuração de entrada mostrada abaixo:

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

ao acionar o pipeline, ele processará:

- '/tmp/1970-01-03/train/data' como uma divisão de treinamento (train)
- '/tmp/1970-01-03/eval/data' como uma divisão de avaliação (eval)

com número de span igual '2'. Se mais tarde '/tmp/1970-01-04/...' estiverem prontos, basta acionar o pipeline novamente e ele pegará o span '3' para processamento. Abaixo está mostrado um exemplo de código que usa a especificação Date:

```python
input = proto.Input(splits=[
                proto.Input.Split(name='train',
                                            pattern='{YYYY}-{MM}-{DD}/train/*'),
                proto.Input.Split(name='eval',
                                            pattern='{YYYY}-{MM}-{DD}/eval/*')
            ])
example_gen = CsvExampleGen(input_base='/tmp', input_config=input)
```

### Version

Observação: este recurso só está disponível para versões posteriores ao TFX 0.24.0.

A versão pode ser recuperada usando a especificação '{VERSION}' no [padrão de entrada glob](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto):

- Esta especificação combina dígitos e mapeia os dados para os números de VERSÃO relevantes no SPAN. Observe que a especificação Version pode ser usada em combinação com a especificação Span ou Data.
- Esta especificação também pode ser opcionalmente especificada com a largura da mesma forma que a especificação SPAN. por exemplo, 'span-{SPAN}/version-{VERSION:4}/data-*'.
- Quando a especificação VERSION está faltando, a versão é definida como None.
- Se SPAN e VERSION forem especificados, o pipeline processará a versão mais recente para o span mais recente e armazenará o número da versão nos metadados.
- Se VERSION for especificado, mas não SPAN (ou especificação Date), um erro será gerado.

Por exemplo, vamos supor que existam os seguintes dados de entrada:

- '/tmp/span-1/ver-1/train/data'
- '/tmp/span-1/ver-1/eval/data'
- '/tmp/span-2/ver-1/train/data'
- '/tmp/span-2/ver-1/eval/data'
- '/tmp/span-2/ver-2/train/data'
- '/tmp/span-2/ver-2/eval/data'

e a configuração de entrada mostrada abaixo:

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

ao acionar o pipeline, ele processará:

- '/tmp/span-2/ver-2/train/data' como uma divisão de treinamento (train)
- '/tmp/span-2/ver-2/train/data' como uma divisão de avaliação (eval)

com número de span igual a '2' e número de versão igual a '2'. Se mais tarde '/tmp/span-2/ver-3/...' estiverem prontos, basta acionar o pipeline novamente e ele selecionará o span '2' e a versão '3' para processamento. Abaixo está mostrado um exemplo de código que usa a especificação Version:

```python
input = proto.Input(splits=[
                proto.Input.Split(name='train',
                                            pattern='span-{SPAN}/ver-{VERSION}/train/*'),
                proto.Input.Split(name='eval',
                                            pattern='span-{SPAN}/ver-{VERSION}/eval/*')
            ])
example_gen = CsvExampleGen(input_base='/tmp', input_config=input)
```

### Range Config (configuração de intervalo)

Observação: este recurso só está disponível para versões posteriores ao TFX 0.24.0.

O TFX oferece suporte à recuperação e ao processamento de um span específico em ExampleGen baseado em arquivo usando range config (configuração de intervalo), uma configuração abstrata usada para descrever intervalos (ranges) para diferentes entidades TFX. Para recuperar um span específico, defina `range_config` para um componente ExampleGen baseado em arquivo. Por exemplo, vamos supor que existam os seguintes dados de entrada:

- '/tmp/span-01/train/data'
- '/tmp/span-01/eval/data'
- '/tmp/span-02/train/data'
- '/tmp/span-02/eval/data'

Para recuperar e processar especificamente dados com span '1', especificamos uma configuração de intervalo (range config) além da configuração de entrada. Observe que ExampleGen oferece suporte apenas a intervalos estáticos de span único (para especificar o processamento de spans individuais específicos). Assim, para StaticRange, start_span_number deve ser igual a end_span_number. Usando o span fornecido e as informações de largura do span (se fornecidas) para preenchimento com zeros, o ExampleGen substituirá a especificação SPAN nos padrões de divisão fornecidos pelo número do span desejado. Um exemplo de uso é mostrado abaixo:

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

A configuração de intervalo também pode ser usada para processar datas específicas, se a especificação de data for usada em vez da especificação SPAN. Por exemplo, vamos supor que existam dados de entrada organizados pela data de calendário:

- '/tmp/1970-01-02/train/data'
- '/tmp/1970-01-02/eval/data'
- '/tmp/1970-01-03/train/data'
- '/tmp/1970-01-03/eval/data'

Para recuperar e processar especificamente dados de 2 de janeiro de 1970, fazemos o seguinte:

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

Se os componentes do ExampleGen atualmente disponíveis não atenderem às suas necessidades, você poderá criar um ExampleGen personalizado, que permitirá a leitura de diferentes fontes de dados ou em diferentes formatos de dados.

### Personalização de ExampleGen baseada em arquivo (experimental)

Primeiro, estenda BaseExampleGenExecutor com um Beam PTransform personalizado, que fornece a conversão de sua divisão de entrada de treinamento/avaliação para exemplos do TF. Por exemplo, o [executor CsvExampleGen](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/csv_example_gen/executor.py) fornece a conversão de uma divisão CSV de entrada para exemplos do TF.

Em seguida, crie um componente com o executor acima, como feito em [CsvExampleGen component](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/csv_example_gen/component.py). Como alternativa, passe um executor personalizado para o componente ExampleGen padrão conforme mostrado abaixo.

```python
from tfx.components.base import executor_spec
from tfx.components.example_gen.csv_example_gen import executor

example_gen = FileBasedExampleGen(
    input_base=os.path.join(base_dir, 'data/simple'),
    custom_executor_spec=executor_spec.ExecutorClassSpec(executor.Executor))
```

Agora também oferecemos suporte à leitura de arquivos Avro e Parquet usando este [método](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/custom_executors/).

### Formatos de dados adicionais

O Apache Beam oferece suporte à leitura de vários [formatos de dados adicionais](https://beam.apache.org/documentation/io/built-in/) através de transformações de E/S do Beam (Beam I/O Transforms). Você pode criar componentes ExampleGen personalizados aproveitando o Beam I/O Transforms usando um padrão semelhante ao do [exemplo de leitura de arquivos Avro](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/custom_executors/avro_executor.py#L56)

```python
  return (pipeline
          | 'ReadFromAvro' >> beam.io.ReadFromAvro(avro_pattern)
          | 'ToTFExample' >> beam.Map(utils.dict_to_example))
```

No momento em que este artigo foi escrito, os formatos e fontes de dados atualmente suportados pelo Beam Python SDK são os seguintes:

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

Confira a [Documentação do Beam](https://beam.apache.org/documentation/io/built-in/) para obter a lista mais recente.

### Personalização de ExampleGen baseada em consulta (experimental)

Primeiro, estenda o BaseExampleGenExecutor com um Beam PTransform personalizado, que lê a fonte de dados externa. Em seguida, crie um componente simples estendendo QueryBasedExampleGen.

Isto poderá ou não exigir configurações de conexão adicionais. Por exemplo, o [executor BigQuery](https://github.com/tensorflow/tfx/blob/master/tfx/extensions/google_cloud_big_query/example_gen/executor.py) lê usando um conector beam.io padrão, que abstrai os detalhes de configuração da conexão. O [executor Presto](https://github.com/tensorflow/tfx/blob/master/tfx/examples/custom_components/presto_example_gen/presto_component/executor.py) requer um Beam PTransform personalizado e um [protobuf de configuração de conexão personalizada](https://github.com/tensorflow/tfx/blob/master/tfx/examples/custom_components/presto_example_gen/proto/presto_config.proto) como entrada.

Se uma configuração de conexão for necessária para um componente ExampleGen personalizado, crie um novo protobuf e passe-o via custom_config, que agora é um parâmetro de execução opcional. Abaixo está um exemplo de como usar um componente configurado.

```python
from tfx.examples.custom_components.presto_example_gen.proto import presto_config_pb2
from tfx.examples.custom_components.presto_example_gen.presto_component.component import PrestoExampleGen

presto_config = presto_config_pb2.PrestoConnConfig(host='localhost', port=8080)
example_gen = PrestoExampleGen(presto_config, query='SELECT * FROM chicago_taxi_trips')
```

## Componentes downstream do ExampleGen

A configuração personalizada de divisões (splits) é suportada para componentes que aparecem no pipeline depois do ExampleGen (componentes downstream).

### StatisticsGen

O comportamento padrão é gerar estatísticas para todas as divisões.

Para excluir quaisquer divisões, defina `exclude_splits` para o componente StatisticsGen. Por exemplo:

```python
# Exclude the 'eval' split.
statistics_gen = StatisticsGen(
             examples=example_gen.outputs['examples'],
             exclude_splits=['eval'])
```

### SchemaGen

O comportamento padrão é gerar um esquema baseado em todas as divisões.

Para excluir quaisquer divisões, defina `exclude_splits` para o componente SchemaGen. Por exemplo:

```python
# Exclude the 'eval' split.
schema_gen = SchemaGen(
             statistics=statistics_gen.outputs['statistics'],
             exclude_splits=['eval'])
```

### ExampleValidator

O comportamento padrão é validar as estatísticas de todas as divisões em exemplos de entrada em relação a um esquema.

Para excluir quaisquer divisões, defina `exclude_splits` para o componente ExampleValidator. Por exemplo:

```python
# Exclude the 'eval' split.
example_validator = ExampleValidator(
             statistics=statistics_gen.outputs['statistics'],
             schema=schema_gen.outputs['schema'],
             exclude_splits=['eval'])
```

### Transform

O comportamento padrão é analisar e produzir os metadados da divisão 'train' e transformar todas as divisões.

Para especificar as divisões de análise e de transformação, defina `splits_config` para o componente Transform. Por exemplo:

```python
# Analyze the 'train' split and transform all splits.
transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=_taxi_module_file,
      splits_config=proto.SplitsConfig(analyze=['train'],
                                               transform=['train', 'eval']))
```

### Trainer e Tuner

O comportamento padrão é treinar na divisão 'train' e avaliar na divisão 'eval'.

Para especificar as divisões de treinamento e avaliação, defina `train_args` e `eval_args` para o componente Trainer. Por exemplo:

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

O comportamento padrão é fornecer métricas computadas na divisão 'eval'.

Para computar estatísticas de avaliação em divisões personalizadas, defina `example_splits` para o componente Evaluator. Por exemplo:

```python
# Compute metrics on the 'eval1' split and the 'eval2' split.
evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      example_splits=['eval1', 'eval2'])
```

Mais detalhes estão disponíveis na [Referência da API do CsvExampleGen](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/CsvExampleGen), na [Implementação da API FileBasedExampleGen](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/component.py) e na [Referência da API ImportExampleGen](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/ImportExampleGen).
