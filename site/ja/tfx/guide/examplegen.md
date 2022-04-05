# ExampleGen TFX パイプラインコンポーネント

ExampleGen TFX パイプラインコンポーネントは、データを TFX パイプラインに取り込むコンポーネントです。ほかの TFX コンポーネントが読み取る Example を生成するために、外部のファイルやサービスを消費します。また、一貫性のある構成可能なパーティションを提供し、ML のベストプラクティスを実践できるよう、データセットをシャッフルします。

- 入力: CSV や `TFRecord`、Avro、Parquet、BigQuery といった外部のデータソースからのデータ
- 出力: ペイロード形式に応じて、`tf.Example` レコード、`tf.SequenceExample` レコード、または写真形式

## ExampleGen とその他のコンポーネント

ExampleGen は [TensorFlow Data Validation](tfdv.md) ライブラリを利用するコンポーネントにデータを提供します。 これには [SchemaGen](schemagen.md)、[StatisticsGen](statsgen.md)、[Example Validator](exampleval.md) が含まれます。 また、[TensorFlow Transform](transform.md) ライブラリを利用する [Transform](tft.md) にもデータを提供し、最終的には推論時にデプロイターゲットへとデータを供給します。

## ExampleGen コンポーネントの使い方

現在、標準的な TFX のインストールには、以下のデータソースと形式をサポートする完全な ExampleGen コンポーネントが含まれています。

- CSV
- tf.Record
- BigQuery

以下のデータソースと形式をサポートする ExampleGen コンポーネントの開発を可能にするカスタム Executor も利用できます。

- Avro
- Parquet

カスタム Executor の使用方法と開発方法についての詳細は、ソースコードに含まれる使用例と[こちらのディスカッション](/tfx/guide/examplegen#custom_examplegen)をご覧ください。

注意: ほとんどの場合、`base_executor` ではなく `base_example_gen_executor` から継承する方がよい結果を生み出します。そのため、Executor ソースコードの Avro または Parquet の例に従うことをお勧めします。

また、次のデータソースと形式は[カスタムコンポーネント](/tfx/guide/understanding_custom_components)の例としても利用できます。

- [Presto](https://github.com/tensorflow/tfx/tree/master/tfx/examples/custom_components/presto_example_gen)

### Apache Beam でサポートされている取り込みデータ形式

Apache Beam は、[多様なデータソースと形式](https://beam.apache.org/documentation/io/built-in/)のデータを取り込むことができます（[以下をご覧ください](#additional_data_formats)）。これらの機能は、TFX のカスタム ExampleGen コンポーネントを作成するために使用することができます。これは、既存の ExampleGen コンポーネントで実演されています（[以下をご覧ください](#additional_data_formats)）。

## ExampleGen コンポーネントの使用方法

サポートされるデータソース（現在、CSV ファイル、`tf.Example`、`tf.SequenceExample`、および proto フォーマットの TFRecord ファイル、BigQuery のクエリ結果の 3 つがサポートされています）を用いる場合、ExampleGen パイプラインコンポーネントは直接デプロイに使用可能で、カスタマイズはほとんど不要です。

```python
example_gen = CsvExampleGen(input_base='data_root')
```

または以下のように、`tf.Example` で外部の TFRecord を直接インポートします。

```python
example_gen = ImportExampleGen(input_base=path_to_tfrecord_dir)
```

## Span、Version、Split

Span はトレーニングの Example のグループを指します。ファイルシステム上のデータが永続である場合、Span ごとに個別のディレクトリに保存されます。Span のセマンティクスは TFX にハードコーディングされていないため、1 日分のデータ、1 時間分のデータ、またはタスクに意味のあるグループに対応付けられます。

Span にはそれぞれ複数のバージョンのデータが保存されることがあります。たとえば、質の悪いデータをクリーンアップするために Span から一部の Example を削除した場合、Span の新しい Version が作成されます。デフォルトでは、TFX コンポーネントは Span 内の最新の Version を使用します。

Span 内の各バージョンを、さらに複数の Split に分割することができます。Span を training データと eval データに分割するのが最も一般的なユースケースです。

![Spans and Splits](images/spans_splits.png)

### カスタム入力/出力 Split

注意: この機能は、TFX 0.14 以降でのみ利用できます。

ExampleGen が出力する train と eval の分割率をカスタマイズするには、ExampleGen コンポーネントの `output_config` を設定します。以下に例を示します。

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

上記の例で `hash_buckets` が設定されているところに注目してください。

分割済みの入力ソースについては、ExampleGen コンポーネントの `input_config` を設定します。

```python

# Input train split is 'input_dir/train/*', eval split is 'input_dir/eval/*'.
# Output splits are generated one-to-one mapping from input splits.
input = proto.Input(splits=[
                example_gen_pb2.Input.Split(name='train', pattern='train/*'),
                example_gen_pb2.Input.Split(name='eval', pattern='eval/*')
            ])
example_gen = CsvExampleGen(input_base=input_dir, input_config=input)
```

ファイルベースの ExampleGen（CsvExampleGen、ImportExampleGen など）の場合、`pattern` は、入力ベースパスが指定するルートディレクトリで入力ファイルにマッピングする Glob 相対ファイルパターンです。クエリベースの ExampleGen の場合（BigQueryExampleGen、PrestoExampleGen）、`pattern` は SQL クエリです。

デフォルトでは、入力ベースディレクトリ全体が 1 つの入力 Split として扱われるため、train および eval 出力 Split は 2 対 1 の比率で生成されます。

ExampleGen の入力と出力の Split 構成については、[proto/example_gen.proto](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto) をご覧ください。また、下流でのカスタム Split の使用方法については、[下流のコンポーネントガイド](#examplegen_downstream_components)をご覧ください。

#### メソッドの分割

レコード全体の代わりに `hash_buckets` による分割方法を使用する場合、Example をパーティション化する特徴量を使用することができます。特徴量が存在する場合、ExampleGen はその特徴量のフィンガープリントをパーティションキーとして使用します。

この特徴量は、Example の特定のプロパティに関して安定した分割を維持するために使用できます。たとえば、パーティション特徴量名として「user_id」が選択されている場合、ユーザーは必ず同じ Split に含められます。

「特徴量」の意味と「特徴量」を特定の名前に一致させる方法の解釈は、ExampleGen の実装と Example の型によって異なります。

既製の ExampleGen 実装の場合:

- tf.Example を生成する場合の「特徴量」は tf.Example.features.feature 内のエントリを指します。
- tf.Example を生成する場合の「特徴量」は tf.SequenceExample.context.feature 内のエントリを指します。
- サポートされている特徴量は int64 型とバイト型のみです。

次の場合、ExampleGen はランタイムエラーをスローします。

- 指定された特徴量名が Example に存在しない。
- 特徴量が空である（`tf.train.Feature()`）。
- 浮動小数点型の特徴量など、サポートされていない型の特徴量である。

Exmple の特徴量に基づいて train/eval Split を出力するには、ExampleGen コンポーネントの `output_config` を設定します。以下に例を示します。

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

上記の例で `partition_feature_name` が設定されているところに注目してください。

### Span

注意: この機能は、TFX 0.15 以降でのみ利用できます。

Span は[入力 glob パターン](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto) の「{SPAN}」仕様を使用して取得できます。

- この仕様は数字に一致し、関連する SPAN 番号にデータマッピングします。たとえば、「data_{SPAN}-*.tfrecord」は「data_12-a.tfrecord」や「date_12-b.tfrecord」といったファイルを収集します。
- オプションとして、マッピングされたときの整数の幅をこの仕様に指定することができます。たとえば、「data_{SPAN:2}.file」は「data_02.file」や「data_27.file」（順に、Span-2 と Span-27 の入力）といったファイルにマッピングしますが、「data_1.file」や「data_123.file」にはマッピングしません。
- SPAN 仕様が欠落している場合は必ず Span '0' とみなされます。
- SPAN が指定されている場合、パイプラインは最新の Span を処理し、メタデータに Span 番号を保存します。

例として、次のような入力データがあるとします。

- '/tmp/span-1/train/data'
- '/tmp/span-1/eval/data'
- '/tmp/span-2/train/data'
- '/tmp/span-2/eval/data'

また、以下のような入力構成があるとします。

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

パイプラインをトリガーすると、次のように処理されます。

- '/tmp/span-2/train/data' を train Split として処理
- '/tmp/span-2/eval/data' を eval Split として処理

Span 番号は '2' です。後で '/tmp/span-3/...' が使用できるようになったら、もう一度パイプラインをトリガーするだけで、Span '3' が処理されるようになります。以下に、Span 仕様を使用したコード例を示します。

```python
input = proto.Input(splits=[
                proto.Input.Split(name='train',
                                            pattern='span-{SPAN}/train/*'),
                proto.Input.Split(name='eval',
                                            pattern='span-{SPAN}/eval/*')
            ])
example_gen = CsvExampleGen(input_base='/tmp', input_config=input)
```

特定の Span を取得するには、RangeConfig を使用することができます。これについては以下で詳しく説明します。

### Date

注意: この機能は、TFX 0.24.0 以降でのみ利用できます。

ファイルシステム上のデータソースが日付別に編成されている場合、TFX は日付を直接 Span 番号にマッピングすることができます。日付から Span へのマッピングには、{YYYY}、{MM}、および {DD} の 3 つの仕様が使われます。

- この 3 つの仕様は、いずれかが使用される場合に[入力 glob パターン](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto)にすべてを含める必要があります。
- {SPAN} 仕様または Date 仕様セットのいずれかを排他的に指定することができます。
- YYYY の年、MM の月、DD の日を使って暦の日付が計算され、その後に Span 番号が Unix エポック（1970-01-01）からの日数として計算されます。たとえば、「log-{YYYY}{MM}{DD}.data」はファイル「log-19700101.data」に一致して Span-0 の入力として消費され、「log-20170101.data」は Span-17167 の入力として消費されます。
- この Date 仕様セットが指定されている場合、パイプラインは最新の Date を処理し、メタデータに対応する Span 番号を保存します。

例として、次のような暦の日付で編成された入力データがあるとしましょう。

- '/tmp/1970-01-02/train/data'
- '/tmp/1970-01-02/eval/data'
- '/tmp/1970-01-03/train/data'
- '/tmp/1970-01-03/eval/data'

また、以下のような入力構成があるとします。

```python
from tfx.examples.custom_components.presto_example_gen.proto import presto_config_pb2
from tfx.examples.custom_components.presto_example_gen.presto_component.component import PrestoExampleGen

presto_config = presto_config_pb2.PrestoConnConfig(host='localhost', port=8080)
example_gen = PrestoExampleGen(presto_config, query='SELECT * FROM chicago_taxi_trips')
```

パイプラインをトリガーすると、次のように処理されます。

- '/tmp/1970-01-03/train/data' を train Split として処理
- '/tmp/1970-01-03/eval/data' を eval split として処理

Span 番号は '2' です。後で '/tmp/1970-01-04/...' が使用できるようになったら、もう一度パイプラインをトリガーするだけで、Span '3' が処理されるようになります。以下に、Date 仕様を使用したコード例を示します。

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

注意: この機能は、TFX 0.24.0 以降でのみ利用できます。

Version は[入力 glob パターン](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto)の「{VERSION}」を使用して取得できます。

- この仕様は数字に一致して SPAN にある関連する VERSION 番号にデータをマッピングします。Version 仕様は Span または Date 仕様と併せて使用できることに注意してください。
- この仕様はオプションとして SPAN 仕様と同じように幅を指定することもできます（「span-{SPAN}/version-{VERSION:4}/data-*」など）。
- VERSION 仕様が欠落している場合、Version は None に設定されます。
- SPAN と VERSION の両方が指定されている場合、パイプラインは最新の Span の最新の Version を処理し、メタデータにバージョン番号を保存します。
- VERSION が指定されていても SPAN（または DATE）が指定されていない場合、エラーがスローされます。

例として、次のような入力データがあるとします。

- '/tmp/span-1/ver-1/train/data'
- '/tmp/span-1/ver-1/eval/data'
- '/tmp/span-2/ver-1/train/data'
- '/tmp/span-2/ver-1/eval/data'
- '/tmp/span-2/ver-2/train/data'
- '/tmp/span-2/ver-2/eval/data'

また、以下のような入力構成があるとします。

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

パイプラインをトリガーすると、次のように処理されます。

- '/tmp/span-2/ver-2/train/data' を train Split として処理
- '/tmp/span-2/ver-2/eval/data' を eval Split として処理

Span 番号は '2' で Version 番号は '2' です。後で '/tmp/span-2/ver-3/...' が使用できるようになったら、もう一度パイプラインをトリガーするだけで、Span '2' と Version '3' が処理されるようになります。以下に、Version 仕様を使用したコード例を示します。

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

注意: この機能は、TFX 0.24.0 以降でのみ利用できます。

TFX は、さまざまな TFX エントリの範囲を指定するために使用する Range Config という抽象的な構成を使って、ファイルベースの ExampleGen の特定の Span を取得して処理することができます。特定の Span を取得するには、ファイルベースの ExampleGen コンポーネントの `range_config` を設定します。例として、次のような入力データがあるとしましょう。

- '/tmp/span-01/train/data'
- '/tmp/span-01/eval/data'
- '/tmp/span-02/train/data'
- '/tmp/span-02/eval/data'

具体的に Span '1' のデータを取得して処理するには、入力構成のほかに範囲構成も指定します。ExampleGen は単一 Span の静的範囲（特定の個別の Span の処理を指定）のみをサポートしていることに注意してください。したがって、StaticRange の場合、start_span_number は end_span_number と同一である必要があります。指定された Span とゼロパディングの Span 幅情報（指定されている場合）を使用すると、ExampleGen は指定された Spilt パターンの SPAN 仕様を目的の Span 番号に置き換えます。以下に使用方法を示します。

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

SPAN 仕様の代わりに DATE 仕様を使用すると、特定の日付を処理するために Range config を使用することができます。例として、暦の日付で編成された次のような入力データがあるとしましょう。

- '/tmp/1970-01-02/train/data'
- '/tmp/1970-01-02/eval/data'
- '/tmp/1970-01-03/train/data'
- '/tmp/1970-01-03/eval/data'

具体的に 1970 年 1 月 2 日のデータを取得して処理するには、次のように行います。

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

## カスタム ExampleGen

現在提供されている ExampleGen コンポーネントでニーズを賄えない場合は、カスタム ExampleGen を作成することができます。カスタム ExampleGen を作成すると、さまざまなデータソースやデータ形式でデータを読み取れるようになります。

### ファイルベースの ExampleGen カスタマイズ（実験的）

まず、カスタム Beam PTransform を使って BaseExampleGenExecutor を拡張します。train/eval 入力 Split を TF Example に変換するものです。たとえば、[CsvExampleGen Executor](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/csv_example_gen/executor.py) は入力 CSV Split を TF Example に変換します。

次に、[CsvExampleGen コンポーネント](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/csv_example_gen/component.py)と同じ方法で、上記の Executor を使用してコンポーネントを作成します。または、以下のようにカスタム Executor を標準の ExampleGen コンポーネントに渡すこともできます。

```python
from tfx.components.base import executor_spec
from tfx.components.example_gen.csv_example_gen import executor

example_gen = FileBasedExampleGen(
    input_base=os.path.join(base_dir, 'data/simple'),
    custom_executor_spec=executor_spec.ExecutorClassSpec(executor.Executor))
```

これで、この[メソッド](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/custom_executors/)を使用して、Avro ファイルと Parquet ファイルも読み取れるようになりました。

### その他のデータ形式

Apache Beam は、[その他にも多数のデータ形式](https://beam.apache.org/documentation/io/built-in/)の読み取りをサポートしています。Beam I/O Transforms を介する方法です。Beam I/O Transforms を利用し、[Avro の例](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/custom_executors/avro_executor.py#L56)に似たパターンを使用して、カスタム ExampleGen コンポーネントを作成することができます。

```python
  return (pipeline
          | 'ReadFromAvro' >> beam.io.ReadFromAvro(avro_pattern)
          | 'ToTFExample' >> beam.Map(utils.dict_to_example))
```

この記事を執筆している時点では、Beam Python SDK では次の形式とデータソースがサポートされています。

- Amazon S3
- Apache Avro
- Apache Hadoop
- Apache Kafka
- Apache Parquet
- Google Cloud BigQuery
- Google Cloud BigTable
- Google Cloud Datastore
- Google Cloud Pub/Sub
- Google Cloud Storage（GCS）
- MongoDB

最新のリストについては、[Beam ドキュメント](https://beam.apache.org/documentation/io/built-in/)をご覧ください。

### クエリベースの ExampleGen カスタマイズ（実験的）

まず、カスタム Beam PTransform を使って BaseExampleGenExecutor を拡張します。外部のデータソースからデータを読み取るものです。次に、QueryBasedExampleGen を拡張して単純なコンポーネントを作成します。

これには追加の接続構成が必要となる場合とならない場合があります。たとえば、[BigQuery Executor](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/big_query_example_gen/executor.py) はデフォルトの beam.io コネクタを使った読み取るため、接続構成情報が抽出されます。[Presto Executor](https://github.com/tensorflow/tfx/blob/master/tfx/examples/custom_components/presto_example_gen/presto_component/executor.py) の場合は、カスタム Beam PTransform と[カスタム接続構成の protobuf](https://github.com/tensorflow/tfx/blob/master/tfx/examples/custom_components/presto_example_gen/proto/presto_config.proto)が入力として必要です。

カスタム ExampleGen コンポーネントで接続構成が必要な場合は、新しい protobuf を作成して custom_config を介して渡します。現時点ではオプションの実行パラメーターです。以下に、構成済みのコンポーネントの使用例を示します。

```python
from tfx.examples.custom_components.presto_example_gen.proto import presto_config_pb2
from tfx.examples.custom_components.presto_example_gen.presto_component.component import PrestoExampleGen

presto_config = presto_config_pb2.PrestoConnConfig(host='localhost', port=8080)
example_gen = PrestoExampleGen(presto_config, query='SELECT * FROM chicago_taxi_trips')
```

## ExampleGen の下流コンポーネント

カスタム Split 構成は下流のコンポーネントでサポートされています。

### StatisticsGen

デフォルトでは、すべての Split に対して統計生成を実行します。

任意の Split を除外するには、StatisticsGen コンポーネントの `exclude_splits` を設定します。以下に例を示します。

```python
# Exclude the 'eval' split.
statistics_gen = StatisticsGen(
             examples=example_gen.outputs['examples'],
             exclude_splits=['eval'])
```

### SchemaGen

デフォルトでは、すべての Split に基づいてスキーマを生成します。

任意の Split を除外するには、SchemaGen コンポーネントの `exclude_splits` を設定します。以下に例を示します。

```python
# Exclude the 'eval' split.
schema_gen = SchemaGen(
             statistics=statistics_gen.outputs['statistics'],
             exclude_splits=['eval'])
```

### ExampleValidator

デフォルトでは、入力 Example ですべての Split の統計をスキーマに対して検証します。

任意の Split を除外するには、ExampleValidator コンポーネントの `exclude_splits` を設定します。以下に例を示します。

```python
# Exclude the 'eval' split.
example_validator = ExampleValidator(
             statistics=statistics_gen.outputs['statistics'],
             schema=schema_gen.outputs['schema'],
             exclude_splits=['eval'])
```

### 変換

デフォルトでは、'train' Split を分析してメタデータを生成し、すべての Split を変換します。

分析用 Split と変換用 Split を指定するには、Transform コンポーネントの `splits_config` を設定します。以下に例を示します。

```python
# Analyze the 'train' split and transform all splits.
transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=_taxi_module_file,
      splits_config=proto.SplitsConfig(analyze=['train'],
                                               transform=['train', 'eval']))
```

### Trainer および Tuner

デフォルトでは、'train' Split でトレーニングし、'eval' Split で評価します。

トレーニング用 Split と評価用 Split を指定するには、Trainer コンポーネントの `train_args` と `eval_args` を設定します。以下に例を示します。

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

デフォルトでは、'eval' Split で計算されたメトリクスを提供します。

To compute evaluation statistics on custom splits, set the `example_splits` for Evaluator component. For example:

```python
# Compute metrics on the 'eval1' split and the 'eval2' split.
evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      example_splits=['eval1', 'eval2'])
```

さらに詳細については、[CsvExampleGen API リファレンス](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/CsvExampleGen)、[FileBasedExampleGen API リファレンス](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/FileBasedExampleGen)、および [ImportExampleGen API リファレンス](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/ImportExampleGen)をご覧ください。
