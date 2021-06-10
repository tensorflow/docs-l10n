# StatisticsGen TFX パイプラインコンポーネント

StatisticsGen TFX パイプラインコンポーネントは、training と serving データの両方で、ほかのパイプラインコンポーネントが使用できる特徴量の統計を生成します。StatisticsGen は Beam を使用して、大型のデータセットに合わせて拡張することができます。

- 入力: ExampleGen パイプラインコンポーネントが作成したデータセット
- 出力: データセットの統計

## StatisticsGen と TensorFlow Data Validation

StatisticsGen はデータセットから統計を生成するために [TensorFlow Data Validation](tfdv.md) を多大に使用しています。

## StatsGen コンポーネントを使用する

StatisticsGen パイプラインコンポーネントは通常、非常にデプロイしやすく、ほとんどカスタマイズする必要はありません。一般的なコードは次のように記述されます。

```python
compute_eval_stats = StatisticsGen(
      examples=example_gen.outputs['examples'],
      name='compute-eval-stats'
      )
```

## StatsGen コンポーネントをスキーマと使用する

パイプラインを初めて実行する場合、スキーマの推論に StatisticsGen が使用されます。しかし以降の実行では、データセットに関する追加情報が含まれる手動作成のスキーマが存在することがあります。このスキーマを StatisticsGen に提供すると、TFDV はデータセットの宣言済みのプロパティに基づいてさらに有用な統計を提供することができます。

この設定では、以下のように ImpoterNode によってインポート済みのキュレートされたスキーマを使って StatisticsGen を呼び出します。

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

### キュレートされたスキーマを作成する

TFX の `Schema` は TensorFlow メタデータの <a href="https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto" data-md-type="link">`Schema` proto</a> のインスタンスです。これは、[テキスト形式](https://googleapis.dev/python/protobuf/latest/google/protobuf/text_format.html)で新規に作成することができますが、`SchemaGen` が出発点として生成した推論済みのスキーマを使用する方が簡単です。`SchemaGen` コンポーネントが実行すると、次のパスのパイプラインルートの下にスキーマが配置されます。

```
<pipeline_root>/SchemaGen/schema/<artifact_id>/schema.pbtxt
```

上記の `<artifact_id>` は、MLMD におけるこのスキーマバージョンの一意の ID です。このスキーマ proto は、確実には推論できないデータセットに関する情報を通知するように変更することができます。この情報により、`StatisticsGen` の出力がより有用になり、[`ExampleValidator`](https://www.tensorflow.org/tfx/guide/exampleval) コンポーネントで実施される検証がより厳密になります。

詳細については、[StatisticsGen API リファレンス](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/StatisticsGen)をご覧ください。
