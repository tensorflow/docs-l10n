# ExampleValidator TFX パイプラインコンポーネント

ExampleValidator パイプラインコンポーネントはトレーニングデータとサービングデータの異常を特定します。データ内のさまざまな種類の異常を検出することが可能です。以下に、検出例を示します。

1. データの統計量を、ユーザーの期待を記述したスキーマと比較することで妥当性チェックを実行する。
2. トレーニングデータとサービングデータを比較して、トレーニング/サービングスキューを検出する。
3. データの系列を確認することで、データドリフトを検出する。

ExampleValidator パイプラインコンポーネントは、StatisticsGen パイプラインコンポーネントが計算した統計データをスキーマと比較し、Example データ内の異常を特定します。推論されたスキーマは入力データが満たすと期待されているプロパティを記述しますが、開発者が変更することもできます。

- 入力: SchemaGen コンポーネントのスキーマと StatisticsGen コンポーネントの統計
- 出力: 検証結果

## ExampleValidator と TensorFlow Data Validation

ExampleValidator は入力データの検証に [TensorFlow Data Validation](tfdv.md) を多大に使用しています。

## ExampleValidator コンポーネントを使用する

ExampleValidator パイプラインコンポーネントは通常、非常にデプロイしやすく、ほとんどカスタマイズする必要がありません。一般的なコードは次のように記述されます。

```python
validate_stats = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema']
      )
```

詳細については、[ExampleValidator API リファレンス](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/ExampleValidator)をご覧ください。
