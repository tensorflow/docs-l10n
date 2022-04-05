# SchemaGen TFX パイプラインコンポーネント

一部の TFX コンポーネントは、*スキーマ*と呼ばれる入力データの記述を使用します。スキーマは [schema.proto](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto) のインスタンスです。特徴量値のデータ型、特徴量がすべての Example に存在する必要があるかどうか、値の許容範囲、およびその他のプロパティを指定できます。SchemaGen パイプラインコンポーネントは、型、カテゴリ、および範囲をトレーニングデータから推論して自動的にスキーマを生成します。

- 入力: StatisticsGen コンポーネントの統計
- 出力: データスキーマ proto

以下にスキーマ proto の例を示します。

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

次の TFX ライブラリはスキーマを使用します。

- TensorFlow Data Validation
- TensorFlow Transform
- TensorFlow Model Analysis

典型的な TFX パイプラインの SchemaGen は他のパイプラインコンポーネントで消費されるスキーマを生成します。ただし、自動生成スキーマはベストエフォートであり、データの基本プロパティのみの推論を試みます。開発者は必要に応じてそれをレビューし、変更する必要があります。

変更されたスキーマは、ImportSchemaGen コンポーネントを使用してピアプラインに戻すことができます。最初のスキーマ生成に使用される SchemaGen コンポーネントを削除し、すべての下流コンポーネントに ImportSchemaGen の出力を使用させることが可能です。また、インポートされたスキーマを使用して [ExampleValidator](https://www.tensorflow.org/tfx/guide/exampleval) を追加し、継続的にトレーニングデータを調べることも推奨されます。

## SchemaGen と TensorFlow Data Validation

SchemaGen はスキーマの推論に [TensorFlow Data Validation](tfdv.md) を多大に使用しています。

## SchemaGen コンポーネントを使用する

### 最初のスキーマ生成

SchemaGen パイプラインコンポーネントは通常、非常にデプロイしやすく、ほとんどカスタマイズする必要はありません。一般的なコードは次のように記述されます。

```python
schema_gen = tfx.components.SchemaGen(
    statistics=stats_gen.outputs['statistics'])
```

詳細については、[SchemaGen API リファレンス](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/SchemaGen)をご覧ください。

### レビュー済みのスキーマインポート

ImportSchemaGen コンポーネントをパイプラインに追加し、レビュー済みのスキーマ定義をパイプラインに戻します。

```python
schema_gen = tfx.components.ImportSchemaGen(
    schema_file='/some/path/schema.pbtxt')
```

`schema_file` は、テキストprotobuf ファイルへのフルパスである必要があります。

詳細については、[ImportSchemaGen API リファレンス](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/ImportSchemaGen)をご覧ください。
