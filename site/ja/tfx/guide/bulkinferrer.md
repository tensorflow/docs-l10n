# BulkInferrer TFX パイプラインコンポーネント

BulkInferrer TFX コンポーネントは、ラベルの無いデータで一括推論を実行します。生成される InferenceResult([tensorflow_serving.apis.prediction_log_pb2.PredictionLog](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_log.proto)) には、元の特徴量と予測結果が含まれます。

BulkInferrer は次の項目を消費します。

- [SavedModel](https://www.tensorflow.org/guide/saved_model.md) 形式のトレーニング済みモデル
- 特徴量を含むラベル無しの tf.Examples
- （オプション）[Evaluator](https://www.tensorflow.org/tfx/guide/evaluator.md) コンポーネントの検証結果

BulkInferrer は次の項目を発します。

- [InferenceResult](https://github.com/tensorflow/tfx/blob/master/tfx/types/standard_artifacts.py)

## BulkInferrer コンポーネントを使用する

BulkInferrer TFX コンポーネントは、ラベル無しの tf.Examples に対して一括推論を実行するために使用されます。通常、検証済みのモデルで推論を実行するように [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator.md) の後にデプロイされるか、エクスポートされたモデルで直接推論を実行するように [Trainer](https://www.tensorflow.org/tfx/guide/trainer.md) コンポーネントの後にデプロイされます。

現在、インメモリモデル推論とリモート推論を実行します。リモート推論の場合は、モデルが Cloud AI Platform にホストされている必要があります。

次は、典型的なコードです。

```python
bulk_inferrer = BulkInferrer(
    examples=examples_gen.outputs['examples'],
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    data_spec=bulk_inferrer_pb2.DataSpec(),
    model_spec=bulk_inferrer_pb2.ModelSpec()
)
```

より詳細な情報は、[BulkInferrer API リファレンス](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/BulkInferrer)をご覧ください。
