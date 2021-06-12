# Pusher TFX パイプラインコンポーネント

Pusher コンポーネントは、検証済みのモデルをモデルのトレーニング中または再トレーニング中に[デプロイターゲット](index.md#deployment_targets)にプッシュするために使用するコンポーネントです。デプロイする前に、Pusher はほかの検証コンポーネントの 1 つ以上の blessing に依存してモデルをプッシュするかどうかを判断します。

- [Evaluator](evaluator) は、新しいトレーニングモデルが本番にプッシュするのに「十分に良い」場合にモデルを blessed に判定します。
- （オプションですが推奨されます）[InfraValidator](infra_validator) は、 モデルが機械的に本番環境にサービング可能である場合にモデルを blessed に判定します。

Pusher コンポーネントは、トレーニング済みの [SavedModel](/guide/saved_model) 形式のモデルを消費し、バージョン管理メタデータと共に同じ SavedModel を生成します。

## Pusher コンポーネントを使用する

Pusher パイプラインコンポーネントは通常、非常にデプロイしやすく、ほとんどカスタマイズする必要がありません。すべての作業は Pusher TFX コンポーネントが実行するためです。一般的なコードは次のようになります。

```python
pusher = Pusher(
  model=trainer.outputs['model'],
  model_blessing=evaluator.outputs['blessing'],
  infra_blessing=infra_validator.outputs['blessing'],
  push_destination=pusher_pb2.PushDestination(
    filesystem=pusher_pb2.PushDestination.Filesystem(
        base_directory=serving_model_dir)
  )
)
```

詳細については、[Pusher API リファレンス](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Pusher)をご覧ください。
