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
  push_destination=tfx.proto.PushDestination(
    filesystem=tfx.proto.PushDestination.Filesystem(
        base_directory=serving_model_dir)
  )
)
```

### InfraValidator から生成されたモデルをプッシュする

（バージョン 0.30.0 以降）

InfraValidator は `InfraBlessing` アーティファクトに[ウォームアップ付きのモデル](infra_validator#producing_a_savedmodel_with_warmup)を含めて生成することも可可能で、Pusher はそのアーティファクトを `Model` アーティファクトと同様にプッシュすることができます。

```python
infra_validator = InfraValidator(
    ...,
    # make_warmup=True will produce a model with warmup requests in its
    # 'blessing' output.
    request_spec=tfx.proto.RequestSpec(..., make_warmup=True)
)

pusher = Pusher(
    # Push model from 'infra_blessing' input.
    infra_blessing=infra_validator.outputs['blessing'],
    push_destination=tfx.proto.PushDestination(...)
)
```

詳細については、[Pusher API リファレンス](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Pusher)をご覧ください。
