# InfraValidator TFX パイプラインコンポーネント

InfraValidator は、モデルを本番にプッシュする前の早期警告レイヤーとして使用される TFX コンポーネントです。"Infra" Validator の名前は、実際のモデルをサービングする「インフラストラクチャ」でモデルを検証することに由来しています。[Evaluator](evaluator.md) がモデルのパフォーマンスを保証するものであるならば、InfraValidator はモデルが機械的に優良であることを保証し、不良モデルがプッシュされないようにします。

## 仕組み

InfraValidator はモデルを取り、サンドボックス化されたモデルサーバーをそのモデルで起動して、正常に読み込めるのか、またオプションとしてクエリできるのかどうかを確認します。インフラ検証の結果は `blessing` 出力に生成されます。これは [Evaluator](evaluator.md) と同じ仕組みです。

InfraValidator はモデルサーバーのバイナリー（[TensorFlow Serving](serving.md) など）とデプロイするモデルの互換性に焦点を当てています。「インフラ」バリデーターという名前ではありますが、環境を正しく構成するのは**ユーザーの責任**であり、インフラバリデーターはユーザーが構成した環境にあるモデルサーバーと対話して、正しく動作するかどうかを確認するだけです。この環境を正しく構成しておけば、インフラ検証の合格または不合格に基づいて、モデルを本番サービング環境にサービングできるかどうかを知ることができます。これは次のようなことを示しますが、ここに記載されていることがすべてではありません。

1. InfraValidator は本番で使用されるモデルサーバーバイナリーと同じバイナリーを使用している。これは、インフラ検証の環境が収束するための最低レベルです。
2. InfraValidator は、本番で使用されるリソースと同じリソース（CPU の割り当て量と種類、メモリ、アクセラレータなど）を使用している。
3. InfraValidator は本番で使用されるモデルサーバー構成と同じ構成を使用している

状況によって、InfraValidator がどの程度本番環境と同等であるかをユーザーが選択できます。厳密には、モデルはローカルの Docker 環境でインフラ検証した後で、まったく異なる環境（Kubernetes クラスタなど）に問題なく配布することができますが、InfraValidator はこの違いに関してはチェックしていないということになります。

### 操作モード

構成に応じて、次のいずれかのモードでインフラ検証を行えます。

- `LOAD_ONLY` モード: モデルがサービングインフラストラクチャに正常に読み込まれたかどうかをチェックします。**または**
- `LOAD_AND_QUERY` モード: `LOAD_ONLY` モードのチェックに加え、サンプルリクエストを送信して、モデルが推論を提供できるかどうかをチェックします。InfraValidator は予測が正しいかどうかはチェックしません。リクエストが正常に送信されたかどうかのみをチェックします。

## 使用方法

通常、InfraValidator は Evaluator コンポーネントの次の定義され、その出力は Pusher にフィードされます。InfraValidator が失敗した場合、そのモデルはプッシュされません。

```python
evaluator = Evaluator(
    model=trainer.outputs['model'],
    examples=example_gen.outputs['examples'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=tfx.proto.EvalConfig(...)
)

infra_validator = InfraValidator(
    model=trainer.outputs['model'],
    serving_spec=tfx.proto.ServingSpec(...)
)

pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    infra_blessing=infra_validator.outputs['blessing'],
    push_destination=tfx.proto.PushDestination(...)
)
```

### InfraValidator コンポーネントを構成する

InfraValidator の構成には、3 種類の protos を使用できます。

#### `ServingSpec`

`ServingSpec` は InfraValidator の最も重要な構成で、次のことを定義します。

- 実行するモデルサーバーの<u>種類</u>
- 実行する<u>場所</u>

次のモデルサーバーの種類（サービングバイナリー）がサポートされています。

- [TensorFlow Serving](serving.md)

注意: InfraValidator では、モデルの互換性に影響を及ぼすことなくモデルサーバーのバージョンをアップグレードするために、同じサーバータイプの複数のバージョンを指定することができます。たとえば、`tensorflow/serving` 画像を `2.1.0` と `latest` バージョンの両方でテストすることができるため、モデルが最新の `tensorflow/serving` バージョンとも互換することを保証することができます。

現在、次のサービングプラットフォームがサポートされています。

- ローカルの Docker（Docker が事前にインストールされている必要があります）
- Kubernetes（KubeflowDagRunner のみ制限サポート）

サービングバイナリーとサービングプラットフォームの選択は、[`oneof`](https://developers.google.com/protocol-buffers/docs/proto3#oneof) ブロックという  `ServingSpec` で指定します。たとえば、Kubernetes クラスタで実行している TensorFlow Serving バイナリーを使用するには、`tensorflow_serving` と `kubernetes` フィールドが設定されている必要があります。

```python
infra_validator=InfraValidator(
    model=trainer.outputs['model'],
    serving_spec=tfx.proto.ServingSpec(
        tensorflow_serving=tfx.proto.TensorFlowServing(
            tags=['latest']
        ),
        kubernetes=tfx.proto.KubernetesConfig()
    )
)
```

さらに `ServingSpec` を構成するには、[protobuf の定義](https://github.com/tensorflow/tfx/blob/master/tfx/proto/infra_validator.proto)をご覧ください。

#### `ValidationSpec`

インフラ検証の基準またはワークフローを調整するためのオプションの構成です。

```python
infra_validator=InfraValidator(
    model=trainer.outputs['model'],
    serving_spec=tfx.proto.ServingSpec(...),
    validation_spec=tfx.proto.ValidationSpec(
        # How much time to wait for model to load before automatically making
        # validation fail.
        max_loading_time_seconds=60,
        # How many times to retry if infra validation fails.
        num_tries=3
    )
)
```

すべての ValidationSpec フィールドには適切なデフォルト値があります。詳細は、[protobuf 定義](https://github.com/tensorflow/tfx/blob/master/tfx/proto/infra_validator.proto)をご覧ください。

#### `RequestSpec`

インフラ検証を `LOAD_AND_QUERY` モードで実行している場合のサンプルリクエストの構築方法を指定するためのオプションの構成です。`LOAD_AND_QUERY` モードを使用するには、`request_spec` 実行プロパティと `examples` 入力チャネルの両方がコンポーネントの定義に指定されている必要があります。

```python
infra_validator = InfraValidator(
    model=trainer.outputs['model'],
    # This is the source for the data that will be used to build a request.
    examples=example_gen.outputs['examples'],
    serving_spec=tfx.proto.ServingSpec(
        # Depending on what kind of model server you're using, RequestSpec
        # should specify the compatible one.
        tensorflow_serving=tfx.proto.TensorFlowServing(tags=['latest']),
        local_docker=tfx.proto.LocalDockerConfig(),
    ),
    request_spec=tfx.proto.RequestSpec(
        # InfraValidator will look at how "classification" signature is defined
        # in the model, and automatically convert some samples from `examples`
        # artifact to prediction RPC requests.
        tensorflow_serving=tfx.proto.TensorFlowServingRequestSpec(
            signature_names=['classification']
        ),
        num_examples=10  # How many requests to make.
    )
)
```

### ウォームアップで SavedModel を生成する

（バージョン 0.30.0 以降）

InfraValidator は実際のリクエストでモデルを検証するため、これらの検証リクエストを SavedModel の[ウォームアップリクエスト](https://www.tensorflow.org/tfx/serving/saved_model_warmup)として簡単に再利用することができます。InfraValidator にはウォームアップで SavedModel をエクスポートするオプション（`RequestSpec.make_warmup`）が用意されています。

```python
infra_validator = InfraValidator(
    ...,
    request_spec=tfx.proto.RequestSpec(..., make_warmup=True)
)
```

そして、出力 `InfraBlessing` アーティファクトにはウォームアップによる SavedModel が含まれているため、[Pusher](pusher.md) で、`Model` アーティファクトと同様にプッシュすることもできます。

## 制限事項

現在の InfraValidator は未完全であるため、次のような制限があります。

- TensorFlow [SavedModel](/guide/saved_model) モデル形式のみを検証できます。

- Kubernetes で TFX を実行している場合、パイプラインは Kuberflow パイプライン内で `KubeflowDagRunner` によって実行される必要があります。モデルサーバーは Kuberflow が使用しているのと同じ Kubernetes Kurasuta と名前空間で起動します。

- InfraValidator は [TensorFlow Serving](serving.md) にデプロイすることを主な焦点としているため、[TensorFlow Lite](/lite) と  [TensorFlow.js](/js)、またはその他の推論フレームワークへのデプロイでは役に立つとはいえ、精度に劣ります。

- `LOAD_AND_QUERY` モードのサポートは [Predict](/versions/r1.15/api_docs/python/tf/saved_model/predict_signature_def) メソッドシグネチャ（TensorFlow 2 でのみエクスポート可能なメソッド）において制限されています。InfraValidator ではシリアル化された [`tf.Example`](/tutorials/load_data/tfrecord#tfexample) を唯一の入力として消費するために Predict シグネチャが必要となります。

    ```python
    @tf.function
    def parse_and_run(serialized_example):
      features = tf.io.parse_example(serialized_example, FEATURES)
      return model(features)

    model.save('path/to/save', signatures={
      # This exports "Predict" method signature under name "serving_default".
      'serving_default': parse_and_run.get_concrete_function(
          tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'))
    })
    ```

    - このシグネチャが TFX のほかのコンポーネントとどのように対話するかについては、[Penguin の例](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local_infraval.py)のサンプルコードをご覧ください。
