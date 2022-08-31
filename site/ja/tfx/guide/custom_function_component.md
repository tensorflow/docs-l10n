# カスタム Python 関数コンポーネント

Python 関数ベースのコンポーネント定義を使用すると、コンポーネント仕様クラス、実行クラス、およびコンポーネントインターフェイスクラスを定義する手間が省けるため、TFX カスタムコンポーネントを簡単に作成できます。このコンポーネント定義スタイルでは、型ヒントで注釈が付けられた関数を記述します。型ヒントは、コンポーネントの入力アーティファクト、出力アーティファクト、およびパラメータを記述します。

次の例のように、簡単にカスタムコンポーネントを作成できます。

```python
@component
def MyValidationComponent(
    model: InputArtifact[Model],
    blessing: OutputArtifact[Model],
    accuracy_threshold: Parameter[int] = 10,
    ) -> OutputDict(accuracy=float):
  '''My simple custom model validation component.'''

  accuracy = evaluate_model(model)
  if accuracy >= accuracy_threshold:
    write_output_blessing(blessing)

  return {
    'accuracy': accuracy
  }
```

内部的には、これは [`BaseComponent`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/components/base/base_component.py){: .external } のサブクラスであるカスタムコンポーネントとその Spec クラスおよび Executor クラスを定義します。

注意: 以下で説明する機能（関数に `@component(use_beam=True)` でアノテーションを付けた BaseBeamComponent ベースのコンポーネント）は実験的なものであり、下位互換性の保証はありません。

[`BaseBeamComponent`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/components/base/base_beam_component.py){: .external } のサブクラスを定義して、TFX パイプラインごとの共有構成でビームパイプラインを使用できるようにする、つまり、パイプラインをコンパイルするときに `beam_pipeline_args`（[Chicago Taxi Pipeline Example](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_simple.py#L192){: .external }）を使用する場合、次の例のように、デコレータで `use_beam=True` を設定し、関数にデフォルト値 `None` で別の `BeamComponentParameter` を追加できます。

```python
@component(use_beam=True)
def MyDataProcessor(
    examples: InputArtifact[Example],
    processed_examples: OutputArtifact[Example],
    beam_pipeline: BeamComponentParameter[beam.Pipeline] = None,
    ) -> None:
  '''My simple custom model validation component.'''

  with beam_pipeline as p:
    # data pipeline definition with beam_pipeline begins
    ...
    # data pipeline definition with beam_pipeline ends
```

TFX パイプラインが初めての方は、[TFX パイプラインの中心的概念の学習](understanding_tfx_pipelines)をご確認ください。

## 入力、出力、およびパラメータ

TFX では、入力と出力は、基になるデータの場所とそれに関連付けられたメタデータプロパティを記述するアーティファクトオブジェクトとして追跡されます。この情報は ML メタデータに保存されます。アーティファクトは、int、float、bytes、unicode 文字列などの複雑なデータ型または単純なデータ型を記述できます。

パラメータは、パイプラインの構築時に認識されているコンポーネントへの引数（int、float、bytes、または Unicode 文字列）です。パラメータは、引数と、トレーニングの反復回数、ドロップアウト率、その他の構成などのハイパーパラメータをコンポーネントに指定するのに役立ちます。パラメータは、ML メタデータで追跡される際に、コンポーネント実行のプロパティとして保存されます。

注：現在、出力された単純なデータ型の値は、実行時に不明であるため、パラメータとして使用できません。同様に、入力された単純なデータ型の値は、現在、パイプラインの構築時に既知の具象値を取ることはできません。TFX の今後のリリースでは、この制限が削除される可能性があります。

## 定義

カスタムコンポーネントを作成するには、カスタムロジックを実装する関数を記述し、`tfx.dsl.component.experimental.decorators`モジュールからの[`@component`デコレータ](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/decorators.py){: .external }でデコレートします。コンポーネントの入力スキーマと出力スキーマを定義するには、`tfx.dsl.component.experimental.annotations`モジュールの注釈を使用して、関数の引数と戻り値に注釈を付けます。

- それぞれの**入力アーティファクト**に、`InputArtifact[ArtifactType]` の型ヒント注釈を適用します。`ArtifactType` を、`tfx.types.Artifact` のサブクラスであるアーティファクトの型に置き換えます。これらの入力はオプションの引数にすることができます。

- それぞれの**出力アーティファクト**に、`OutputArtifact[ArtifactType]` の型ヒント注釈を適用します。`ArtifactType` を、`tfx.types.Artifact` のサブクラスであるアーティファクトの型に置き換えます。コンポーネントの出力アーティファクトは、関数の入力引数として渡す必要があります。これにより、コンポーネントは、システム管理の場所に出力を書き込み、適切なアーティファクトメタデータプロパティを設定できます。この引数はオプションにすることも、デフォルト値で定義することもできます。

- **パラメータ**ごとに、型ヒント注釈 `Parameter[T]` を使用します。`T`をパラメータの方に置き換えます。現在サポートされているプリミティブ Python 型は、`bool`、`int`、`float`、`str`、および `bytes` のみです。

- **ビームパイプライン**の場合は、型ヒント注釈 `BeamComponentParameter[beam.Pipeline]` を使用します。デフォルト値を `None` に設定します。値 `None` は、[`BaseBeamExecutor`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/components/base/base_beam_executor.py){: .external } の `_make_beam_pipeline()` によって作成されたインスタンス化されたビームパイプラインに置き換えられます。

- パイプラインの構築時に不明な**単純なデータ型の入力**（`int`、`float`、`str` または `bytes`）には、それぞれ型ヒント `T` を使用してください。TFX 0.22 リリースでは、この型の入力のパイプライン構築時に具象値を渡すことができないことに注意してください（前のセクションで説明したように、代わりに `Parameter` 注釈を使用します）。この引数はオプションにすることも、デフォルト値で定義することもできます。コンポーネントに単純なデータ型出力（`int`、`float`、`str`または`bytes`）がある場合、これらの出力は、`OutputDict` インスタンスを使用して返すことができます。`OutputDict` 型のヒントをコンポーネントの戻り値として適用します。

- それぞれの**出力**で、引数 `<output_name>=<T>` を `OutputDict` コンストラクタに追加します。`<output_name>` は出力名、`<T>` は出力の型です（`int`、`float`、`str` または `bytes`など）。

関数の本体では、入力アーティファクトと出力アーティファクトは`tfx.types.Artifact`オブジェクトとして渡されます。`.uri`を調べて、システム管理の場所を取得し、プロパティを読み取り/設定できます。入力パラメータと単純なデータ型の入力は、指定された型のオブジェクトとして渡されます。単純なデータ型の出力はディクショナリとして返される必要があります。鍵は適切な出力名であり、値は期待される戻り値です。

完成した関数コンポーネントは次のようになります。

```python
import tfx.v1 as tfx
from tfx.dsl.component.experimental.decorators import component

@component
def MyTrainerComponent(
    training_data: tfx.dsl.components.InputArtifact[tfx.types.standard_artifacts.Examples],
    model: tfx.dsl.components.OutputArtifact[tfx.types.standard_artifacts.Model],
    dropout_hyperparameter: float,
    num_iterations: tfx.dsl.components.Parameter[int] = 10
    ) -> tfx.v1.dsl.components.OutputDict(loss=float, accuracy=float):
  '''My simple trainer component.'''

  records = read_examples(training_data.uri)
  model_obj = train_model(records, num_iterations, dropout_hyperparameter)
  model_obj.write_to(model.uri)

  return {
    'loss': model_obj.loss,
    'accuracy': model_obj.accuracy
  }

# Example usage in a pipeline graph definition:
# ...
trainer = MyTrainerComponent(
    examples=example_gen.outputs['examples'],
    dropout_hyperparameter=other_component.outputs['dropout'],
    num_iterations=1000)
pusher = Pusher(model=trainer.outputs['model'])
# ...
```

上記の例では、`MyTrainerComponent`を Python 関数ベースのカスタムコンポーネントとして定義しています。このコンポーネントは、入力として`examples`アーティファクトを消費し、出力として`model`アーティファクトを生成します。コンポーネントは`artifact_instance.uri`を使用して、システム管理の場所でアーティファクトの読み取りまたは書き込みを行います。コンポーネントは`num_iterations`入力パラメータと`dropout_hyperparameter`の単純なデータ型値を取り、コンポーネントは単純なデータ型の出力値として`loss`と`accuracy`メトリックを出力します。出力された`model`アーティファクトは、`Pusher`コンポーネントにより使用されます。
