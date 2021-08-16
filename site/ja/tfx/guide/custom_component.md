# フルカスタムコンポーネントの構築

このガイドでは、TFX API を使用してフルカスタムコンポーネントを構築する方法について説明します。フルカスタムコンポーネントを使用すると、コンポーネント仕様、Executor、およびコンポーネントインターフェイスクラスを定義し、コンポーネントを構築できます。これより、ニーズに合わせて標準コンポーネントを再利用および拡張できます。

TFX パイプラインが初めての方は、[TFX パイプラインの中心的概念の学習](understanding_tfx_pipelines)をご確認ください。

## カスタム Executor またはカスタムコンポーネント

コンポーネントの入力、出力、および実行プロパティが既存のコンポーネントと同じで、カスタム処理ロジックのみが必要な場合は、カスタム Executor で十分です。入力、出力、または実行プロパティのいずれかが既存の TFX コンポーネントと異なる場合は、フルカスタムコンポーネントが必要です。

## カスタムコンポーネントの作成

フルカスタムのコンポーネントを作成するには、以下が必要です。

- 新しいコンポーネントの定義済み入力および出力アーティファクト仕様。特に、入力アーティファクトの型は、アーティファクトを生成するコンポーネントの出力アーティファクト型と一致している必要があり、出力アーティファクトの型は、アーティファクトを使用するコンポーネントの入力アーティファクト型と一致している必要があります。
- 新しいコンポーネントに必要なアーティファクト以外の実行パラメータ。

### ComponentSpec

`ComponentSpec`クラスは、コンポーネントへの入力アーティファクトと出力アーティファクト、およびコンポーネントの実行に使用されるパラメータを定義することにより、コンポーネントコントラクトを定義します。これには 3 つの部分があります。

- *入力*: コンポーネント Executor にある入力アーティファクトの型付きパラメータのディクショナリ。通常、入力アーティファクトはアップストリームコンポーネントからの出力であるため、同じ型を共有します。
- *出力*: コンポーネントが生成する出力アーティファクトの型付きパラメータのディクショナリ。
- *パラメータ*: コンポーネント Executor に渡される追加の[ExecutionParameter](https://github.com/tensorflow/tfx/blob/54aa6fbec6bffafa8352fe51b11251b1e44a2bf1/tfx/types/component_spec.py#L274) アイテムのディクショナリ。これらは、パイプライン DSL で柔軟に定義し、実行に渡すアーティファクト以外のパラメータです。

以下は、ComponentSpec の例です。

```python
class HelloComponentSpec(types.ComponentSpec):
  """ComponentSpec for Custom TFX Hello World Component."""

  PARAMETERS = {
      # These are parameters that will be passed in the call to
      # create an instance of this component.
      'name': ExecutionParameter(type=Text),
  }
  INPUTS = {
      # This will be a dictionary with input artifacts, including URIs
      'input_data': ChannelParameter(type=standard_artifacts.Examples),
  }
  OUTPUTS = {
      # This will be a dictionary which this component will populate
      'output_data': ChannelParameter(type=standard_artifacts.Examples),
  }
```

### Executor

次に、新しいコンポーネントの Executor コードを記述します。基本的に、`base_executor.BaseExecutor`の新しいサブクラスを作成し、その`Do`関数をオーバーライドする必要があります。`Do`関数では、`INPUTS`、`OUTPUTS`および`PARAMETERS`にマップして渡される`input_dict`、`output_dict`および`exec_properties`引数は、ComponentSpec でそれぞれ定義されます。`exec_properties`の場合、値はディクショナリルックアップを介して直接フェッチできます。`input_dict`および`output_dict`のアーティファクトの場合、アーティファクトインスタンスまたはアーティファクト URI をフェッチするための [artifact_utils](https://github.com/tensorflow/tfx/blob/41823f91dbdcb93195225a538968a80ba4bb1f55/tfx/types/artifact_utils.py) クラスで使用できる便利な関数があります。

```python
class Executor(base_executor.BaseExecutor):
  """Executor for HelloComponent."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    ...

    split_to_instance = {}
    for artifact in input_dict['input_data']:
      for split in json.loads(artifact.split_names):
        uri = artifact_utils.get_split_uri([artifact], split)
        split_to_instance[split] = uri

    for split, instance in split_to_instance.items():
      input_dir = instance
      output_dir = artifact_utils.get_split_uri(
          output_dict['output_data'], split)
      for filename in tf.io.gfile.listdir(input_dir):
        input_uri = os.path.join(input_dir, filename)
        output_uri = os.path.join(output_dir, filename)
        io_utils.copy_file(src=input_uri, dst=output_uri, overwrite=True)
```

#### カスタム Executor の単体テスト

カスタム Executor の単体テストは、[こちら](https://github.com/tensorflow/tfx/blob/r0.15/tfx/components/transform/executor_test.py)のように作成できます。

### コンポーネントインターフェイス

最も複雑な部分は以上です。次の手順では、これらの部分をコンポーネントインターフェイスにアセンブルし、コンポーネントをパイプラインで使用できるようにします。これには、いくつかのステップがあります。

- コンポーネントインターフェイスを`base_component.BaseComponent`のサブクラスにします
- 以前に定義された`ComponentSpec`クラスをもつクラス変数`SPEC_CLASS`を割り当てます
- 以前に定義された Executor クラスをもつクラス変数`EXECUTOR_SPEC`を割り当てます
- 関数の引数を使用して`__init__()`コンストラクタ関数を定義し、ComponentSpec クラスのインスタンスを構築し、その値とオプションの名前を使用してスーパー関数を呼び出します。

コンポーネントのインスタンスが作成されると、`base_component.BaseComponent`クラスの型チェックロジックが呼び出され、渡された引数が`ComponentSpec`クラスで定義された型情報と互換性があることを確認します。

```python
from tfx.types import standard_artifacts
from hello_component import executor

class HelloComponent(base_component.BaseComponent):
  """Custom TFX Hello World Component."""

  SPEC_CLASS = HelloComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               input_data: types.Channel = None,
               output_data: types.Channel = None,
               name: Optional[Text] = None):
    if not output_data:
      examples_artifact = standard_artifacts.Examples()
      examples_artifact.split_names = input_data.get()[0].split_names
      output_data = channel_utils.as_channel([examples_artifact])

    spec = HelloComponentSpec(input_data=input_data,
                              output_data=output_data, name=name)
    super(HelloComponent, self).__init__(spec=spec)
```

### TFX パイプラインにアセンブルする

最後のステップは、新しいカスタムコンポーネントを TFX パイプラインにプラグすることです。新しいコンポーネントのインスタンスの他に、以下を追加する必要があります。

- 新しいコンポーネントのアップストリームコンポーネントとダウンストリームコンポーネントを適切に接続します。これは、新しいコンポーネントでアップストリームコンポーネントの出力を参照し、ダウンストリームコンポーネントで新しいコンポーネントの出力を参照することによって行われます。
- パイプラインを構築するときに、新しいコンポーネントインスタンスをコンポーネントリストに追加します。

以下の例は、前述の変更を示しています。完全な例は、[TFX GitHub リポジトリ](https://github.com/tensorflow/tfx/tree/master/tfx/examples/custom_components/hello_world)をご覧ください。

```python
def _create_pipeline():
  ...
  example_gen = CsvExampleGen(input_base=examples)
  hello = component.HelloComponent(
      input_data=example_gen.outputs['examples'], name='HelloWorld')
  statistics_gen = StatisticsGen(examples=hello.outputs['output_data'])
  ...
  return pipeline.Pipeline(
      ...
      components=[example_gen, hello, statistics_gen, ...],
      ...
  )
```

## フルカスタムコンポーネントのデプロイ

パイプラインを適切に実行するには、コードの変更に加えて、新しく追加されたすべてのパーツ（`ComponentSpec`、`Executor`、コンポーネントインターフェイス）がパイプライン実行環境でアクセス可能である必要があります。
