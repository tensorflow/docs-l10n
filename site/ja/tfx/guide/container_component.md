# コンテナベースのコンポーネントの構築

コンテナベースのコンポーネントは、任意の言語で記述されたコードをパイプラインに統合する柔軟性を提供します。コードは Docker コンテナで実行できる必要があります。

TFX パイプラインが初めての方は、読み進める前に、[TFX パイプラインの中心的概念を学習](understanding_tfx_pipelines)してください。

## コンテナベースのコンポーネントの作成

コンテナベースのコンポーネントは、コンテナ化されたコマンドラインプログラムによりサポートされています。コンテナイメージがすでにある場合は、`create_container_component` 関数{: .external }を使用して入力と出力を宣言し、TFX を使用してコンポーネントを作成します。

- **名前:** コンポーネントの名前
- **入力:** 入力名を型にマップする辞書。 出力：出力名を型パラメータにマップする辞書：パラメータ名を型にマップする辞書。
- **イメージ:** コンテナイメージ名、およびオプションのイメージタグ。
- **コマンド:** コンテナエントリポイントのコマンドライン。シェル内では実行されません。コマンドラインでは、コンパイル時に入力、出力、またはパラメータに置き換えられるプレースホルダーオブジェクトを使用できます。プレースホルダーオブジェクトは、[`tfx.dsl.component.experimental.placeholders`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/placeholders.py){: .external }からインポートできます。 Jinja テンプレートはサポートされていないのでご注意ください。

**戻り値:** パイプライン内でインスタンス化して使用できる base_component.BaseComponent を継承する Component クラス。

### プレースホルダー

入力または出力を持つコンポーネントの場合、`command`には、実行時に実際のデータに置き換えられるプレースホルダーが必要になることがよくあります。このために、いくつかのプレースホルダーが提供されています。

- `InputValuePlaceholder`: 入力アーティファクトの値のプレースホルダー。実行時に、このプレースホルダーはアーティファクトの値の文字列表現に置き換えられます。

- `InputUriPlaceholder`: 入力アーティファクト引数の URI のプレースホルダー。実行時に、このプレースホルダーは入力アーティファクトのデータの URI に置き換えられます。

- `OutputUriPlaceholder`: 出力アーティファクト引数の URI のプレースホルダー。実行時に、このプレースホルダーは、コンポーネントが出力アーティファクトのデータを格納する URI に置き換えられます。

[TFX コンポーネントのコマンドラインプレースホルダー](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/placeholders.py){: .external }の詳細をご覧ください。

### コンテナベースのコンポーネントの例

以下は、データをダウンロード、変換、およびアップロードする Python 以外のコンポーネントの例です。

```python
import tfx.v1 as tfx

grep_component = tfx.dsl.components.create_container_component(
    name='FilterWithGrep',
    inputs={
        'text': tfx.standard_artifacts.ExternalArtifact,
    },
    outputs={
        'filtered_text': tfx.standard_artifacts.ExternalArtifact,
    },
    parameters={
        'pattern': str,
    },
    # The component code uses gsutil to upload the data to Google Cloud Storage, so the
    # container image needs to have gsutil installed and configured.
    image='google/cloud-sdk:278.0.0',
    command=[
        'sh', '-exc',
        '''
          pattern="$1"
          text_uri="$3"/data  # Adding suffix, because currently the URI are "directories". This will be fixed soon.
          text_path=$(mktemp)
          filtered_text_uri="$5"/data  # Adding suffix, because currently the URI are "directories". This will be fixed soon.
          filtered_text_path=$(mktemp)

          # Getting data into the container
          gsutil cp "$text_uri" "$text_path"

          # Running the main code
          grep "$pattern" "$text_path" >"$filtered_text_path"

          # Getting data out of the container
          gsutil cp "$filtered_text_path" "$filtered_text_uri"
        ''',
        '--pattern', tfx.dsl.placeholders.InputValuePlaceholder('pattern'),
        '--text', tfx.dsl.placeholders.InputUriPlaceholder('text'),
        '--filtered-text', tfx.dsl.placeholders.OutputUriPlaceholder('filtered_text'),
    ],
)
```
