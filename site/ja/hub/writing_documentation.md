<!--* freshness: { owner: 'wgierke' reviewed: '2022-07-27' review_interval: '6 months' } *-->

# ドキュメントの作成

tfhub.dev にモデルを貢献するには、Markdown 形式のドキュメントを提供する必要があります。tfhub.dev にモデルを追加するプロセスの完全な概要については、[モデルの貢献](contribute_a_model.md)ガイドをご覧ください。

## Markdown ドキュメントの種類

tfhub.dev では次の 3 種類の Markdown ドキュメントが使用されています。

- パブリッシャー Markdown - パブリッシャーに関する情報（[Markdown 構文をご覧ください](#publisher)）
- モデル Markdown - 特定のモデルとその使用方法に関する情報 ([Markdown 構文をご覧ください](#model))
- コレクション Markdown - パブリッシャーが定義したモデルコレクションに関する情報が含まれます（[Markdown 構文をご覧ください](#collection)）。

## コンテンツの編成

[TensorFlow Hub GitHub](https://github.com/tensorflow/tfhub.dev) リポジトリに貢献する際は、次のコンテンツで編成する必要があります。

- 各パブリッシャーディレクトリは  `assets/docs` ディレクトリに配置します。
- オプションの `models` ディレクトリと `collections` ディレクトリは、各パブリッシャディレクトリに含めます。
- 各モデルには、`assets/docs/<publisher_name>/models` の下にそれぞれのディレクトリを用意します。
- 各コレクションには、`assets/docs/<publisher_name>/collections` の下にそれぞれのディレクトリを用意します。

パブリッシャーの Markdown はバージョン管理されていませんが、モデルにはさまざまなバージョンを設けることができます。モデルの各バージョンには、そのバージョンにちなんだ個別の Markdown ファイル（1.md、2.md など）が必要です。コレクションはバージョン管理されていますが、1つのバージョンのみ（1.md）がサポートされています。

特定のモデルのすべてのバージョンを、そのモデルのディレクトリに配置してください。

Markdown コンテンツの編成を次の図に示しています。

```
assets/docs
├── <publisher_name_a>
│   ├── <publisher_name_a>.md  -> Documentation of the publisher.
│   └── models
│       └── <model_name>       -> Model name with slashes encoded as sub-path.
│           ├── 1.md           -> Documentation of the model version 1.
│           └── 2.md           -> Documentation of the model version 2.
├── <publisher_name_b>
│   ├── <publisher_name_b>.md  -> Documentation of the publisher.
│   ├── models
│   │   └── ...
│   └── collections
│       └── <collection_name>
│           └── 1.md           -> Documentation for the collection.
├── <publisher_name_c>
│   └── ...
└── ...
```

## パブリッシャー Markdown 形式 {:#publisher}

パブリッシャードキュメントは、モデルと同じ種類の Markdown ファイルで宣言されますが、構文の違いが若干あります。

TensorFlow Hub リポジトリのパブリッシャーファイルの正しい場所は次の通りです: [tfhub.dev/assets/docs](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs)/&lt;publisher_id&gt;/&lt;publisher_id.md&gt;

"vtab" の最小限のパブリッシャードキュメントの例をご覧ください。

```markdown
# Publisher vtab
Visual Task Adaptation Benchmark

[![Icon URL]](https://storage.googleapis.com/vtab/vtab_logo_120.png)

## VTAB
The Visual Task Adaptation Benchmark (VTAB) is a diverse, realistic and
challenging benchmark to evaluate image representations.
```

上記の例では、パブリッシャー ID、パブリッシャー名、使用するアイコンへのパス、およびより長い自由形式の Markdown ドキュメントを指定しています。パブリッシャーIDには、小文字、数字、ハイフンのみを含める必要があることに注意してください。

### パブリッシャーの名前のガイドライン

パブリッシャーの名前には GitHub ユーザー名または管理する GitHub 組織の名前を使用します。

## モデルページの Markdown 形式 {:#model}

モデルドキュメントは、複数のアドオンの構文を備えた Markdown ファイルです。最小限の例、またはより現実的な例の Markdown ファイルについては以下をご覧ください。

### ドキュメントの例

高品質のモデルドキュメントには、コードスニペット、モデルのトレーニング方法および使用目的に関する情報が含まれています。また、ユーザーがあなたのモデルを tfhub.dev で素早く検索できるように、以下に説明されているモデル固有のメタデータプロパティもご利用ください。

```markdown
# Module google/text-embedding-model/1

Simple one sentence description.

<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- task: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->

## Overview

Here we give more information about the model including how it was trained,
expected use cases, and code snippets demonstrating how to use the model:

```
Code snippet demonstrating use (e.g. for a TF model using the tensorflow_hub library)

import tensorflow_hub as hub

model = hub.KerasLayer(<model name>)
inputs = ...
output = model(inputs)
```
```

### モデルのデプロイとデプロイのグループ化

tfhub.dev では、TensorFlow SavedModel を TF.js、TFLite、およびCoral デプロイで公開できます。

Markdown ファイルの最初の行には、形式の種類が指定されている必要があります。

- SavedModel 用の `# Module publisher/model/version`
- TF.js デプロイ用の # Tfjs publisher/model/version
- Lite デプロイ用の `# Lite publisher/model/version`
- Coral デプロイ用の `# Coral publisher/model/version`

tfhub.dev の同じモデルページ上にこれらの異なる形式の同じ概念モデルを表示することを習慣付けると良いでしょう。特定の TF.js、TFLite、または Coral のデプロイと TensorFlow SavedModel モデルを関連付けるには、親モデルのタグを指定します。

```markdown
<!-- parent-model: publisher/model/version -->
```

場合によっては TensorFlow SavedModel なしで 1 つ以上のデプロイを公開することが考えられます。その場合は、プレースホルダーモデルを作成して、parent-model のタグにそのハンドルを指定します。プレースホルダーの Markdown は TensorFlow モデルの Markdownとまったく同じですが、最初の行だけは # Placeholder publisher/model/version とし、asset-path プロパティは必要ありません。

### モデルの Markdown 固有のメタデータプロパティ {:#metadata}

Markdown ファイルにはメタデータのプロパティを含めることができます。これらはユーザーがモデルを見つけやすくするためにフィルターとタグを提供するために使用されます。メタデータ属性は、Markdown ファイルの短い説明の後に Markdown コメントとして含まれます。例えば次のとおりです。

```markdown
# Module google/universal-sentence-encoder/1
Encoder of greater-than-word length text trained on a variety of data.

<!-- task: text-embedding -->
...
```

次のメタデータプロパティがサポートされています。

- format: TensorFlow モデルの場合: TensorFlow Hub 形式のモデル。有効な値として、hub はモデルがレガシーの TF1 hub 形式である場合、そして  saved_model_2 はモデルが TF2 Saved Model 経由でエクスポートされている場合に使用できます。
- `asset-path`: Google Cloud Storage バケットなど、アップロードする実際のモデルアセットへの world-readable なリモートパスです。URL は、robots.txt ファイルによって取得できる必要があります（この理由により、"https://github.com/./releases/download/." は https://github.com/robots.txt で禁止されているためサポートされていません）。必要なファイルの種類とコンテンツに関する詳しい情報は、[以下](#model-specific-asset-content)をご覧ください。
- parent-model: TF.js、TFLite、Coral モデルの場合: 同伴する SavedModel/プレースホルダーのハンドルです。
- fine-tunable: ブール値。ユーザーがモデルをファインチューニングできるかどうか。
- `task`: 問題のドメイン、たとえば "text-embedding" です。サポートされているすべての値は [task.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/task.yaml) で定義されています。
- `dataset`: モデルがトレーニングされたデータセット、たとえば、"wikipedia" です。サポートされているすべての値は、[dataset.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/dataset.yaml) で定義されています。
- `network-architecture`: モデルがトレーニングされたネットワークアーキテクチャ、たとえば "mobilenet-v3" です。 サポートされているすべての値は [network_architecture.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/network_architecture.yaml) で定義されています。
- `language`: テキストがトレーニングされた言語の言語コード、たとえば "en" です。サポートされているすべての値は [language.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/language.yaml) で定義されています。
- `license`: モデルに適用されるライセンス、たとえば "mit" です。公開されたモデルのデフォルトの想定ライセンスは、[Apache2.0ライセンス](https://opensource.org/licenses/Apache-2.0)です。サポートされているすべての値は [license.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/license.yaml) で定義されています。`custom` ライセンスでは、個別にで特別な配慮が必要になることに注意してください。
- `colab`: モデルの使用方法またはトレーニング方法を示すノートブックへの HTTPS URL（[bigbigan-resnet50 の](https://colab.sandbox.google.com/github/tensorflow/hub/blob/master/examples/colab/bigbigan_with_tf_hub.ipynb)[例](https://tfhub.dev/deepmind/bigbigan-resnet50/1)）。 `colab.research.google.com` である必要があります。 GitHub でホストされている Jupyter ノートブックには、 `https://colab.research.google.com/github/ORGANIZATION/PROJECT/ blob/master/.../my_notebook.ipynb` からアクセスできることに注意してください。
- `demo`: TF.js モデルの使用方法を示すWebサイトへの HTTPS URL（[posenet](https://teachablemachine.withgoogle.com/train/pose) の[例](https://tfhub.dev/tensorflow/tfjs-model/posenet/mobilenet/float/075/1/default/1)）。
- `interactive-visualizer`: モデルページに埋め込む必要のあるビジュアライザー名、たとえば "vision"。ビジュアライザーを表示すると、ユーザーはモデルの予測をインタラクティブに調べることができます。サポートされているすべての値は、 [interactive_visualizer.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/interactive_visualizer.yaml) で定義されています。

Markdown ドキュメントの種類は、さまざまな必須メタデータプロパティとオプションのメタデータプロパティをサポートしています。

タイプ | 必須 | オプション
--- | --- | ---
パブリッシャー |  |
コレクション | task | dataset, language,
: : : network-architecture : |  |
プレースホルダー | task | dataset, fine-tunable,
: : : interactive-visualizer, language, : |  |
: : : license, network-architecture : |  |
SavedModel | asset-path, task, | colab, dataset,
: : fine-tunable, format : interactive-visualizer, language, : |  |
: : : license, network-architecture : |  |
Tfjs | asset-path、parent-model | colab, demo, interactive-visualizer
Lite | asset-path、parent-model | colab, demo, interactive-visualizer
Coral | asset-path、parent-model | colab, interactive-visualizer

### モデル固有のアセットコンテンツ

モデルタイプに応じて、次のファイルタイプとコンテンツが必要です。

- SavedModel: 次のようなコンテンツを含む tar.gz アーカイブ:

```
saved_model.tar.gz
├── assets/            # Optional.
├── assets.extra/      # Optional.
├── variables/
│     ├── variables.data-?????-of-?????
│     └──  variables.index
├── saved_model.pb
├── keras_metadata.pb  # Optional, only required for Keras models.
└── tfhub_module.pb    # Optional, only required for TF1 models.
```

- TF.js: 次のようなコンテンツを含む tar.gz アーカイブ:

```
tf_js_model.tar.gz
├── group*
├── *.json
├── *.txt
└── *.pb
```

- TFLite: .tflite ファイル
- Coral: .tflite ファイル

tar.gz アーカイブの場合: モデルファイルがディレクトリ `my_model`（たとえば、SavedModels の場合は my_model `my_model/saved_model.pb` `my_model/model.json` 、TF.js モデルの場合は my_model / model.json）にあると仮定すると、 `cd my_model && tar -czvf ../model.tar.gz *` 経由で [tar](https://www.gnu.org/software/tar/manual/tar.html) ツールを使用して有効な tar.gz アーカイブを作成できます。

一般に、すべてのファイルとディレクトリ（圧縮されているかどうかに関係なく）は単語文字で始まる必要があるため、たとえばドットはファイル名/ディレクトリの有効なプレフィックスではありません。

## コレクションページの Markdown 形式 <br>{:#collection}

コレクションは、パブリッシャーが関連するモデルをバンドルすることにより、ユーザーの検索体験の向上を可能にする tfhub.dev の機能です。

すべてのコレクションリストは tfhub.dev をご覧ください。

リポジトリ [github.com/tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev) 内のコレクションファイルの正しい場所は、[assets/docs](https://github.com/tensorflow/tfhub.dev)/<b>publisher_name&gt;</b>/collections/<b>&lt;collection_name&gt;</b>/<b>1</b>.md です。

これは、assets/docs/<b>vtab</b>/collections/<b>benchmark</b>/<b>1</b>.md に入れられるごく小さな例です。最初の行のコレクションの名前には、ファイルパスに含まれている `collections/` 部分が含まれていないことに注意してください。

```markdown
# Collection vtab/benchmark/1
Collection of visual representations that have been evaluated on the VTAB
benchmark.

<!-- task: image-feature-vector -->

## Overview
This is the list of visual representations in TensorFlow Hub that have been
evaluated on VTAB. Results can be seen in
[google-research.github.io/task_adaptation/](https://google-research.github.io/task_adaptation/)

#### Models
|                   |
|-------------------|
| [vtab/sup-100/1](https://tfhub.dev/vtab/sup-100/1)   |
| [vtab/rotation/1](https://tfhub.dev/vtab/rotation/1) |
|------------------------------------------------------|
```

この例では、コレクションの名前、1 文の短い説明、問題ドメインのメタデータ、自由形式の Markdown ドキュメントを指定しています。
