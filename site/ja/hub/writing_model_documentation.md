<!--* freshness: { owner: 'wgierke' reviewed: '2021-02-25' review_interval: '3 months' } *-->

# モデルドキュメントの作成

tfhub.dev にモデルを貢献するには、Markdown のドキュメントを提供する必要があります。tfhub.dev にモデルを追加するための詳細なプロセスについては、[モデルの貢献](contribute_a_model.md)ガイドをご覧ください。

## Markdown ドキュメントの種類

tfhub.dev では次の 3 種類の Markdown ドキュメントが使用されています。

- Publisher Markdown - パブリッシャに関する情報が含まれます（[パブリッシャになるには](publish.md) ガイドをご覧ください）。
- Model Markdown - 特定のモデルに関する情報が含まれます。
- Collection Markdown - パブリッシャが定義したモデルコレクションに関する情報が含まれます（詳細は、[コレクションの作成](creating_a_collection.md)ガイドをご覧ください）。

## コンテンツの編成

[TensorFlow Hub GitHub](https://github.com/tensorflow/hub) リポジトリに貢献する際は、次のコンテンツで編成することをお勧めします。

- 各パブリッシャディレクトリは `assets` ディレクトリに配置します。
- オプションの `models` ディレクトリと `collections` ディレクトリは、各パブリッシャディレクトリに含めます。
- 各モデルには、`assets/publisher_name/models` の下にそれぞれのディレクトリを用意します。
- 各コレクションには、`assets/publisher_name/collections` の下にそれぞれのディレクトリを用意します。

パブリッシャとコレクションの Markdown はバージョン管理されていませんが、モデルにはさまざまなバージョンを設けることができます。モデルの各バージョンには、そのバージョンにちなんだ個別の Markdown ファイル（1.md、2.md など）が必要です。

特定のモデルのすべてのバージョンを、そのモデルのディレクトリに配置してください。

Markdown コンテンツの編成を次の図に示しています。

```
assets
├── publisher_name_a
│   ├── publisher_name_a.md  -> Documentation of the publisher.
│   └── models
│       └── model          -> Model name with slashes encoded as sub-path.
│           ├── 1.md       -> Documentation of the model version 1.
│           └── 2.md       -> Documentation of the model version 2.
├── publisher_name_b
│   ├── publisher_name_b.md  -> Documentation of the publisher.
│   ├── models
│   │   └── ...
│   └── collections
│       └── collection     -> Documentation for the collection feature.
│           └── 1.md
├── publisher_name_c
│   └── ...
└── ...
```

## モデルページ固有の Markdown 形式

モデルドキュメントは、複数のアドオンの構文を備えた Markdown ファイルです。最小限の例、または[より現実的な例の Markdown ファイル](https://github.com/tensorflow/tfhub.dev/blob/master/examples/docs/tf2_model_example.md)については以下をご覧ください。

### ドキュメントの例

高品質のモデルドキュメントには、コードスニペット、モデルのトレーニング方法および使用目的に関する情報が含まれています。また、ユーザーがあなたのモデルを tfhub.dev で素早く検索できるように、[以下に説明されている](#model-markdown-specific-metadata-properties)モデル固有のメタデータプロパティもご利用ください。

```markdown
# Module google/text-embedding-model/1

Simple one sentence description.

<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- module-type: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->

## Overview

Here we give more information about the model including how it was trained,
expected use cases, and code snippets demonstrating how to use the model:

``
Code snippet demonstrating use (e.g. for a TF model using the tensorflow_hub library)

import tensorflow_hub as hub

model = hub.KerasLayer(<model name>)
inputs = ...
output = model(inputs)
``
```

### モデルのデプロイとデプロイのグループ化

tfhub.dev では、TensorFlow モデルを TF.js、TFLite、およびCoral デプロイで公開できます。

Markdown ファイルの最初の行には、デプロイ形式の種類が指定されている必要があります。

- TF.js デプロイ用の `# Tfjs publisher/model/version`
- Lite デプロイ用の `# Lite publisher/model/version`
- Coral デプロイ用の `# Coral publisher/model/version`

tfhub.dev の同じモデルページ上にこれらの異なるデプロイを表示することを習慣付けると良いでしょう。特定の TF.js、TFLite、または Coral のデプロイと TensorFlow モデルを関連付けるには、親モデルのタグを指定します。

```markdown
<!-- parent-model: publisher/model/version -->
```

場合によっては TensorFlow SavedModel なしで 1 つ以上のデプロイを公開することが考えられます。その場合は、プレースホルダーモデルを作成して、`parent-model` のタグにそのハンドルを指定します。プレースホルダーの Markdown は TensorFlow モデルの Markdownとまったく同じですが、最初の行だけは `# Placeholder publisher/model/version` とし、`asset-path` プロパティは必要ありません。

### Model Markdown 固有のメタデータプロパティ

Markdown ファイルにはメタデータプロパティを含めることができます。これらは、Markdown ファイルの説明の後に、Markdown のコメントとして示されます。次に例を示します。

```
# Module google/universal-sentence-encoder/1
Encoder of greater-than-word length text trained on a variety of data.

<!-- module-type: text-embedding -->
...
```

次のメタデータプロパティがあります。

- `format`: TensorFlow モデルの場合: TensorFlow Hub 形式のモデル。有効な値として、`hub` はモデルがレガシーの [TF1 hub 形式](exporting_hub_format.md)である場合、そして  `saved_model_2` はモデルが [TF2 Saved Model](exporting_tf2_saved_model.md) 経由でエクスポートされている場合に使用できます。
- `asset-path`: Google Cloud Storage バケットなど、アップロードする実際のモデルアセットへの world-readable なリモートパスです。URL は、robots.txt ファイルによって取得できる必要があります（この理由により、"https://github.com/.*/releases/download/.*" は https://github.com/robots.txt で禁止されているためサポートされていません）。
- `parent-model`: TF.js、TFLite、Coral モデルの場合: 同伴する SavedModel/プレースホルダーのハンドルです。
- `module-type`: 問題の分野。"text-embedding" や "image-classification" など。
- `dataset`: モデルがトレーニングされたデータセット。"ImageNet-21k" や "Wikipedia" など。
- `network-architecture`: モデルが基づくネットワークアーキテクチャ。"BERT" や "Mobilenet V3" など。
- `language`: テキストモデルがトレーニングされた言語の言語コード。"en" や "fr" など。
- `fine-tunable`: ブール値。ユーザーがモデルをファインチューニングできるかどうか。
- `license`: モデルに適用されるライセンス。公開されたモデルのライセンスは [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0) であることがデフォルトで前提となっています。使用できる他のオプションは、[OSI Approved Licenses](https://opensource.org/licenses) にリストされています。可能な（リテラル）値: `Apache-2.0`、`BSD-3-Clause`、`BSD-2-Clause`、`GPL-2.0`、`GPL-3.0`、`LGPL-2.0`、`LGPL-2.1`、`LGPL-3.0`、`MIT`、`MPL-2.0`、`CDDL-1.0`、`EPL-2.0`、`custom`。カスタムライセンスには、ケースごとに特別な考慮事項があります。

Markdown ドキュメントの種類は、さまざまな必須メタデータプロパティとオプションのメタデータプロパティをサポートしています。

種類 | 必須 | オプション
--- | --- | ---
パブリッシャ |  |
コレクション | module-type | dataset、language
:             :                          : ネットワークアーキテクチャ             : |  |
プレースホルダー | module-type | dataset、fine-tunable、language
:             :                          : ライセンス、ネットワークアーキテクチャ    : |  |
SavedModel | asset-path、module-type | dataset、language、license
:             : ファインチューニング可能、形式     : ネットワークアーキテクチャ             : |  |
Tfjs | asset-path、parent-model |
Lite | asset-path、parent-model |
Coral | asset-path、parent-model | 
