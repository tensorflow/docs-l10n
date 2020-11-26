<!--* freshness: { owner: 'maringeo' reviewed: '2020-09-14' review_interval: '3 months' } *-->

# モデルのコントリビューション

## モデルを提供する

マークダウンファイルの適切な場所を識別した後（[モデルドキュメントの記述](writing_model_documentation.md)ガイドを参照）、以下のいずれかの方法で [tensorflow/hub](https://github.com/tensorflow/hub/tree/master/tensorflow_hub) のマスターブランチにファイルをプルすることができます。

### Git CLI で提供する

識別されたマークダウンファイルのパスが `tfhub_dev/assets/publisher/model/1.md` であると仮定して、標準の Git[Hub] の手順に従って、新たに追加されたファイルで新しいプルリクエストを作成することができます。

これにはまず、TensorFlow Hub の GitHub リポジトリをフォークし、[このフォークからプルリクエストを](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork) TensorFlow Hub のマスターブランチに作成することから始まります。

以下は、フォークされたリポジトリのマスターブランチに新しいファイルを追加するために必要な、典型的な CLI の Git コマンドです。

```bash
git clone https://github.com/[github_username]/hub.git
cd hub
mkdir -p tfhub_dev/assets/publisher/model
cp my_markdown_file.md ./tfhub_dev/assets/publisher/model/1.md
git add *
git commit -m "Added model file."
git push origin master
```

### GitHub GUI で提出する

もう少し簡単な提供方法として、GitHub のグラフィカルユーザーインターフェース (GUI) を利用する方法があります。GitHub では、[新規ファイル](https://help.github.com/en/github/managing-files-in-a-repository/creating-new-files)や[ファイル編集](https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-your-repository)の PR を GUI から直接作成することができます。

1. <a>TensorFlow Hub の GitHub のページ</a>で<code>Create new file</code>ボタンを押します。
2. 適切なファイルパスを設定します: `hub/tfhub_dev/assets/publisher/model/1.md`
3. 既存のマークダウンをコピーして貼り付けます。
4. 一番下で「Create a new branch for this commit and start a pull request（このコミットの新しいブランチを作成してプルリクエストを開始する）」を選択します。

## モデルページ固有のマークダウン形式

モデルドキュメントは、複数のアドオンの構文を備えたマークダウンファイルです。最小限の例については以下の例を、[より現実的な例についてはマークダウンファイルを](https://github.com/tensorflow/hub/blob/master/tfhub_dev/examples/example-markdown.md)参照してください。

### ドキュメントの例

高品質のモデルドキュメントには、コードスニペット、モデルのトレーニング方法および使用目的に関する情報が含まれています。また、モデル固有の[下記の](#model-markdown-specific-metadata-properties)メタデータプロパティおよび[モデルドキュメントの記述](writing_model_documentation.md)で説明している一般的なプロパティを利用する必要があります。

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

### モデルのデプロイとデプロイのグループ化を同時に行う

tfhub.dev では TensorFlow モデルの TFJS、TFLite、Coral のデプロイを公開することができます。

マークダウンファイルの最初の行では、デプロイ形式の種類を指定する必要があります。以下を使用します。

- `# Tfjs publisher/model/version` TFJS のデプロイ用
- `# Lite publisher/model/version` Lite のデプロイ用
- `# Coral publisher/model/version` Coral のデプロイ用

 tfhub.dev の同じモデルページ上にこれらの異なるデプロイを表示することは良い習慣です。特定の TFJS、TFLite、Coral のデプロイと TensorFlow モデルの関連付けには、親モデルのタグを指定します。

```markdown
<!-- parent-model: publisher/model/version -->
```

場合によっては TensorFlow モデルなしで 1 つ以上のデプロイを公開することが考えられます。その場合は、プレースホルダーモデルを作成して、親モデルのタグにそのハンドルを指定します。プレースホルダのマークダウンは TensorFlow モデルのマークダウンと全く同じですが、最初の行だけは`# Placeholder publisher/model/version`とし、`asset-path`プロパティは必要ありません。

### モデルのマークダウン固有のメタデータプロパティ

[モデルドキュメントの記述](writing_model_documentation.md)で説明されている共有メタデータのプロパティとは別に、モデルのマークダウンでは以下のプロパティをサポートしています。

- `fine-tunable`: モデルが微調整可能かどうか。
- `format`: モデルの TensorFlow Hub 形式。有効な値は、モデルがレガシーの [TF1 Hub形式](exporting_hub_format.md)でエクスポートされた場合は`hub`、モデルが [TF2 Saved Model](exporting_tf2_saved_model.md) でエクスポートされた場合は`saved_model_2`。
- `asset-path`: Google Cloud Storage バケットなどにアップロードする、実際のモデルアセットへの誰もが読めるリモートパス。
- `licence`: 以下の項目をご覧ください。

### ライセンス情報

公開モデルのデフォルトの想定ライセンスは [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0) です。その他に承認されているライセンスのオプションは、[OSI 承認ライセンス](https://opensource.org/licenses)に記載されています。可能な（リテラルな）値は以下の通りです。

- `Apache-2.0`
- `BSD-3-Clause`
- `BSD-2-Clause`
- `GPL-2.0`
- `GPL-3.0`
- `LGPL-2.0`
- `LGPL-2.1`
- `LGPL-3.0`
- `MIT`
- `MPL-2.0`
- `CDDL-1.0`
- `EPL-2.0`
- `custom` - カスタムライセンスについては、個々のケースに合わせて特別な配慮が必要です。

Apache 2.0 以外のライセンスのメタデータ行の記述例:

```markdown
<!-- license: BSD-3-Clause -->
```
