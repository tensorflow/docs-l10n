<!--* freshness: { owner: 'arnoegw' reviewed: '2020-07-06' } *-->

# TF Hub からダウンロードしたモデルのキャッシュ

## 概要

`tensorflow_hub` ライブラリは、tfhub.dev（またはその他の[ホスティングサイト](hosting.md)）からダウンロードされて解凍されたモデルをファイルシステム上にキャッシュします。ダウンロード場所はデフォルトではローカルの一時ディレクトリですが、`TFHUB_CACHE_DIR` 環境変数を設定するか（推奨）、コマンド行に `--tfhub_cache_dir` フラグを指定して変更できます。永続的な場所を使用する場合、自動クリーンアップは実行されませんのでご注意ください。

実際の Python コードで `tensorflow_hub` 関数を呼び出す際には、モデルの正規の tfhub.dev URL を引き続き使用できます。この URL はシステム間で移植可能で、ドキュメントに移動できます。

## 具体的な実行環境

デフォルトの `TFHUB_CACHE_DIR` を変更する必要性の有無やその変更方法は、実行環境によって異なります。

### ワークステーションでローカルに実行する

ワークステーションで TensorFlow プログラムを実行しているユーザーの場合、ほとんどの場合はデフォルトの場所 `/tmp/tfhub_modules` か、Python で `os.path.join(tempfile.gettempdir(), "tfhub_modules")` が返す任意の場所をそのまま使用できます。

システムを再起動後も永続的なキャッシュを利用したいユーザーは、代わりに `TFHUB_CACHE_DIR` をホームディレクトリ内の場所に設定できます。たとえば、Linux システムで bash シェルを使用している場合は、次のような行を `~/.bashrc` に追加できます。

```bash
export TFHUB_CACHE_DIR=$HOME/.cache/tfhub_modules
```

その後シェルを再起動すると、この場所が使用されます。

### Colab ノートブックの TPU で実行する

[Colab](https://colab.research.google.com/) ノートブック内の CPU と GPU で TensorFlow を実行する場合、デフォルトのローカルキャッシュの場所を使用するだけで十分です。

TPU での実行は、デフォルトのローカルキャッシュの場所にアクセスできない別のマシンに委任されます。自前の Google Cloud Storage（GCS）バケットを所有している場合は、次のようなコードを使用してバケット内のディレクトリをキャッシュ場所に設定することでこの問題を回避できます。

```python
import os
os.environ["TFHUB_CACHE_DIR"] = "gs://my-bucket/tfhub-modules-cache"
```

このコードは `tensorflow_hub` ライブラリを呼び出す前に追加してください。
