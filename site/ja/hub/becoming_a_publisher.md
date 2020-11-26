<!--* freshness: { owner: 'maringeo' } *-->

# パブリッシャーになるには

## 利用規約

公開のためにモデルを提出することで、[https://tfhub.dev/terms](https://tfhub.dev/terms) の TensorFlow Hub 利用規約に同意したことになります。

## 公開プロセスの概要

公開の全プロセスは以下のように構成されています。

1. モデルを作成する（[モデルのエクスポート](exporting_tf2_saved_model.md)方法を参照）
2. ドキュメントを記述する（[モデルドキュメントの記述](writing_model_documentation.md)方法を参照）
3. 公開リクエストを作成する（[コントリビューション](contribute_a_model.md)方法を参照）

## パブリッシャーページ固有のマークダウン形式

パブリッシャードキュメントは、[モデルドキュメントの記述](writing_model_documentation)ガイドで説明されているのと同じ種類のマークダウンファイルで宣言されますが、構文の違いが若干あります。

TensorFlow Hub リポジトリのパブリッシャーファイルの正しい場所は次の通りです: [hub/tfhub_dev/assets/](https://github.com/tensorflow/hub/tree/master/tfhub_dev/assets)/<publisher_name>/<publisher_name.md>

最小限のパブリッシャードキュメントの例をご覧ください。

```markdown
# Publisher vtab
Visual Task Adaptation Benchmark

[![Icon URL]](https://storage.googleapis.com/vtab/vtab_logo_120.png)

## VTAB
The Visual Task Adaptation Benchmark (VTAB) is a diverse, realistic and
challenging benchmark to evaluate image representations.
```

上記の例では、パブリッシャーの名前、短い説明、使用するアイコンへのパス、長い自由形式のマークダウンドキュメントを指定しています。

### パブリッシャーの名前のガイドライン

パブリッシャーの名前には GitHub ユーザー名または管理する GitHub 組織の名前を使用します。
