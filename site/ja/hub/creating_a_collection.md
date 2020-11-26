<!--* freshness: { owner: 'maringeo' reviewed: '2020-09-14' review_interval: '3 months' } *-->

# コレクションを作成する

コレクションは、パブリッシャーが関連するモデルをバンドルすることにより、ユーザーの検索体験の向上を可能にする tfhub.dev の機能です。

tfhub.dev の[すべてのコレクションリスト](https://tfhub.dev/s?subtype=model-family)をご覧ください。

TensorFlow Hub リポジトリのコレクションファイルの正しい場所は次の通りです: [hub/tfhub_dev/assets/](https://github.com/tensorflow/hub/tree/master/tfhub_dev/assets)/<publisher_name>/<collection_name>/<collection_name.md>

パブリッシャードキュメントの最小の例をご覧ください。

```markdown
# Collection vtab/benchmark/1
Collection of visual representations that have been evaluated on the VTAB
benchmark.

<!-- module-type: image-feature-vector -->

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

この例では、コレクションの名前、1 文の短い説明、問題ドメインのメタデータ、自由形式のマークダウンドキュメントを指定しています。
