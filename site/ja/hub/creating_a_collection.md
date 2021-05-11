<!--* freshness: { owner: 'maringeo' reviewed: '2021-04-12' review_interval: '6 months' } *-->

# コレクションを作成する

コレクションは、パブリッシャーが関連するモデルをバンドルすることにより、ユーザーの検索体験の向上を可能にする tfhub.dev の機能です。

[すべてのコレクションリスト](https://tfhub.dev/s?subtype=model-family)は tfhub.dev をご覧ください。

[github.com/tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev) リポジトリのコレクションファイルの正しい場所は [assets/docs](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs)/<b>&lt;publisher_name&gt;</b>/collections/<b>&lt;collection_name&gt;</b>/<b>1</b>.md です。

次は、assets/docs/<b>vtab</b>/collections/<b>benchmark</b>/<b>1</b>.md に入れられるごく小さな例です。1 行目のコレクションの名前がファイルの名前より短いところに注意してください。

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
