<!--* freshness: { owner: 'maringeo' reviewed: '2020-12-30' review_interval: '3 months' } *-->

# 创建集合

集合是 tfhub.dev 的一项功能，借助集合，发布者可以将相关模型捆绑在一起来改善用户搜索体验。

请参见 tfhub.dev 上的[所有集合的列表](https://tfhub.dev/s?subtype=model-family)。

[github.com/tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev) 仓库中集合文件的正确位置为 [assets/docs](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs)/<b>&lt;publisher_name&gt;</b>/collections/<b>&lt;collection_name&gt;</b>/<b>1</b>.md

下面是一个将进入 assets/docs/<b>vtab</b>/collections/<b>benchmark</b>/<b>1</b>.md 的简单示例。请注意，第一行中集合的名称比文件的名称短。

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

该示例指定了集合的名称、简短的一句话描述、问题领域元数据和形式不限的 Markdown 文档。
