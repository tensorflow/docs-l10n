# gooaq

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/gooaq)
- [Huggingface](https://huggingface.co/datasets/gooaq)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gooaq')
```

- **说明**：

```
GooAQ is a large-scale dataset with a variety of answer types. This dataset contains over
5 million questions and 3 million answers collected from Google. GooAQ questions are collected
semi-automatically from the Google search engine using its autocomplete feature. This results in
naturalistic questions of practical interest that are nonetheless short and expressed using simple
language. GooAQ answers are mined from Google's responses to our collected questions, specifically from
the answer boxes in the search results. This yields a rich space of answer types, containing both
textual answers (short and long) as well as more structured ones such as collections.
```

- **许可**：在 Apache License, Version 2.0 下许可
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2500
`'train'` | 3112679
`'validation'` | 2500

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "short_answer": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "answer": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "answer_type": {
        "num_classes": 6,
        "names": [
            "feat_snip",
            "collection",
            "knowledge",
            "unit_conv",
            "time_conv",
            "curr_conv"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
