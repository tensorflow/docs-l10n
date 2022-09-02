# iapp_wiki_qa_squad

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/iapp_wiki_qa_squad)
- [Huggingface](https://huggingface.co/datasets/iapp_wiki_qa_squad)

## iapp_wiki_qa_squad

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:iapp_wiki_qa_squad/iapp_wiki_qa_squad')
```

- **说明**：

```
`iapp_wiki_qa_squad` is an extractive question answering dataset from Thai Wikipedia articles.
It is adapted from [the original iapp-wiki-qa-dataset](https://github.com/iapp-technology/iapp-wiki-qa-dataset)
to [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) format, resulting in
5761/742/739 questions from 1529/191/192 articles.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 739
`'train'` | 5761
`'validation'` | 742

- **特征**：

```json
{
    "question_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "article_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "context": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "answers": {
        "feature": {
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "answer_start": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            },
            "answer_end": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```
