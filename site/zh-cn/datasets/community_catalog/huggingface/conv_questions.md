# conv_questions

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/conv_questions)
- [Huggingface](https://huggingface.co/datasets/conv_questions)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:conv_questions')
```

- **说明**：

```
ConvQuestions is the first realistic benchmark for conversational question answering over knowledge graphs.
It contains 11,200 conversations which can be evaluated over Wikidata. The questions feature a variety of complex
question phenomena like comparisons, aggregations, compositionality, and temporal reasoning.
```

- **许可**：CC BY 4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2240
`'train'` | 6720
`'validation'` | 2240

- **特征**：

```json
{
    "domain": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "seed_entity": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "seed_entity_text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "questions": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answers": {
        "feature": {
            "feature": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer_texts": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```
