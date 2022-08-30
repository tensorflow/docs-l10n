# commonsense_qa

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/commonsense_qa)
- [Huggingface](https://huggingface.co/datasets/commonsense_qa)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:commonsense_qa')
```

- **说明**：

```
CommonsenseQA is a new multiple-choice question answering dataset that requires different types of commonsense knowledge
to predict the correct answers . It contains 12,102 questions with one correct answer and four distractor answers.
The dataset is provided in two major training/validation/testing set splits: "Random split" which is the main evaluation
split, and "Question token split", see paper for details.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1140
`'train'` | 9741
`'validation'` | 1221

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question_concept": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "label": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
