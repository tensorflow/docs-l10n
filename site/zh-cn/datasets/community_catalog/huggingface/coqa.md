# coqa

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/coqa)
- [Huggingface](https://huggingface.co/datasets/coqa)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:coqa')
```

- **Description**:

```
CoQA: A Conversational Question Answering Challenge
```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 7199
`'validation'` | 500

- **特征**：

```json
{
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "story": {
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
            "input_text": {
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
