# arcd

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/arcd)
- [Huggingface](https://huggingface.co/datasets/arcd)

## plain_text

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:arcd/plain_text')
```

- **Description**:

```
Arabic Reading Comprehension Dataset (ARCD) composed of 1,395 questions      posed by crowdworkers on Wikipedia articles.
```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 693
`'validation'` | 702

- **特征**：

```json
{
    "id": {
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
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```
