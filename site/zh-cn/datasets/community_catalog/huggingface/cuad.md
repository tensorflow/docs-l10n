# cuad

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/cuad)
- [Huggingface](https://huggingface.co/datasets/cuad)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:cuad')
```

- **Description**:

```
Contract Understanding Atticus Dataset (CUAD) v1 is a corpus of more than 13,000 labels in 510
commercial legal contracts that have been manually labeled to identify 41 categories of important
clauses that lawyers look for when reviewing contracts in connection with corporate transactions.
```

- **许可**：CUAD 在 Creative Commons Attribution 4.0 (CC BY 4.0) 许可下许可。
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 4182
`'train'` | 22450

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
