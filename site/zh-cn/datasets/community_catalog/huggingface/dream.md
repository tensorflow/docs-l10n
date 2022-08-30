# dream

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/dream)
- [Huggingface](https://huggingface.co/datasets/dream)

## plain_text

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:dream/plain_text')
```

- **Description**:

```
DREAM is a multiple-choice Dialogue-based REAding comprehension exaMination dataset. In contrast to existing reading comprehension datasets, DREAM is the first to focus on in-depth multi-turn multi-party dialogue understanding.
```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2041
`'train'` | 6116
`'validation'` | 2040

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "dialogue_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "dialogue": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choice": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
