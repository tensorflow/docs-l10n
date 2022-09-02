# art

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/art)
- [Huggingface](https://huggingface.co/datasets/art)

## anli

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:art/anli')
```

- **Description**:

```
the Abductive Natural Language Inference Dataset from AI2
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 169654
`'validation'` | 1532

- **特征**：

```json
{
    "observation_1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "observation_2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "hypothesis_1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "hypothesis_2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 3,
        "names": [
            "0",
            "1",
            "2"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
