# igbo_ner

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/igbo_ner)
- [Huggingface](https://huggingface.co/datasets/igbo_ner)

## ner_data

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:igbo_ner/ner_data')
```

- **Description**:

```
Igbo Named Entity Recognition Dataset
```

- **许可**：无已知许可
- **Version**: 1.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 30715

- **特征**：

```json
{
    "content_n": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "named_entity": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sentences": {
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

## free_text

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:igbo_ner/free_text')
```

- **Description**:

```
Igbo Named Entity Recognition Dataset
```

- **许可**：无已知许可
- **Version**: 1.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 10000

- **特征**：

```json
{
    "sentences": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
