# igbo_ner

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/igbo_ner)
- [Huggingface](https://huggingface.co/datasets/igbo_ner)

## ner_data

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:igbo_ner/ner_data')
```

- **说明**：

```
Igbo Named Entity Recognition Dataset
```

- **许可**：无已知许可
- **版本**：1.0.0
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

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:igbo_ner/free_text')
```

- **说明**：

```
Igbo Named Entity Recognition Dataset
```

- **许可**：无已知许可
- **版本**：1.0.0
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
