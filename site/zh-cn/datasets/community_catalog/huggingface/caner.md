# caner

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/caner)
- [Huggingface](https://huggingface.co/datasets/caner)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:caner')
```

- **说明**：

```
Classical Arabic Named Entity Recognition corpus as a new corpus of tagged data that can be useful for handling the issues in recognition of Arabic named entities.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 258240

- **特征**：

```json
{
    "token": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "ner_tag": {
        "num_classes": 21,
        "names": [
            "Allah",
            "Book",
            "Clan",
            "Crime",
            "Date",
            "Day",
            "Hell",
            "Loc",
            "Meas",
            "Mon",
            "Month",
            "NatOb",
            "Number",
            "O",
            "Org",
            "Para",
            "Pers",
            "Prophet",
            "Rlig",
            "Sect",
            "Time"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
