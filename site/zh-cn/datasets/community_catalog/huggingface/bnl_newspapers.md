# bnl_newspapers

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/bnl_newspapers)
- [Huggingface](https://huggingface.co/datasets/bnl_newspapers)

## processed

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bnl_newspapers/processed')
```

- **说明**：

```
Digitised historic newspapers from the Bibliothèque nationale (BnL) - the National Library of Luxembourg.
```

- **许可**：CC0
- **版本**：1.17.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 537558

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "ispartof": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "pub_date": {
        "dtype": "timestamp[s]",
        "id": null,
        "_type": "Value"
    },
    "publisher": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "article_type": {
        "num_classes": 18,
        "names": [
            "ADVERTISEMENT_SECTION",
            "BIBLIOGRAPHY",
            "CHAPTER",
            "INDEX",
            "CONTRIBUTION",
            "TABLE_OF_CONTENTS",
            "WEATHER",
            "SHIPPING",
            "SECTION",
            "ARTICLE",
            "TITLE_SECTION",
            "DEATH_NOTICE",
            "SUPPLEMENT",
            "TABLE",
            "ADVERTISEMENT",
            "CHART_DIAGRAM",
            "ILLUSTRATION",
            "ISSUE"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "extent": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    }
}
```
