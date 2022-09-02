# arsentd_lev

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/arsentd_lev)
- [Huggingface](https://huggingface.co/datasets/arsentd_lev)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:arsentd_lev')
```

- **说明**：

```
The Arabic Sentiment Twitter Dataset for Levantine dialect (ArSenTD-LEV) contains 4,000 tweets written in Arabic and equally retrieved from Jordan, Lebanon, Palestine and Syria.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 4000

- **特征**：

```json
{
    "Tweet": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Country": {
        "num_classes": 4,
        "names": [
            "jordan",
            "lebanon",
            "syria",
            "palestine"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "Topic": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Sentiment": {
        "num_classes": 5,
        "names": [
            "negative",
            "neutral",
            "positive",
            "very_negative",
            "very_positive"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "Sentiment_Expression": {
        "num_classes": 3,
        "names": [
            "explicit",
            "implicit",
            "none"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "Sentiment_Target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
