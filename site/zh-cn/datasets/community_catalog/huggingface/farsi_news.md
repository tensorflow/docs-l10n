# farsi_news

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/farsi_news)
- [Huggingface](https://huggingface.co/datasets/farsi_news)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:farsi_news')
```

- **Description**:

```

```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'hamshahri'` | 2203
`'radiofarda'` | 284

- **特征**：

```json
{
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "summary": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "link": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "tags": {
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
