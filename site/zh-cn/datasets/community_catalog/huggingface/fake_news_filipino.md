# fake_news_filipino

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/fake_news_filipino)
- [Huggingface](https://huggingface.co/datasets/fake_news_filipino)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:fake_news_filipino')
```

- **Description**:

```
Low-Resource Fake News Detection Corpora in Filipino. The first of its kind. Contains 3,206 expertly-labeled news samples, half of which are real and half of which are fake.
```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 3206

- **特征**：

```json
{
    "label": {
        "num_classes": 2,
        "names": [
            "0",
            "1"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
