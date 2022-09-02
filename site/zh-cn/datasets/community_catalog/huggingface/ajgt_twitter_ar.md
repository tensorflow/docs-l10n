# ajgt_twitter_ar

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/ajgt_twitter_ar)
- [Huggingface](https://huggingface.co/datasets/ajgt_twitter_ar)

## plain_text

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:ajgt_twitter_ar/plain_text')
```

- **Description**:

```
Arabic Jordanian General Tweets (AJGT) Corpus consisted of 1,800 tweets annotated as positive and negative. Modern Standard Arabic (MSA) or Jordanian dialect.
```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1800

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 2,
        "names": [
            "Negative",
            "Positive"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
