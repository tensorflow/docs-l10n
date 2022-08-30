# emotone_ar

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/emotone_ar)
- [Huggingface](https://huggingface.co/datasets/emotone_ar)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:emotone_ar')
```

- **Description**:

```
Dataset of 10065 tweets in Arabic for Emotion detection in Arabic text
```

- **许可**：无已知许可
- **Version**: 0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 10065

- **特征**：

```json
{
    "tweet": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 8,
        "names": [
            "none",
            "anger",
            "joy",
            "sadness",
            "love",
            "sympathy",
            "surprise",
            "fear"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
