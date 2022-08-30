# hard

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/hard)
- [Huggingface](https://huggingface.co/datasets/hard)

## plain_text

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hard/plain_text')
```

- **说明**：

```
This dataset contains 93700 hotel reviews in Arabic language.The hotel reviews were collected from Booking.com website during June/July 2016.The reviews are expressed in Modern Standard Arabic as well as dialectal Arabic.The following table summarize some tatistics on the HARD Dataset.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 105698

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 5,
        "names": [
            "1",
            "2",
            "3",
            "4",
            "5"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
