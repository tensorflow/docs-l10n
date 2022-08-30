# ar_res_reviews

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/ar_res_reviews)
- [Huggingface](https://huggingface.co/datasets/ar_res_reviews)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:ar_res_reviews')
```

- **说明**：

```
Dataset of 8364 restaurant reviews scrapped from qaym.com in Arabic for sentiment analysis
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 8364

- **特征**：

```json
{
    "polarity": {
        "num_classes": 2,
        "names": [
            "negative",
            "positive"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "restaurant_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "user_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
