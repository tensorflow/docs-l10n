# amazon_polarity

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/amazon_polarity)
- [Huggingface](https://huggingface.co/datasets/amazon_polarity)

## amazon_polarity

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_polarity/amazon_polarity')
```

- **说明**：

```
The Amazon reviews dataset consists of reviews from amazon.
The data span a period of 18 years, including ~35 million reviews up to March 2013.
Reviews include product and user information, ratings, and a plaintext review.
```

- **许可**：Apache License 2.0
- **版本**：3.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 400000
`'train'` | 3600000

- **特征**：

```json
{
    "label": {
        "num_classes": 2,
        "names": [
            "negative",
            "positive"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "content": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
