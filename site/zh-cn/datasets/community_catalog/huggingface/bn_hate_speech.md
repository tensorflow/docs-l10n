# bn_hate_speech

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/bn_hate_speech)
- [Huggingface](https://huggingface.co/datasets/bn_hate_speech)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bn_hate_speech')
```

- **说明**：

```
The Bengali Hate Speech Dataset is a collection of Bengali articles collected from Bengali news articles,
news dump of Bengali TV channels, books, blogs, and social media. Emphasis was placed on Facebook pages and
newspaper sources because they attract close to 50 million followers and is a common source of opinions
and hate speech. The raw text corpus contains 250 million articles and the full dataset is being prepared
for release. This is a subset of the full dataset.

This dataset was prepared for hate-speech text classification benchmark on Bengali, an under-resourced language.
```

- **许可**：MIT License
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 3418

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
            "Personal",
            "Political",
            "Religious",
            "Geopolitical",
            "Gender abusive"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
