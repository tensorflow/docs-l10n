# hausa_voa_topics

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/hausa_voa_topics)
- [Huggingface](https://huggingface.co/datasets/hausa_voa_topics)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hausa_voa_topics')
```

- **说明**：

```
A collection of news article headlines in Hausa from VOA Hausa.
Each headline is labeled with one of the following classes: Nigeria,
Africa, World, Health or Politics.

The dataset was presented in the paper:
Hedderich, Adelani, Zhu, Alabi, Markus, Klakow: Transfer Learning and
Distant Supervision for Multilingual Transformer Models: A Study on
African Languages (EMNLP 2020).
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 582
`'train'` | 2045
`'validation'` | 290

- **特征**：

```json
{
    "news_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 5,
        "names": [
            "Africa",
            "Health",
            "Nigeria",
            "Politics",
            "World"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
