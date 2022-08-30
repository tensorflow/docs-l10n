# clickbait_news_bg

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/clickbait_news_bg)
- [Huggingface](https://huggingface.co/datasets/clickbait_news_bg)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:clickbait_news_bg')
```

- **说明**：

```
Dataset with clickbait and fake news in Bulgarian. Introduced for the Hack the Fake News 2017.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2815
`'validation'` | 761

- **特征**：

```json
{
    "fake_news_score": {
        "num_classes": 2,
        "names": [
            "legitimate",
            "fake"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "click_bait_score": {
        "num_classes": 2,
        "names": [
            "normal",
            "clickbait"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "content_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "content_url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "content_published_time": {
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
