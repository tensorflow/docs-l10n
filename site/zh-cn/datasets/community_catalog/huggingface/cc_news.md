# cc_news

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/cc_news)
- [Huggingface](https://huggingface.co/datasets/cc_news)

## plain_text

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cc_news/plain_text')
```

- **说明**：

```
CC-News containing news articles from news sites all over the world The data is available on AWS S3 in the Common Crawl bucket at /crawl-data/CC-NEWS/. This version of the dataset has 708241 articles. It represents a small portion of English  language subset of the CC-News dataset created using news-please(Hamborg et al.,2017) to collect and extract English language portion of CC-News.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 708241

- **特征**：

```json
{
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "domain": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "description": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "image_url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
