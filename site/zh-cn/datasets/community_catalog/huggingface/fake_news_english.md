# fake_news_english

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/fake_news_english)
- [Huggingface](https://huggingface.co/datasets/fake_news_english)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:fake_news_english')
```

- **说明**：

```
Fake news has become a major societal issue and a technical challenge for social media companies to identify. This content is difficult to identify because the term "fake news" covers intentionally false, deceptive stories as well as factual errors, satire, and sometimes, stories that a person just does not like. Addressing the problem requires clear definitions and examples. In this work, we present a dataset of fake news and satire stories that are hand coded, verified, and, in the case of fake news, include rebutting stories. We also include a thematic content analysis of the articles, identifying major themes that include hyperbolic support or condemnation of a gure, conspiracy theories, racist themes, and discrediting of reliable sources. In addition to releasing this dataset for research use, we analyze it and show results based on language that are promising for classification purposes. Overall, our contribution of a dataset and initial analysis are designed to support future work by fake news researchers.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 492

- **特征**：

```json
{
    "article_number": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "url_of_article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "fake_or_satire": {
        "num_classes": 2,
        "names": [
            "Satire",
            "Fake"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "url_of_rebutting_article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
