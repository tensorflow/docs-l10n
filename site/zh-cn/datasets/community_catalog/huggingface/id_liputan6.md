# id_liputan6

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/id_liputan6)
- [Huggingface](https://huggingface.co/datasets/id_liputan6)

## canonical

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:id_liputan6/canonical')
```

- **说明**：

```
In this paper, we introduce a large-scale Indonesian summarization dataset. We harvest articles from this http URL,
an online news portal, and obtain 215,827 document-summary pairs. We leverage pre-trained language models to develop
benchmark extractive and abstractive summarization methods over the dataset with multilingual and monolingual
BERT-based models. We include a thorough error analysis by examining machine-generated summaries that have
low ROUGE scores, and expose both issues with ROUGE it-self, as well as with extractive and abstractive
summarization models.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 10972
`'train'` | 193883
`'validation'` | 10972

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "clean_article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "clean_summary": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "extractive_summary": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## xtreme

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:id_liputan6/xtreme')
```

- **说明**：

```
In this paper, we introduce a large-scale Indonesian summarization dataset. We harvest articles from this http URL,
an online news portal, and obtain 215,827 document-summary pairs. We leverage pre-trained language models to develop
benchmark extractive and abstractive summarization methods over the dataset with multilingual and monolingual
BERT-based models. We include a thorough error analysis by examining machine-generated summaries that have
low ROUGE scores, and expose both issues with ROUGE it-self, as well as with extractive and abstractive
summarization models.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 3862
`'validation'` | 4948

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "clean_article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "clean_summary": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "extractive_summary": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
