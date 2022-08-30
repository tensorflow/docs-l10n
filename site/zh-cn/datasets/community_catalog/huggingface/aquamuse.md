# aquamuse

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/aquamuse)
- [Huggingface](https://huggingface.co/datasets/aquamuse)

## abstractive

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:aquamuse/abstractive')
```

- **说明**：

```
AQuaMuSe is a novel scalable approach to automatically mine dual query based multi-document summarization datasets for extractive and abstractive summaries using question answering dataset (Google Natural Questions) and large document corpora (Common Crawl)
```

- **许可**：无已知许可
- **版本**：2.3.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 811
`'train'` | 6253
`'validation'` | 661

- **特征**：

```json
{
    "query": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "input_urls": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## extractive

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:aquamuse/extractive')
```

- **说明**：

```
AQuaMuSe is a novel scalable approach to automatically mine dual query based multi-document summarization datasets for extractive and abstractive summaries using question answering dataset (Google Natural Questions) and large document corpora (Common Crawl)
```

- **许可**：无已知许可
- **版本**：2.3.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 811
`'train'` | 6253
`'validation'` | 661

- **特征**：

```json
{
    "query": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "input_urls": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
