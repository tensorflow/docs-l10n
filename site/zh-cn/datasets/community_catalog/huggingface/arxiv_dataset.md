# arxiv_dataset

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/arxiv_dataset)
- [Huggingface](https://huggingface.co/datasets/arxiv_dataset)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:arxiv_dataset')
```

- **Description**:

```
A dataset of 1.7 million arXiv articles for applications like trend analysis, paper recommender engines, category prediction, co-citation networks, knowledge graph construction and semantic search interfaces.
```

- **许可**：无已知许可
- **Version**: 1.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1796911

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "submitter": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "authors": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "comments": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "journal-ref": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "doi": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "report-no": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "categories": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "license": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "abstract": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "update_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
