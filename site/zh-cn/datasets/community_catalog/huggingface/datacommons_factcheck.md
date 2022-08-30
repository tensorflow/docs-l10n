# datacommons_factcheck

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/datacommons_factcheck)
- [Huggingface](https://huggingface.co/datasets/datacommons_factcheck)

## fctchk_politifact_wapo

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:datacommons_factcheck/fctchk_politifact_wapo')
```

- **说明**：

```
A dataset of fact checked claims by news media maintained by datacommons.org
```

- **许可**：CC-BY-NC-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 5632

- **特征**：

```json
{
    "reviewer_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "claim_text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_rating": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "claim_author_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "claim_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## weekly_standard

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:datacommons_factcheck/weekly_standard')
```

- **说明**：

```
A dataset of fact checked claims by news media maintained by datacommons.org
```

- **许可**：CC-BY-NC-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 132

- **特征**：

```json
{
    "reviewer_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "claim_text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_rating": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "claim_author_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "claim_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
