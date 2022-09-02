# hover

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/hover)
- [Huggingface](https://huggingface.co/datasets/hover)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hover')
```

- **说明**：

```
HoVer is an open-domain, many-hop fact extraction and claim verification dataset built upon the Wikipedia corpus. The original 2-hop claims are adapted from question-answer pairs from HotpotQA. It is collected by a team of NLP researchers at UNC Chapel Hill and Verisk Analytics.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 4000
`'train'` | 18171
`'validation'` | 4000

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "uid": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "claim": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "supporting_facts": [
        {
            "key": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "value": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            }
        }
    ],
    "label": {
        "num_classes": 2,
        "names": [
            "NOT_SUPPORTED",
            "SUPPORTED"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "num_hops": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "hpqa_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
