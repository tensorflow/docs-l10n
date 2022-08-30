# great_code

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/great_code)
- [Huggingface](https://huggingface.co/datasets/great_code)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:great_code')
```

- **说明**：

```
The dataset for the variable-misuse task, described in the ICLR 2020 paper 'Global Relational Models of Source Code' [https://openreview.net/forum?id=B1lnbRNtwr]

This is the public version of the dataset used in that paper. The original, used to produce the graphs in the paper, could not be open-sourced due to licensing issues. See the public associated code repository [https://github.com/VHellendoorn/ICLR20-Great] for results produced from this dataset.

This dataset was generated synthetically from the corpus of Python code in the ETH Py150 Open dataset [https://github.com/google-research-datasets/eth_py150_open].
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 968592
`'train'` | 1798742
`'validation'` | 185656

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "source_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "has_bug": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "error_location": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "repair_candidates": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "bug_kind": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "bug_kind_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "repair_targets": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "edges": [
        [
            {
                "before_index": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "after_index": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "edge_type": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "edge_type_name": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            }
        ]
    ],
    "provenances": [
        {
            "datasetProvenance": {
                "datasetName": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "filepath": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "license": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "note": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            }
        }
    ]
}
```
