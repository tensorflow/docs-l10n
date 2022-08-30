# climate_fever

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/climate_fever)
- [Huggingface](https://huggingface.co/datasets/climate_fever)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:climate_fever')
```

- **说明**：

```
A dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change collected on the internet. Each claim is accompanied by five manually annotated evidence sentences retrieved from the English Wikipedia that support, refute or do not give enough information to validate the claim totalling in 7,675 claim-evidence pairs. The dataset features challenging claims that relate multiple facets and disputed cases of claims where both supporting and refuting evidence are present.
```

- **许可**：无已知许可
- **版本**：1.0.1
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1535

- **特征**：

```json
{
    "claim_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "claim": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "claim_label": {
        "num_classes": 4,
        "names": [
            "SUPPORTS",
            "REFUTES",
            "NOT_ENOUGH_INFO",
            "DISPUTED"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "evidences": [
        {
            "evidence_id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "evidence_label": {
                "num_classes": 3,
                "names": [
                    "SUPPORTS",
                    "REFUTES",
                    "NOT_ENOUGH_INFO"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "article": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "evidence": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "entropy": {
                "dtype": "float32",
                "id": null,
                "_type": "Value"
            },
            "votes": [
                {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            ]
        }
    ]
}
```
