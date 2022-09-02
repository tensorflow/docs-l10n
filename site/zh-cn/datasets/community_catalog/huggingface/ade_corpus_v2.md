# ade_corpus_v2

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/ade_corpus_v2)
- [Huggingface](https://huggingface.co/datasets/ade_corpus_v2)

## Ade_corpus_v2_classification

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:ade_corpus_v2/Ade_corpus_v2_classification')
```

- **说明**：

```
ADE-Corpus-V2  Dataset: Adverse Drug Reaction Data.
 This is a dataset for Classification if a sentence is ADE-related (True) or not (False) and Relation Extraction between Adverse Drug Event and Drug.
 DRUG-AE.rel provides relations between drugs and adverse effects.
 DRUG-DOSE.rel provides relations between drugs and dosages.
 ADE-NEG.txt provides all sentences in the ADE corpus that DO NOT contain any drug-related adverse effects.
```

- **许可证**：无已知许可证
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 23516

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 2,
        "names": [
            "Not-Related",
            "Related"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## Ade_corpus_v2_drug_ade_relation

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:ade_corpus_v2/Ade_corpus_v2_drug_ade_relation')
```

- **说明**：

```
ADE-Corpus-V2  Dataset: Adverse Drug Reaction Data.
 This is a dataset for Classification if a sentence is ADE-related (True) or not (False) and Relation Extraction between Adverse Drug Event and Drug.
 DRUG-AE.rel provides relations between drugs and adverse effects.
 DRUG-DOSE.rel provides relations between drugs and dosages.
 ADE-NEG.txt provides all sentences in the ADE corpus that DO NOT contain any drug-related adverse effects.
```

- **许可证**：无已知许可证
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 6821

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "drug": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "effect": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "indexes": {
        "drug": {
            "feature": {
                "start_char": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "end_char": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        },
        "effect": {
            "feature": {
                "start_char": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "end_char": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    }
}
```

## Ade_corpus_v2_drug_dosage_relation

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:ade_corpus_v2/Ade_corpus_v2_drug_dosage_relation')
```

- **说明**：

```
ADE-Corpus-V2  Dataset: Adverse Drug Reaction Data.
 This is a dataset for Classification if a sentence is ADE-related (True) or not (False) and Relation Extraction between Adverse Drug Event and Drug.
 DRUG-AE.rel provides relations between drugs and adverse effects.
 DRUG-DOSE.rel provides relations between drugs and dosages.
 ADE-NEG.txt provides all sentences in the ADE corpus that DO NOT contain any drug-related adverse effects.
```

- **许可证**：无已知许可证
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 279

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "drug": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "dosage": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "indexes": {
        "drug": {
            "feature": {
                "start_char": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "end_char": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        },
        "dosage": {
            "feature": {
                "start_char": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "end_char": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    }
}
```
