# ethos

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/ethos)
- [Huggingface](https://huggingface.co/datasets/ethos)

## binary

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:ethos/binary')
```

- **说明**：

```

```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 998

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
            "no_hate_speech",
            "hate_speech"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## multilabel

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:ethos/multilabel')
```

- **说明**：

```

```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 433

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "violence": {
        "num_classes": 2,
        "names": [
            "not_violent",
            "violent"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "directed_vs_generalized": {
        "num_classes": 2,
        "names": [
            "generalied",
            "directed"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "gender": {
        "num_classes": 2,
        "names": [
            "false",
            "true"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "race": {
        "num_classes": 2,
        "names": [
            "false",
            "true"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "national_origin": {
        "num_classes": 2,
        "names": [
            "false",
            "true"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "disability": {
        "num_classes": 2,
        "names": [
            "false",
            "true"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "religion": {
        "num_classes": 2,
        "names": [
            "false",
            "true"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "sexual_orientation": {
        "num_classes": 2,
        "names": [
            "false",
            "true"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
