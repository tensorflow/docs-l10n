# ett

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/ett)
- [Huggingface](https://huggingface.co/datasets/ett)

## h1

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:ett/h1')
```

- **Description**:

```
The data of Electricity Transformers from two separated counties
in China collected for two years at hourly and 15-min frequencies.
Each data point consists of the target value "oil temperature" and
6 power load features. The train/val/test is 12/4/4 months.
```

- **许可**：Creative Commons Attribution 4.0 International License。https://creativecommons.org/licenses/by/4.0/
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 240
`'train'` | 1
`'validation'` | 120

- **特征**：

```json
{
    "start": {
        "dtype": "timestamp[s]",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "feature": {
            "dtype": "float32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "feat_static_cat": {
        "feature": {
            "dtype": "uint64",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "feat_dynamic_real": {
        "feature": {
            "feature": {
                "dtype": "float32",
                "id": null,
                "_type": "Value"
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "item_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## h2

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:ett/h2')
```

- **Description**:

```
The data of Electricity Transformers from two separated counties
in China collected for two years at hourly and 15-min frequencies.
Each data point consists of the target value "oil temperature" and
6 power load features. The train/val/test is 12/4/4 months.
```

- **许可**：Creative Commons Attribution 4.0 International License。https://creativecommons.org/licenses/by/4.0/
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 240
`'train'` | 1
`'validation'` | 120

- **特征**：

```json
{
    "start": {
        "dtype": "timestamp[s]",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "feature": {
            "dtype": "float32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "feat_static_cat": {
        "feature": {
            "dtype": "uint64",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "feat_dynamic_real": {
        "feature": {
            "feature": {
                "dtype": "float32",
                "id": null,
                "_type": "Value"
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "item_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## m1

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:ett/m1')
```

- **Description**:

```
The data of Electricity Transformers from two separated counties
in China collected for two years at hourly and 15-min frequencies.
Each data point consists of the target value "oil temperature" and
6 power load features. The train/val/test is 12/4/4 months.
```

- **许可**：Creative Commons Attribution 4.0 International License。https://creativecommons.org/licenses/by/4.0/
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 0
`'train'` | 1
`'validation'` | 0

- **特征**：

```json
{
    "start": {
        "dtype": "timestamp[s]",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "feature": {
            "dtype": "float32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "feat_static_cat": {
        "feature": {
            "dtype": "uint64",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "feat_dynamic_real": {
        "feature": {
            "feature": {
                "dtype": "float32",
                "id": null,
                "_type": "Value"
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "item_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## m2

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:ett/m2')
```

- **Description**:

```
The data of Electricity Transformers from two separated counties
in China collected for two years at hourly and 15-min frequencies.
Each data point consists of the target value "oil temperature" and
6 power load features. The train/val/test is 12/4/4 months.
```

- **许可**：Creative Commons Attribution 4.0 International License。https://creativecommons.org/licenses/by/4.0/
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 0
`'train'` | 1
`'validation'` | 0

- **特征**：

```json
{
    "start": {
        "dtype": "timestamp[s]",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "feature": {
            "dtype": "float32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "feat_static_cat": {
        "feature": {
            "dtype": "uint64",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "feat_dynamic_real": {
        "feature": {
            "feature": {
                "dtype": "float32",
                "id": null,
                "_type": "Value"
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "item_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
