# ascent_kb

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/ascent_kb)
- [Huggingface](https://huggingface.co/datasets/ascent_kb)

## canonical

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:ascent_kb/canonical')
```

- **Description**:

```
This dataset contains 8.9M commonsense assertions extracted by the Ascent pipeline (https://ascent.mpi-inf.mpg.de/).
```

- **许可**：The Creative Commons Attribution 4.0 International License. https://creativecommons.org/licenses/by/4.0/
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 8904060

- **特征**：

```json
{
    "arg1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "rel": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "arg2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "support": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "facets": [
        {
            "value": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "support": {
                "dtype": "int64",
                "id": null,
                "_type": "Value"
            }
        }
    ],
    "source_sentences": [
        {
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "source": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            }
        }
    ]
}
```

## open

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:ascent_kb/open')
```

- **Description**:

```
This dataset contains 8.9M commonsense assertions extracted by the Ascent pipeline (https://ascent.mpi-inf.mpg.de/).
```

- **许可**：The Creative Commons Attribution 4.0 International License. https://creativecommons.org/licenses/by/4.0/
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 8904060

- **特征**：

```json
{
    "subject": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "predicate": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "object": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "support": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "facets": [
        {
            "value": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "support": {
                "dtype": "int64",
                "id": null,
                "_type": "Value"
            }
        }
    ],
    "source_sentences": [
        {
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "source": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            }
        }
    ]
}
```
