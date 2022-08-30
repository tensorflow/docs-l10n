# deal_or_no_dialog

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/deal_or_no_dialog)
- [Huggingface](https://huggingface.co/datasets/deal_or_no_dialog)

## dialogues

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:deal_or_no_dialog/dialogues')
```

- **Description**:

```
A large dataset of human-human negotiations on a multi-issue bargaining task, where agents who cannot observe each other’s reward functions must reach anagreement (o a deal) via natural language dialogue.
```

- **许可**：该项目在 CC-by-NC 下许可
- **Version**: 1.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1052
`'train'` | 10095
`'validation'` | 1087

- **特征**：

```json
{
    "input": {
        "feature": {
            "count": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            },
            "value": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "dialogue": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "output": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "partner_input": {
        "feature": {
            "count": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            },
            "value": {
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
```

## self_play

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:deal_or_no_dialog/self_play')
```

- **Description**:

```
A large dataset of human-human negotiations on a multi-issue bargaining task, where agents who cannot observe each other’s reward functions must reach anagreement (o a deal) via natural language dialogue.
```

- **许可**：该项目在 CC-by-NC 下许可
- **Version**: 1.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 8172

- **特征**：

```json
{
    "input": {
        "feature": {
            "count": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            },
            "value": {
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
```
