# blended_skill_talk

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/blended_skill_talk)
- [Huggingface](https://huggingface.co/datasets/blended_skill_talk)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:blended_skill_talk')
```

- **说明**：

```
A dataset of 7k conversations explicitly designed to exhibit multiple conversation modes: displaying personality, having empathy, and demonstrating knowledge.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 980
`'train'` | 4819
`'validation'` | 1009

- **特征**：

```json
{
    "personas": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "additional_context": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "previous_utterance": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "context": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "free_messages": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "guided_messages": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "suggestions": {
        "feature": {
            "convai2": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "empathetic_dialogues": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "wizard_of_wikipedia": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "guided_chosen_suggestions": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "label_candidates": {
        "feature": {
            "feature": {
                "dtype": "string",
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
    }
}
```
