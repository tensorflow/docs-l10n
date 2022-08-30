# curiosity_dialogs

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/curiosity_dialogs)
- [Huggingface](https://huggingface.co/datasets/curiosity_dialogs)

## curiosity_dialogs

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:curiosity_dialogs/curiosity_dialogs')
```

- **说明**：

```
This dataset contains 14K dialogs (181K utterances) where users and assistants converse about geographic topics like
geopolitical entities and locations. This dataset is annotated with pre-existing user knowledge, message-level dialog
acts, grounding to Wikipedia, and user reactions to messages.
```

- **许可**：https://github.com/facebookresearch/curiosity/blob/master/LICENSE
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1287
`'test_zero'` | 1187
`'train'` | 10287
`'val'` | 1287

- **特征**：

```json
{
    "messages": {
        "feature": {
            "message": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "liked": {
                "num_classes": 2,
                "names": [
                    "False",
                    "True"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "sender": {
                "num_classes": 2,
                "names": [
                    "user",
                    "assistant"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "facts": {
                "feature": {
                    "fid": {
                        "dtype": "int32",
                        "id": null,
                        "_type": "Value"
                    },
                    "used": {
                        "num_classes": 2,
                        "names": [
                            "False",
                            "True"
                        ],
                        "names_file": null,
                        "id": null,
                        "_type": "ClassLabel"
                    },
                    "source": {
                        "num_classes": 3,
                        "names": [
                            "section",
                            "known",
                            "random"
                        ],
                        "names_file": null,
                        "id": null,
                        "_type": "ClassLabel"
                    }
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "message_id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "dialog_acts": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "known_entities": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "focus_entity": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "dialog_id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inferred_steps": {
        "num_classes": 2,
        "names": [
            "False",
            "True"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "created_time": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "aspects": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "first_aspect": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "second_aspect": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "shuffle_facts": {
        "num_classes": 2,
        "names": [
            "False",
            "True"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "related_entities": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "tag": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "user_id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "assistant_id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "is_annotated": {
        "num_classes": 2,
        "names": [
            "False",
            "True"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "user_dialog_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "user_other_agent_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "assistant_dialog_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "assistant_other_agent_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "reported": {
        "num_classes": 2,
        "names": [
            "False",
            "True"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "annotated": {
        "num_classes": 2,
        "names": [
            "False",
            "True"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
