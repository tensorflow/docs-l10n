# elkarhizketak

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/elkarhizketak)
- [Huggingface](https://huggingface.co/datasets/elkarhizketak)

## plain_text

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:elkarhizketak/plain_text')
```

- **说明**：

```
ElkarHizketak is a low resource conversational Question Answering
(QA) dataset in Basque created by Basque speaker volunteers. The
dataset contains close to 400 dialogues and more than 1600 question
and answers, and its small size presents a realistic low-resource
scenario for conversational QA systems. The dataset is built on top of
Wikipedia sections about popular people and organizations. The
dialogues involve two crowd workers: (1) a student ask questions after
reading a small introduction about the person, but without seeing the
section text; and (2) a teacher answers the questions selecting a span
of text of the section.
```

- **许可**：Creative Commons Attribution-ShareAlike 4.0 International Public License (CC BY-SA 4.0)
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 38
`'train'` | 301
`'validation'` | 38

- **特征**：

```json
{
    "dialogue_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "wikipedia_page_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "background": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "section_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "context": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "turn_ids": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "questions": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "yesnos": {
        "feature": {
            "num_classes": 3,
            "names": [
                "y",
                "n",
                "x"
            ],
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answers": {
        "feature": {
            "texts": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "answer_starts": {
                "feature": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "input_texts": {
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
    "orig_answers": {
        "texts": {
            "feature": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        },
        "answer_starts": {
            "feature": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    }
}
```
