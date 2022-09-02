# ambig_qa

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/ambig_qa)
- [Huggingface](https://huggingface.co/datasets/ambig_qa)

## light

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:ambig_qa/light')
```

- **说明**：

```
AmbigNQ, a dataset covering 14,042 questions from NQ-open, an existing open-domain QA benchmark. We find that over half of the questions in NQ-open are ambiguous. The types of ambiguity are diverse and sometimes subtle, many of which are only apparent after examining evidence provided by a very large text corpus.  AMBIGNQ, a dataset with
14,042 annotations on NQ-OPEN questions containing diverse types of ambiguity.
We provide two distributions of our new dataset AmbigNQ: a full version with all annotation metadata and a light version with only inputs and outputs.
```

- **许可**：CC BY-SA 3.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 10036
`'validation'` | 2002

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotations": {
        "feature": {
            "type": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "answer": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "qaPairs": {
                "feature": {
                    "question": {
                        "dtype": "string",
                        "id": null,
                        "_type": "Value"
                    },
                    "answer": {
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
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## full

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:ambig_qa/full')
```

- **说明**：

```
AmbigNQ, a dataset covering 14,042 questions from NQ-open, an existing open-domain QA benchmark. We find that over half of the questions in NQ-open are ambiguous. The types of ambiguity are diverse and sometimes subtle, many of which are only apparent after examining evidence provided by a very large text corpus.  AMBIGNQ, a dataset with
14,042 annotations on NQ-OPEN questions containing diverse types of ambiguity.
We provide two distributions of our new dataset AmbigNQ: a full version with all annotation metadata and a light version with only inputs and outputs.
```

- **许可**：CC BY-SA 3.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 10036
`'validation'` | 2002

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotations": {
        "feature": {
            "type": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "answer": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "qaPairs": {
                "feature": {
                    "question": {
                        "dtype": "string",
                        "id": null,
                        "_type": "Value"
                    },
                    "answer": {
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
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "viewed_doc_titles": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "used_queries": {
        "feature": {
            "query": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "results": {
                "feature": {
                    "title": {
                        "dtype": "string",
                        "id": null,
                        "_type": "Value"
                    },
                    "snippet": {
                        "dtype": "string",
                        "id": null,
                        "_type": "Value"
                    }
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
    "nq_answer": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "nq_doc_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
