# hotpot_qa

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/hotpot_qa)
- [Huggingface](https://huggingface.co/datasets/hotpot_qa)

## distractor

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hotpot_qa/distractor')
```

- **说明**：

```
HotpotQA is a new dataset with 113k  Wikipedia-based question-answer  pairs with  four  key  features:  (1)  the  questions  require finding and reasoning over multiple supporting  documents  to  answer;  (2)  the  questions  are  diverse  and  not  constrained  to  any pre-existing  knowledge  bases  or  knowledge schemas;  (3)  we  provide  sentence-level  supporting facts required for reasoning, allowingQA systems to reason with strong supervisionand explain the predictions; (4) we offer a new type  of  factoid  comparison  questions  to  testQA  systems’  ability  to  extract  relevant  facts and perform necessary comparison.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 90447
`'validation'` | 7405

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
    "answer": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "type": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "level": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "supporting_facts": {
        "feature": {
            "title": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "sent_id": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "context": {
        "feature": {
            "title": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "sentences": {
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
}
```

## fullwiki

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hotpot_qa/fullwiki')
```

- **说明**：

```
HotpotQA is a new dataset with 113k  Wikipedia-based question-answer  pairs with  four  key  features:  (1)  the  questions  require finding and reasoning over multiple supporting  documents  to  answer;  (2)  the  questions  are  diverse  and  not  constrained  to  any pre-existing  knowledge  bases  or  knowledge schemas;  (3)  we  provide  sentence-level  supporting facts required for reasoning, allowingQA systems to reason with strong supervisionand explain the predictions; (4) we offer a new type  of  factoid  comparison  questions  to  testQA  systems’  ability  to  extract  relevant  facts and perform necessary comparison.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 7405
`'train'` | 90447
`'validation'` | 7405

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
    "answer": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "type": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "level": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "supporting_facts": {
        "feature": {
            "title": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "sent_id": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "context": {
        "feature": {
            "title": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "sentences": {
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
}
```
