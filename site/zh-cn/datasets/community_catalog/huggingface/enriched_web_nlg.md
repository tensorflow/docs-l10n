# enriched_web_nlg

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/enriched_web_nlg)
- [Huggingface](https://huggingface.co/datasets/enriched_web_nlg)

## en

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:enriched_web_nlg/en')
```

- **说明**：

```
WebNLG is a valuable resource and benchmark for the Natural Language Generation (NLG) community. However, as other NLG benchmarks, it only consists of a collection of parallel raw representations and their corresponding textual realizations. This work aimed to provide intermediate representations of the data for the development and evaluation of popular tasks in the NLG pipeline architecture (Reiter and Dale, 2000), such as Discourse Ordering, Lexicalization, Aggregation and Referring Expression Generation.
```

- **许可**：CC Attribution-Noncommercial-Share Alike 4.0 International
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'dev'` | 872
`'test'` | 1862
`'train'` | 6940

- **特征**：

```json
{
    "category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "size": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "eid": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "original_triple_sets": {
        "feature": {
            "otriple_set": {
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
    "modified_triple_sets": {
        "feature": {
            "mtriple_set": {
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
    "shape": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "shape_type": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "lex": {
        "feature": {
            "comment": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "lid": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "template": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "sorted_triple_sets": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "lexicalization": {
                "dtype": "string",
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

## de

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:enriched_web_nlg/de')
```

- **说明**：

```
WebNLG is a valuable resource and benchmark for the Natural Language Generation (NLG) community. However, as other NLG benchmarks, it only consists of a collection of parallel raw representations and their corresponding textual realizations. This work aimed to provide intermediate representations of the data for the development and evaluation of popular tasks in the NLG pipeline architecture (Reiter and Dale, 2000), such as Discourse Ordering, Lexicalization, Aggregation and Referring Expression Generation.
```

- **许可**：CC Attribution-Noncommercial-Share Alike 4.0 International
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'dev'` | 872
`'train'` | 6940

- **特征**：

```json
{
    "category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "size": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "eid": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "original_triple_sets": {
        "feature": {
            "otriple_set": {
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
    "modified_triple_sets": {
        "feature": {
            "mtriple_set": {
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
    "shape": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "shape_type": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "lex": {
        "feature": {
            "comment": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "lid": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "template": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "sorted_triple_sets": {
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
