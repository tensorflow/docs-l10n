# few_rel

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/few_rel)
- [Huggingface](https://huggingface.co/datasets/few_rel)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:few_rel')
```

- **说明**：

```
FewRel is a large-scale few-shot relation extraction dataset, which contains more than one hundred relations and tens of thousands of annotated instances cross different domains.
```

- **许可**：https://raw.githubusercontent.com/thunlp/FewRel/master/LICENSE
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'pubmed_unsupervised'` | 2500
`'train_wiki'` | 44800
`'val_nyt'` | 2500
`'val_pubmed'` | 1000
`'val_semeval'` | 8851
`'val_wiki'` | 11200

- **特征**：

```json
{
    "relation": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "head": {
        "text": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "type": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "indices": {
            "feature": {
                "feature": {
                    "dtype": "int64",
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
    },
    "tail": {
        "text": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "type": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "indices": {
            "feature": {
                "feature": {
                    "dtype": "int64",
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
    },
    "names": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## pid2name

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:few_rel/pid2name')
```

- **说明**：

```
FewRel is a large-scale few-shot relation extraction dataset, which contains more than one hundred relations and tens of thousands of annotated instances cross different domains.
```

- **许可**：https://raw.githubusercontent.com/thunlp/FewRel/master/LICENSE
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'pid2name'` | 744

- **特征**：

```json
{
    "relation": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "names": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```
