# docred

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/docred)
- [Huggingface](https://huggingface.co/datasets/docred)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:docred')
```

- **说明**：

```
Multiple entities in a document generally exhibit complex inter-sentence relations, and cannot be well handled by existing relation extraction (RE) methods that typically focus on extracting intra-sentence relations for single entity pairs. In order to accelerate the research on document-level RE, we introduce DocRED, a new dataset constructed from Wikipedia and Wikidata with three features:
    - DocRED annotates both named entities and relations, and is the largest human-annotated dataset for document-level RE from plain text.
    - DocRED requires reading multiple sentences in a document to extract entities and infer their relations by synthesizing all information of the document.
    - Along with the human-annotated data, we also offer large-scale distantly supervised data, which enables DocRED to be adopted for both supervised and weakly supervised scenarios.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train_annotated'` | 3053
`'train_distant'` | 101873
`'validation'` | 998

- **特征**：

```json
{
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sents": {
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
    },
    "vertexSet": [
        [
            {
                "name": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "sent_id": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "pos": {
                    "feature": {
                        "dtype": "int32",
                        "id": null,
                        "_type": "Value"
                    },
                    "length": -1,
                    "id": null,
                    "_type": "Sequence"
                },
                "type": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            }
        ]
    ],
    "labels": {
        "feature": {
            "head": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            },
            "tail": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            },
            "relation_id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "relation_text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "evidence": {
                "feature": {
                    "dtype": "int32",
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
