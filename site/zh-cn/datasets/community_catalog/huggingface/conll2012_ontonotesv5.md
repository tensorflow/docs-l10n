# conll2012_ontonotesv5

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/conll2012_ontonotesv5)
- [Huggingface](https://huggingface.co/datasets/conll2012_ontonotesv5)

## english_v4

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:conll2012_ontonotesv5/english_v4')
```

- **说明**：

```
OntoNotes v5.0 is the final version of OntoNotes corpus, and is a large-scale, multi-genre,
multilingual corpus manually annotated with syntactic, semantic and discourse information.

This dataset is the version of OntoNotes v5.0 extended and is used in the CoNLL-2012 shared task.
It includes v4 train/dev and v9 test data for English/Chinese/Arabic and corrected version v12 train/dev/test data (English only).

The source of data is the Mendeley Data repo [ontonotes-conll2012](https://data.mendeley.com/datasets/zmycy7t9h9), which seems to be as the same as the official data, but users should use this dataset on their own responsibility.

See also summaries from paperwithcode, [OntoNotes 5.0](https://paperswithcode.com/dataset/ontonotes-5-0) and [CoNLL-2012](https://paperswithcode.com/dataset/conll-2012-1)

For more detailed info of the dataset like annotation, tag set, etc., you can refer to the documents in the Mendeley repo mentioned above.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 222
`'train'` | 1940
`'validation'` | 222

- **特征**：

```json
{
    "document_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sentences": [
        {
            "part_id": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            },
            "words": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "pos_tags": {
                "feature": {
                    "num_classes": 49,
                    "names": [
                        "XX",
                        "``",
                        "$",
                        "''",
                        ",",
                        "-LRB-",
                        "-RRB-",
                        ".",
                        ":",
                        "ADD",
                        "AFX",
                        "CC",
                        "CD",
                        "DT",
                        "EX",
                        "FW",
                        "HYPH",
                        "IN",
                        "JJ",
                        "JJR",
                        "JJS",
                        "LS",
                        "MD",
                        "NFP",
                        "NN",
                        "NNP",
                        "NNPS",
                        "NNS",
                        "PDT",
                        "POS",
                        "PRP",
                        "PRP$",
                        "RB",
                        "RBR",
                        "RBS",
                        "RP",
                        "SYM",
                        "TO",
                        "UH",
                        "VB",
                        "VBD",
                        "VBG",
                        "VBN",
                        "VBP",
                        "VBZ",
                        "WDT",
                        "WP",
                        "WP$",
                        "WRB"
                    ],
                    "id": null,
                    "_type": "ClassLabel"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "parse_tree": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "predicate_lemmas": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "predicate_framenet_ids": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "word_senses": {
                "feature": {
                    "dtype": "float32",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "speaker": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "named_entities": {
                "feature": {
                    "num_classes": 37,
                    "names": [
                        "O",
                        "B-PERSON",
                        "I-PERSON",
                        "B-NORP",
                        "I-NORP",
                        "B-FAC",
                        "I-FAC",
                        "B-ORG",
                        "I-ORG",
                        "B-GPE",
                        "I-GPE",
                        "B-LOC",
                        "I-LOC",
                        "B-PRODUCT",
                        "I-PRODUCT",
                        "B-DATE",
                        "I-DATE",
                        "B-TIME",
                        "I-TIME",
                        "B-PERCENT",
                        "I-PERCENT",
                        "B-MONEY",
                        "I-MONEY",
                        "B-QUANTITY",
                        "I-QUANTITY",
                        "B-ORDINAL",
                        "I-ORDINAL",
                        "B-CARDINAL",
                        "I-CARDINAL",
                        "B-EVENT",
                        "I-EVENT",
                        "B-WORK_OF_ART",
                        "I-WORK_OF_ART",
                        "B-LAW",
                        "I-LAW",
                        "B-LANGUAGE",
                        "I-LANGUAGE"
                    ],
                    "id": null,
                    "_type": "ClassLabel"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "srl_frames": [
                {
                    "verb": {
                        "dtype": "string",
                        "id": null,
                        "_type": "Value"
                    },
                    "frames": {
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
            ],
            "coref_spans": {
                "feature": {
                    "feature": {
                        "dtype": "int32",
                        "id": null,
                        "_type": "Value"
                    },
                    "length": 3,
                    "id": null,
                    "_type": "Sequence"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            }
        }
    ]
}
```

## chinese_v4

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:conll2012_ontonotesv5/chinese_v4')
```

- **说明**：

```
OntoNotes v5.0 is the final version of OntoNotes corpus, and is a large-scale, multi-genre,
multilingual corpus manually annotated with syntactic, semantic and discourse information.

This dataset is the version of OntoNotes v5.0 extended and is used in the CoNLL-2012 shared task.
It includes v4 train/dev and v9 test data for English/Chinese/Arabic and corrected version v12 train/dev/test data (English only).

The source of data is the Mendeley Data repo [ontonotes-conll2012](https://data.mendeley.com/datasets/zmycy7t9h9), which seems to be as the same as the official data, but users should use this dataset on their own responsibility.

See also summaries from paperwithcode, [OntoNotes 5.0](https://paperswithcode.com/dataset/ontonotes-5-0) and [CoNLL-2012](https://paperswithcode.com/dataset/conll-2012-1)

For more detailed info of the dataset like annotation, tag set, etc., you can refer to the documents in the Mendeley repo mentioned above.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 166
`'train'` | 1391
`'validation'` | 172

- **特征**：

```json
{
    "document_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sentences": [
        {
            "part_id": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            },
            "words": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "pos_tags": {
                "feature": {
                    "num_classes": 36,
                    "names": [
                        "X",
                        "AD",
                        "AS",
                        "BA",
                        "CC",
                        "CD",
                        "CS",
                        "DEC",
                        "DEG",
                        "DER",
                        "DEV",
                        "DT",
                        "ETC",
                        "FW",
                        "IJ",
                        "INF",
                        "JJ",
                        "LB",
                        "LC",
                        "M",
                        "MSP",
                        "NN",
                        "NR",
                        "NT",
                        "OD",
                        "ON",
                        "P",
                        "PN",
                        "PU",
                        "SB",
                        "SP",
                        "URL",
                        "VA",
                        "VC",
                        "VE",
                        "VV"
                    ],
                    "id": null,
                    "_type": "ClassLabel"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "parse_tree": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "predicate_lemmas": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "predicate_framenet_ids": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "word_senses": {
                "feature": {
                    "dtype": "float32",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "speaker": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "named_entities": {
                "feature": {
                    "num_classes": 37,
                    "names": [
                        "O",
                        "B-PERSON",
                        "I-PERSON",
                        "B-NORP",
                        "I-NORP",
                        "B-FAC",
                        "I-FAC",
                        "B-ORG",
                        "I-ORG",
                        "B-GPE",
                        "I-GPE",
                        "B-LOC",
                        "I-LOC",
                        "B-PRODUCT",
                        "I-PRODUCT",
                        "B-DATE",
                        "I-DATE",
                        "B-TIME",
                        "I-TIME",
                        "B-PERCENT",
                        "I-PERCENT",
                        "B-MONEY",
                        "I-MONEY",
                        "B-QUANTITY",
                        "I-QUANTITY",
                        "B-ORDINAL",
                        "I-ORDINAL",
                        "B-CARDINAL",
                        "I-CARDINAL",
                        "B-EVENT",
                        "I-EVENT",
                        "B-WORK_OF_ART",
                        "I-WORK_OF_ART",
                        "B-LAW",
                        "I-LAW",
                        "B-LANGUAGE",
                        "I-LANGUAGE"
                    ],
                    "id": null,
                    "_type": "ClassLabel"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "srl_frames": [
                {
                    "verb": {
                        "dtype": "string",
                        "id": null,
                        "_type": "Value"
                    },
                    "frames": {
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
            ],
            "coref_spans": {
                "feature": {
                    "feature": {
                        "dtype": "int32",
                        "id": null,
                        "_type": "Value"
                    },
                    "length": 3,
                    "id": null,
                    "_type": "Sequence"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            }
        }
    ]
}
```

## arabic_v4

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:conll2012_ontonotesv5/arabic_v4')
```

- **说明**：

```
OntoNotes v5.0 is the final version of OntoNotes corpus, and is a large-scale, multi-genre,
multilingual corpus manually annotated with syntactic, semantic and discourse information.

This dataset is the version of OntoNotes v5.0 extended and is used in the CoNLL-2012 shared task.
It includes v4 train/dev and v9 test data for English/Chinese/Arabic and corrected version v12 train/dev/test data (English only).

The source of data is the Mendeley Data repo [ontonotes-conll2012](https://data.mendeley.com/datasets/zmycy7t9h9), which seems to be as the same as the official data, but users should use this dataset on their own responsibility.

See also summaries from paperwithcode, [OntoNotes 5.0](https://paperswithcode.com/dataset/ontonotes-5-0) and [CoNLL-2012](https://paperswithcode.com/dataset/conll-2012-1)

For more detailed info of the dataset like annotation, tag set, etc., you can refer to the documents in the Mendeley repo mentioned above.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 44
`'train'` | 359
`'validation'` | 44

- **特征**：

```json
{
    "document_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sentences": [
        {
            "part_id": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            },
            "words": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "pos_tags": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "parse_tree": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "predicate_lemmas": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "predicate_framenet_ids": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "word_senses": {
                "feature": {
                    "dtype": "float32",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "speaker": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "named_entities": {
                "feature": {
                    "num_classes": 37,
                    "names": [
                        "O",
                        "B-PERSON",
                        "I-PERSON",
                        "B-NORP",
                        "I-NORP",
                        "B-FAC",
                        "I-FAC",
                        "B-ORG",
                        "I-ORG",
                        "B-GPE",
                        "I-GPE",
                        "B-LOC",
                        "I-LOC",
                        "B-PRODUCT",
                        "I-PRODUCT",
                        "B-DATE",
                        "I-DATE",
                        "B-TIME",
                        "I-TIME",
                        "B-PERCENT",
                        "I-PERCENT",
                        "B-MONEY",
                        "I-MONEY",
                        "B-QUANTITY",
                        "I-QUANTITY",
                        "B-ORDINAL",
                        "I-ORDINAL",
                        "B-CARDINAL",
                        "I-CARDINAL",
                        "B-EVENT",
                        "I-EVENT",
                        "B-WORK_OF_ART",
                        "I-WORK_OF_ART",
                        "B-LAW",
                        "I-LAW",
                        "B-LANGUAGE",
                        "I-LANGUAGE"
                    ],
                    "id": null,
                    "_type": "ClassLabel"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "srl_frames": [
                {
                    "verb": {
                        "dtype": "string",
                        "id": null,
                        "_type": "Value"
                    },
                    "frames": {
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
            ],
            "coref_spans": {
                "feature": {
                    "feature": {
                        "dtype": "int32",
                        "id": null,
                        "_type": "Value"
                    },
                    "length": 3,
                    "id": null,
                    "_type": "Sequence"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            }
        }
    ]
}
```

## english_v12

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:conll2012_ontonotesv5/english_v12')
```

- **说明**：

```
OntoNotes v5.0 is the final version of OntoNotes corpus, and is a large-scale, multi-genre,
multilingual corpus manually annotated with syntactic, semantic and discourse information.

This dataset is the version of OntoNotes v5.0 extended and is used in the CoNLL-2012 shared task.
It includes v4 train/dev and v9 test data for English/Chinese/Arabic and corrected version v12 train/dev/test data (English only).

The source of data is the Mendeley Data repo [ontonotes-conll2012](https://data.mendeley.com/datasets/zmycy7t9h9), which seems to be as the same as the official data, but users should use this dataset on their own responsibility.

See also summaries from paperwithcode, [OntoNotes 5.0](https://paperswithcode.com/dataset/ontonotes-5-0) and [CoNLL-2012](https://paperswithcode.com/dataset/conll-2012-1)

For more detailed info of the dataset like annotation, tag set, etc., you can refer to the documents in the Mendeley repo mentioned above.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1200
`'train'` | 10539
`'validation'` | 1370

- **特征**：

```json
{
    "document_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sentences": [
        {
            "part_id": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            },
            "words": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "pos_tags": {
                "feature": {
                    "num_classes": 51,
                    "names": [
                        "XX",
                        "``",
                        "$",
                        "''",
                        "*",
                        ",",
                        "-LRB-",
                        "-RRB-",
                        ".",
                        ":",
                        "ADD",
                        "AFX",
                        "CC",
                        "CD",
                        "DT",
                        "EX",
                        "FW",
                        "HYPH",
                        "IN",
                        "JJ",
                        "JJR",
                        "JJS",
                        "LS",
                        "MD",
                        "NFP",
                        "NN",
                        "NNP",
                        "NNPS",
                        "NNS",
                        "PDT",
                        "POS",
                        "PRP",
                        "PRP$",
                        "RB",
                        "RBR",
                        "RBS",
                        "RP",
                        "SYM",
                        "TO",
                        "UH",
                        "VB",
                        "VBD",
                        "VBG",
                        "VBN",
                        "VBP",
                        "VBZ",
                        "VERB",
                        "WDT",
                        "WP",
                        "WP$",
                        "WRB"
                    ],
                    "id": null,
                    "_type": "ClassLabel"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "parse_tree": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "predicate_lemmas": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "predicate_framenet_ids": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "word_senses": {
                "feature": {
                    "dtype": "float32",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "speaker": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "named_entities": {
                "feature": {
                    "num_classes": 37,
                    "names": [
                        "O",
                        "B-PERSON",
                        "I-PERSON",
                        "B-NORP",
                        "I-NORP",
                        "B-FAC",
                        "I-FAC",
                        "B-ORG",
                        "I-ORG",
                        "B-GPE",
                        "I-GPE",
                        "B-LOC",
                        "I-LOC",
                        "B-PRODUCT",
                        "I-PRODUCT",
                        "B-DATE",
                        "I-DATE",
                        "B-TIME",
                        "I-TIME",
                        "B-PERCENT",
                        "I-PERCENT",
                        "B-MONEY",
                        "I-MONEY",
                        "B-QUANTITY",
                        "I-QUANTITY",
                        "B-ORDINAL",
                        "I-ORDINAL",
                        "B-CARDINAL",
                        "I-CARDINAL",
                        "B-EVENT",
                        "I-EVENT",
                        "B-WORK_OF_ART",
                        "I-WORK_OF_ART",
                        "B-LAW",
                        "I-LAW",
                        "B-LANGUAGE",
                        "I-LANGUAGE"
                    ],
                    "id": null,
                    "_type": "ClassLabel"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "srl_frames": [
                {
                    "verb": {
                        "dtype": "string",
                        "id": null,
                        "_type": "Value"
                    },
                    "frames": {
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
            ],
            "coref_spans": {
                "feature": {
                    "feature": {
                        "dtype": "int32",
                        "id": null,
                        "_type": "Value"
                    },
                    "length": 3,
                    "id": null,
                    "_type": "Sequence"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            }
        }
    ]
}
```
