# conllpp

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/conllpp)
- [Huggingface](https://huggingface.co/datasets/conllpp)

## conllpp

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:conllpp/conllpp')
```

- **说明**：

```
CoNLLpp is a corrected version of the CoNLL2003 NER dataset where labels of 5.38% of the sentences in the test set
have been manually corrected. The training set and development set are included for completeness.
For more details see https://www.aclweb.org/anthology/D19-1519/ and https://github.com/ZihanWangKi/CrossWeigh
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 3453
`'train'` | 14041
`'validation'` | 3250

- **特征**：

```json
{
    "id": {
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
    "pos_tags": {
        "feature": {
            "num_classes": 47,
            "names": [
                "\"",
                "''",
                "#",
                "$",
                "(",
                ")",
                ",",
                ".",
                ":",
                "``",
                "CC",
                "CD",
                "DT",
                "EX",
                "FW",
                "IN",
                "JJ",
                "JJR",
                "JJS",
                "LS",
                "MD",
                "NN",
                "NNP",
                "NNPS",
                "NNS",
                "NN|SYM",
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
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "chunk_tags": {
        "feature": {
            "num_classes": 23,
            "names": [
                "O",
                "B-ADJP",
                "I-ADJP",
                "B-ADVP",
                "I-ADVP",
                "B-CONJP",
                "I-CONJP",
                "B-INTJ",
                "I-INTJ",
                "B-LST",
                "I-LST",
                "B-NP",
                "I-NP",
                "B-PP",
                "I-PP",
                "B-PRT",
                "I-PRT",
                "B-SBAR",
                "I-SBAR",
                "B-UCP",
                "I-UCP",
                "B-VP",
                "I-VP"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "ner_tags": {
        "feature": {
            "num_classes": 9,
            "names": [
                "O",
                "B-PER",
                "I-PER",
                "B-ORG",
                "I-ORG",
                "B-LOC",
                "I-LOC",
                "B-MISC",
                "I-MISC"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```
