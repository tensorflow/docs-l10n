# conll2000

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/conll2000)
- [Huggingface](https://huggingface.co/datasets/conll2000)

## conll2000

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:conll2000/conll2000')
```

- **说明**：

```
Text chunking consists of dividing a text in syntactically correlated parts of words. For example, the sentence
 He reckons the current account deficit will narrow to only # 1.8 billion in September . can be divided as follows:
[NP He ] [VP reckons ] [NP the current account deficit ] [VP will narrow ] [PP to ] [NP only # 1.8 billion ]
[PP in ] [NP September ] .

Text chunking is an intermediate step towards full parsing. It was the shared task for CoNLL-2000. Training and test
data for this task is available. This data consists of the same partitions of the Wall Street Journal corpus (WSJ)
as the widely used data for noun phrase chunking: sections 15-18 as training data (211727 tokens) and section 20 as
test data (47377 tokens). The annotation of the data has been derived from the WSJ corpus by a program written by
Sabine Buchholz from Tilburg University, The Netherlands.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2013
`'train'` | 8937

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
            "num_classes": 44,
            "names": [
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
                "MD",
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
    }
}
```
