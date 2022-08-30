# dane

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/dane)
- [Huggingface](https://huggingface.co/datasets/dane)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:dane')
```

- **说明**：

```
The DaNE dataset has been annotated with Named Entities for PER, ORG and LOC
by the Alexandra Institute.
It is a reannotation of the UD-DDT (Universal Dependency - Danish Dependency Treebank)
which has annotations for dependency parsing and part-of-speech (POS) tagging.
The Danish UD treebank (Johannsen et al., 2015, UD-DDT) is a conversion of
the Danish Dependency Treebank (Buch-Kromann et al. 2003) based on texts
from Parole (Britt, 1998).
```

- **许可**：CC BY-SA 4.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 565
`'train'` | 4383
`'validation'` | 564

- **特征**：

```json
{
    "sent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "tok_ids": {
        "feature": {
            "dtype": "int64",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
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
    "lemmas": {
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
            "num_classes": 17,
            "names": [
                "NUM",
                "CCONJ",
                "PRON",
                "VERB",
                "INTJ",
                "AUX",
                "ADJ",
                "PROPN",
                "PART",
                "ADV",
                "PUNCT",
                "ADP",
                "NOUN",
                "X",
                "DET",
                "SYM",
                "SCONJ"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "morph_tags": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "dep_ids": {
        "feature": {
            "dtype": "int64",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "dep_labels": {
        "feature": {
            "num_classes": 36,
            "names": [
                "parataxis",
                "mark",
                "nummod",
                "discourse",
                "compound:prt",
                "reparandum",
                "vocative",
                "list",
                "obj",
                "dep",
                "det",
                "obl:loc",
                "flat",
                "iobj",
                "cop",
                "expl",
                "obl",
                "conj",
                "nmod",
                "root",
                "acl:relcl",
                "goeswith",
                "appos",
                "fixed",
                "obl:tmod",
                "xcomp",
                "advmod",
                "nmod:poss",
                "aux",
                "ccomp",
                "amod",
                "cc",
                "advcl",
                "nsubj",
                "punct",
                "case"
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
