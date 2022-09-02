# id_nergrit_corpus

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/id_nergrit_corpus)
- [Huggingface](https://huggingface.co/datasets/id_nergrit_corpus)

## ner

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:id_nergrit_corpus/ner')
```

- **说明**：

```
Nergrit Corpus is a dataset collection for Indonesian Named Entity Recognition, Statement Extraction, and Sentiment
Analysis. id_nergrit_corpus is the Named Entity Recognition of this dataset collection which contains 18 entities as
follow:
    'CRD': Cardinal
    'DAT': Date
    'EVT': Event
    'FAC': Facility
    'GPE': Geopolitical Entity
    'LAW': Law Entity (such as Undang-Undang)
    'LOC': Location
    'MON': Money
    'NOR': Political Organization
    'ORD': Ordinal
    'ORG': Organization
    'PER': Person
    'PRC': Percent
    'PRD': Product
    'QTY': Quantity
    'REG': Religion
    'TIM': Time
    'WOA': Work of Art
    'LAN': Language
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2399
`'train'` | 12532
`'validation'` | 2521

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
    "ner_tags": {
        "feature": {
            "num_classes": 39,
            "names": [
                "B-CRD",
                "B-DAT",
                "B-EVT",
                "B-FAC",
                "B-GPE",
                "B-LAN",
                "B-LAW",
                "B-LOC",
                "B-MON",
                "B-NOR",
                "B-ORD",
                "B-ORG",
                "B-PER",
                "B-PRC",
                "B-PRD",
                "B-QTY",
                "B-REG",
                "B-TIM",
                "B-WOA",
                "I-CRD",
                "I-DAT",
                "I-EVT",
                "I-FAC",
                "I-GPE",
                "I-LAN",
                "I-LAW",
                "I-LOC",
                "I-MON",
                "I-NOR",
                "I-ORD",
                "I-ORG",
                "I-PER",
                "I-PRC",
                "I-PRD",
                "I-QTY",
                "I-REG",
                "I-TIM",
                "I-WOA",
                "O"
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

## sentiment

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:id_nergrit_corpus/sentiment')
```

- **说明**：

```
Nergrit Corpus is a dataset collection for Indonesian Named Entity Recognition, Statement Extraction, and Sentiment
Analysis. id_nergrit_corpus is the Named Entity Recognition of this dataset collection which contains 18 entities as
follow:
    'CRD': Cardinal
    'DAT': Date
    'EVT': Event
    'FAC': Facility
    'GPE': Geopolitical Entity
    'LAW': Law Entity (such as Undang-Undang)
    'LOC': Location
    'MON': Money
    'NOR': Political Organization
    'ORD': Ordinal
    'ORG': Organization
    'PER': Person
    'PRC': Percent
    'PRD': Product
    'QTY': Quantity
    'REG': Religion
    'TIM': Time
    'WOA': Work of Art
    'LAN': Language
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2317
`'train'` | 7485
`'validation'` | 782

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
    "ner_tags": {
        "feature": {
            "num_classes": 7,
            "names": [
                "B-NEG",
                "B-NET",
                "B-POS",
                "I-NEG",
                "I-NET",
                "I-POS",
                "O"
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

## statement

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:id_nergrit_corpus/statement')
```

- **说明**：

```
Nergrit Corpus is a dataset collection for Indonesian Named Entity Recognition, Statement Extraction, and Sentiment
Analysis. id_nergrit_corpus is the Named Entity Recognition of this dataset collection which contains 18 entities as
follow:
    'CRD': Cardinal
    'DAT': Date
    'EVT': Event
    'FAC': Facility
    'GPE': Geopolitical Entity
    'LAW': Law Entity (such as Undang-Undang)
    'LOC': Location
    'MON': Money
    'NOR': Political Organization
    'ORD': Ordinal
    'ORG': Organization
    'PER': Person
    'PRC': Percent
    'PRD': Product
    'QTY': Quantity
    'REG': Religion
    'TIM': Time
    'WOA': Work of Art
    'LAN': Language
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 335
`'train'` | 2405
`'validation'` | 176

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
    "ner_tags": {
        "feature": {
            "num_classes": 9,
            "names": [
                "B-BREL",
                "B-FREL",
                "B-STAT",
                "B-WHO",
                "I-BREL",
                "I-FREL",
                "I-STAT",
                "I-WHO",
                "O"
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
