# germeval_14

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/germeval_14)
- [Huggingface](https://huggingface.co/datasets/germeval_14)

## germeval_14

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:germeval_14/germeval_14')
```

- **说明**：

```
The GermEval 2014 NER Shared Task builds on a new dataset with German Named Entity annotation with the following properties:    - The data was sampled from German Wikipedia and News Corpora as a collection of citations.    - The dataset covers over 31,000 sentences corresponding to over 590,000 tokens.    - The NER annotation uses the NoSta-D guidelines, which extend the Tübingen Treebank guidelines,      using four main NER categories with sub-structure, and annotating embeddings among NEs      such as [ORG FC Kickers [LOC Darmstadt]].
```

- **许可**：无已知许可
- **版本**：2.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 5100
`'train'` | 24000
`'validation'` | 2200

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source": {
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
            "num_classes": 25,
            "names": [
                "O",
                "B-LOC",
                "I-LOC",
                "B-LOCderiv",
                "I-LOCderiv",
                "B-LOCpart",
                "I-LOCpart",
                "B-ORG",
                "I-ORG",
                "B-ORGderiv",
                "I-ORGderiv",
                "B-ORGpart",
                "I-ORGpart",
                "B-OTH",
                "I-OTH",
                "B-OTHderiv",
                "I-OTHderiv",
                "B-OTHpart",
                "I-OTHpart",
                "B-PER",
                "I-PER",
                "B-PERderiv",
                "I-PERderiv",
                "B-PERpart",
                "I-PERpart"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "nested_ner_tags": {
        "feature": {
            "num_classes": 25,
            "names": [
                "O",
                "B-LOC",
                "I-LOC",
                "B-LOCderiv",
                "I-LOCderiv",
                "B-LOCpart",
                "I-LOCpart",
                "B-ORG",
                "I-ORG",
                "B-ORGderiv",
                "I-ORGderiv",
                "B-ORGpart",
                "I-ORGpart",
                "B-OTH",
                "I-OTH",
                "B-OTHderiv",
                "I-OTHderiv",
                "B-OTHpart",
                "I-OTHpart",
                "B-PER",
                "I-PER",
                "B-PERderiv",
                "I-PERderiv",
                "B-PERpart",
                "I-PERpart"
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
