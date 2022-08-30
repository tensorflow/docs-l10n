# germaner

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/germaner)
- [Huggingface](https://huggingface.co/datasets/germaner)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:germaner')
```

- **说明**：

```
GermaNER is a freely available statistical German Named Entity Tagger based on conditional random fields(CRF). The tagger is trained and evaluated on the NoSta-D Named Entity dataset, which was used in the GermEval 2014 for named entity recognition. The tagger comes close to the performance of the best (proprietary) system in the competition with 77% F-measure (this is the latest result; the one reported in the paper is 76%) test set performance on the four standard NER classes (PERson, LOCation, ORGanisation and OTHer).

We describe a range of features and their influence on German NER classification and provide a comparative evaluation and some analysis of the results. The software components, the training data and all data used for feature generation are distributed under permissive licenses, thus this tagger can be used in academic and commercial settings without restrictions or fees. The tagger is available as a command-line tool and as an Apache UIMA component.
```

- **许可**：无已知许可
- **版本**：0.9.1
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 26200

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
                "B-LOC",
                "B-ORG",
                "B-OTH",
                "B-PER",
                "I-LOC",
                "I-ORG",
                "I-OTH",
                "I-PER",
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
