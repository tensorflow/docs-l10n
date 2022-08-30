# afrikaans_ner_corpus

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/afrikaans_ner_corpus)
- [Huggingface](https://huggingface.co/datasets/afrikaans_ner_corpus)

## afrikaans_ner_corpus

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:afrikaans_ner_corpus/afrikaans_ner_corpus')
```

- **说明**：

```
Named entity annotated data from the NCHLT Text Resource Development: Phase II Project, annotated with PERSON, LOCATION, ORGANISATION and MISCELLANEOUS tags.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 8962

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
                "OUT",
                "B-PERS",
                "I-PERS",
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
