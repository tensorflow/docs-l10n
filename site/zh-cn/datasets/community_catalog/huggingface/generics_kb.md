# generics_kb

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/generics_kb)
- [Huggingface](https://huggingface.co/datasets/generics_kb)

## generics_kb_best

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:generics_kb/generics_kb_best')
```

- **说明**：

```
The GenericsKB contains 3.4M+ generic sentences about the world, i.e., sentences expressing general truths such as "Dogs bark," and "Trees remove carbon dioxide from the atmosphere." Generics are potentially useful as a knowledge source for AI systems requiring general world knowledge. The GenericsKB is the first large-scale resource containing naturally occurring generic sentences (as opposed to extracted or crowdsourced triples), and is rich in high-quality, general, semantically complete statements. Generics were primarily extracted from three large text sources, namely the Waterloo Corpus, selected parts of Simple Wikipedia, and the ARC Corpus. A filtered, high-quality subset is also available in GenericsKB-Best, containing 1,020,868 sentences. We recommend you start with GenericsKB-Best.
```

- **许可**：cc-by-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1020868

- **特征**：

```json
{
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "term": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "quantifier_frequency": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "quantifier_number": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "generic_sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "score": {
        "dtype": "float64",
        "id": null,
        "_type": "Value"
    }
}
```

## generics_kb

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:generics_kb/generics_kb')
```

- **说明**：

```
The GenericsKB contains 3.4M+ generic sentences about the world, i.e., sentences expressing general truths such as "Dogs bark," and "Trees remove carbon dioxide from the atmosphere." Generics are potentially useful as a knowledge source for AI systems requiring general world knowledge. The GenericsKB is the first large-scale resource containing naturally occurring generic sentences (as opposed to extracted or crowdsourced triples), and is rich in high-quality, general, semantically complete statements. Generics were primarily extracted from three large text sources, namely the Waterloo Corpus, selected parts of Simple Wikipedia, and the ARC Corpus. A filtered, high-quality subset is also available in GenericsKB-Best, containing 1,020,868 sentences. We recommend you start with GenericsKB-Best.
```

- **许可**：cc-by-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 3433000

- **特征**：

```json
{
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "term": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "quantifier_frequency": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "quantifier_number": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "generic_sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "score": {
        "dtype": "float64",
        "id": null,
        "_type": "Value"
    }
}
```

## generics_kb_simplewiki

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:generics_kb/generics_kb_simplewiki')
```

- **说明**：

```
The GenericsKB contains 3.4M+ generic sentences about the world, i.e., sentences expressing general truths such as "Dogs bark," and "Trees remove carbon dioxide from the atmosphere." Generics are potentially useful as a knowledge source for AI systems requiring general world knowledge. The GenericsKB is the first large-scale resource containing naturally occurring generic sentences (as opposed to extracted or crowdsourced triples), and is rich in high-quality, general, semantically complete statements. Generics were primarily extracted from three large text sources, namely the Waterloo Corpus, selected parts of Simple Wikipedia, and the ARC Corpus. A filtered, high-quality subset is also available in GenericsKB-Best, containing 1,020,868 sentences. We recommend you start with GenericsKB-Best.
```

- **许可**：cc-by-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 12765

- **特征**：

```json
{
    "source_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sentences_before": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "sentences_after": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "concept_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "quantifiers": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "bert_score": {
        "dtype": "float64",
        "id": null,
        "_type": "Value"
    },
    "headings": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "categories": {
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

## generics_kb_waterloo

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:generics_kb/generics_kb_waterloo')
```

- **说明**：

```
The GenericsKB contains 3.4M+ generic sentences about the world, i.e., sentences expressing general truths such as "Dogs bark," and "Trees remove carbon dioxide from the atmosphere." Generics are potentially useful as a knowledge source for AI systems requiring general world knowledge. The GenericsKB is the first large-scale resource containing naturally occurring generic sentences (as opposed to extracted or crowdsourced triples), and is rich in high-quality, general, semantically complete statements. Generics were primarily extracted from three large text sources, namely the Waterloo Corpus, selected parts of Simple Wikipedia, and the ARC Corpus. A filtered, high-quality subset is also available in GenericsKB-Best, containing 1,020,868 sentences. We recommend you start with GenericsKB-Best.
```

- **许可**：cc-by-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 3666725

- **特征**：

```json
{
    "source_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sentences_before": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "sentences_after": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "concept_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "quantifiers": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "bert_score": {
        "dtype": "float64",
        "id": null,
        "_type": "Value"
    }
}
```
