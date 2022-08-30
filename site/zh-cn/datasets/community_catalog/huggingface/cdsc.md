# cdsc

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/cdsc)
- [Huggingface](https://huggingface.co/datasets/cdsc)

## cdsc-e

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cdsc/cdsc-e')
```

- **说明**：

```
Polish CDSCorpus consists of 10K Polish sentence pairs which are human-annotated for semantic relatedness and entailment. The dataset may be used for the evaluation of compositional distributional semantics models of Polish. The dataset was presented at ACL 2017. Please refer to the Wróblewska and Krasnowska-Kieraś (2017) for a detailed description of the resource.
```

- **许可**：CC BY-NC-SA 4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 8000
`'validation'` | 1000

- **特征**：

```json
{
    "pair_ID": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "sentence_A": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sentence_B": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "entailment_judgment": {
        "num_classes": 3,
        "names": [
            "NEUTRAL",
            "CONTRADICTION",
            "ENTAILMENT"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## cdsc-r

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cdsc/cdsc-r')
```

- **说明**：

```
Polish CDSCorpus consists of 10K Polish sentence pairs which are human-annotated for semantic relatedness and entailment. The dataset may be used for the evaluation of compositional distributional semantics models of Polish. The dataset was presented at ACL 2017. Please refer to the Wróblewska and Krasnowska-Kieraś (2017) for a detailed description of the resource.
```

- **许可**：CC BY-NC-SA 4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 8000
`'validation'` | 1000

- **特征**：

```json
{
    "pair_ID": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "sentence_A": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sentence_B": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "relatedness_score": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    }
}
```
