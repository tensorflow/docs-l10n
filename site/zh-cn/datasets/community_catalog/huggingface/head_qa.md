# head_qa

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/head_qa)
- [Huggingface](https://huggingface.co/datasets/head_qa)

## es

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:head_qa/es')
```

- **说明**：

```
HEAD-QA is a multi-choice HEAlthcare Dataset. The questions come from exams to access a specialized position in the
Spanish healthcare system, and are challenging even for highly specialized humans. They are designed by the Ministerio
de Sanidad, Consumo y Bienestar Social.

The dataset contains questions about the following topics: medicine, nursing, psychology, chemistry, pharmacology and biology.
```

- **许可**：MIT License
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2742
`'train'` | 2657
`'validation'` | 1366

- **特征**：

```json
{
    "name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "year": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "qid": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "qtext": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "ra": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "image": {
        "id": null,
        "_type": "Image"
    },
    "answers": [
        {
            "aid": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            },
            "atext": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            }
        }
    ]
}
```

## en

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:head_qa/en')
```

- **说明**：

```
HEAD-QA is a multi-choice HEAlthcare Dataset. The questions come from exams to access a specialized position in the
Spanish healthcare system, and are challenging even for highly specialized humans. They are designed by the Ministerio
de Sanidad, Consumo y Bienestar Social.

The dataset contains questions about the following topics: medicine, nursing, psychology, chemistry, pharmacology and biology.
```

- **许可**：MIT License
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2742
`'train'` | 2657
`'validation'` | 1366

- **特征**：

```json
{
    "name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "year": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "qid": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "qtext": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "ra": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "image": {
        "id": null,
        "_type": "Image"
    },
    "answers": [
        {
            "aid": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            },
            "atext": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            }
        }
    ]
}
```
