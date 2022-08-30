# cos_e

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/cos_e)
- [Huggingface](https://huggingface.co/datasets/cos_e)

## v1.0

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cos_e/v1.0')
```

- **说明**：

```
Common Sense Explanations (CoS-E) allows for training language models to
automatically generate explanations that can be used during training and
inference in a novel Commonsense Auto-Generated Explanation (CAGE) framework.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 7610
`'validation'` | 950

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "abstractive_explanation": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "extractive_explanation": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## v1.11

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cos_e/v1.11')
```

- **说明**：

```
Common Sense Explanations (CoS-E) allows for training language models to
automatically generate explanations that can be used during training and
inference in a novel Commonsense Auto-Generated Explanation (CAGE) framework.
```

- **许可**：无已知许可
- **版本**：1.11.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 9741
`'validation'` | 1221

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "abstractive_explanation": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "extractive_explanation": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
