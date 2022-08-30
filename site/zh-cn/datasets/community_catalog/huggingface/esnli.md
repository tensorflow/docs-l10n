# esnli

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/esnli)
- [Huggingface](https://huggingface.co/datasets/esnli)

## plain_text

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:esnli/plain_text')
```

- **说明**：

```
The e-SNLI dataset extends the Stanford Natural Language Inference Dataset to
include human-annotated natural language explanations of the entailment
relations.
```

- **许可**：无已知许可
- **版本**：0.0.2
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 9824
`'train'` | 549367
`'validation'` | 9842

- **特征**：

```json
{
    "premise": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "hypothesis": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 3,
        "names": [
            "entailment",
            "neutral",
            "contradiction"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "explanation_1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "explanation_2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "explanation_3": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
