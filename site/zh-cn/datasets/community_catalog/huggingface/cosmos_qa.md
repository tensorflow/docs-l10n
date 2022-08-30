# cosmos_qa

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/cosmos_qa)
- [Huggingface](https://huggingface.co/datasets/cosmos_qa)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cosmos_qa')
```

- **说明**：

```
Cosmos QA is a large-scale dataset of 35.6K problems that require commonsense-based reading comprehension, formulated as multiple-choice questions. It focuses on reading between the lines over a diverse collection of people's everyday narratives, asking questions concerning on the likely causes or effects of events that require reasoning beyond the exact text spans in the context
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 6963
`'train'` | 25262
`'validation'` | 2985

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "context": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "answer0": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "answer1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "answer2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "answer3": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    }
}
```
