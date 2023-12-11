# hellaswag

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/hellaswag)
- [Huggingface](https://huggingface.co/datasets/hellaswag)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hellaswag')
```

- **说明**：

```

```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 10003
`'train'` | 39905
`'validation'` | 10042

- **特征**：

```json
{
    "ind": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "activity_label": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "ctx_a": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "ctx_b": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "ctx": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "endings": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "source_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "split": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "split_type": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
