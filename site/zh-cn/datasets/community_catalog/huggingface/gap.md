# gap

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/gap)
- [Huggingface](https://huggingface.co/datasets/gap)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gap')
```

- **说明**：

```
GAP is a gender-balanced dataset containing 8,908 coreference-labeled pairs of
(ambiguous pronoun, antecedent name), sampled from Wikipedia and released by
Google AI Language for the evaluation of coreference resolution in practical
applications.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2000
`'train'` | 2000
`'validation'` | 454

- **特征**：

```json
{
    "ID": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Pronoun": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Pronoun-offset": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "A": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "A-offset": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "A-coref": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "B": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "B-offset": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "B-coref": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "URL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
