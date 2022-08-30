# eu_regulatory_ir

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/eu_regulatory_ir)
- [Huggingface](https://huggingface.co/datasets/eu_regulatory_ir)

## eu2uk

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:eu_regulatory_ir/eu2uk')
```

- **说明**：

```
EURegIR: Regulatory Compliance IR (EU/UK)
```

- **许可**：CC BY-SA (Creative Commons / Attribution-ShareAlike)
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 300
`'train'` | 1400
`'uk_corpus'` | 52515
`'validation'` | 300

- **特征**：

```json
{
    "document_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "publication_year": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "relevant_documents": {
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

## uk2eu

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:eu_regulatory_ir/uk2eu')
```

- **说明**：

```
EURegIR: Regulatory Compliance IR (EU/UK)
```

- **许可**：CC BY-SA (Creative Commons / Attribution-ShareAlike)
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'eu_corpus'` | 3930
`'test'` | 300
`'train'` | 1500
`'validation'` | 300

- **特征**：

```json
{
    "document_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "publication_year": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "relevant_documents": {
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
