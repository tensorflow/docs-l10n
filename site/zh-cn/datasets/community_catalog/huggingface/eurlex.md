# eurlex

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/eurlex)
- [Huggingface](https://huggingface.co/datasets/eurlex)

## eurlex57k

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:eurlex/eurlex57k')
```

- **说明**：

```
EURLEX57K contains 57k legislative documents in English from EUR-Lex portal, annotated with EUROVOC concepts.
```

- **许可**：CC BY-SA (Creative Commons / Attribution-ShareAlike)
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 6000
`'train'` | 45000
`'validation'` | 6000

- **特征**：

```json
{
    "celex_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "eurovoc_concepts": {
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
