# acronym_identification

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/acronym_identification)
- [Huggingface](https://huggingface.co/datasets/acronym_identification)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:acronym_identification')
```

- **说明**：

```
Acronym identification training and development sets for the acronym identification task at SDU@AAAI-21.
```

- **许可证**：无已知许可证
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1750
`'train'` | 14006
`'validation'` | 1717

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "labels": {
        "feature": {
            "num_classes": 5,
            "names": [
                "B-long",
                "B-short",
                "I-long",
                "I-short",
                "O"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```
