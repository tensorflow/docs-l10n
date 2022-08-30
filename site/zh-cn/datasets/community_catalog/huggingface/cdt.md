# cdt

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/cdt)
- [Huggingface](https://huggingface.co/datasets/cdt)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cdt')
```

- **说明**：

```
The Cyberbullying Detection task was part of 2019 edition of PolEval competition. The goal is to predict if a given Twitter message contains a cyberbullying (harmful) content.
```

- **许可**：BSD 3-Clause
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 10041

- **特征**：

```json
{
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "num_classes": 2,
        "names": [
            "0",
            "1"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
