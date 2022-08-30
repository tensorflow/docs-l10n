# cdt

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/cdt)
- [Huggingface](https://huggingface.co/datasets/cdt)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:cdt')
```

- **Description**:

```
The Cyberbullying Detection task was part of 2019 edition of PolEval competition. The goal is to predict if a given Twitter message contains a cyberbullying (harmful) content.
```

- **许可**：BSD 3-Clause
- **Version**: 1.1.0
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
