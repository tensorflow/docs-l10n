# dengue_filipino

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/dengue_filipino)
- [Huggingface](https://huggingface.co/datasets/dengue_filipino)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:dengue_filipino')
```

- **Description**:

```
Benchmark dataset for low-resource multiclass classification, with 4,015 training, 500 testing, and 500 validation examples, each labeled as part of five classes. Each sample can be a part of multiple classes. Collected as tweets.
```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 4015
`'train'` | 4015
`'validation'` | 500

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "absent": {
        "num_classes": 2,
        "names": [
            "0",
            "1"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "dengue": {
        "num_classes": 2,
        "names": [
            "0",
            "1"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "health": {
        "num_classes": 2,
        "names": [
            "0",
            "1"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "mosquito": {
        "num_classes": 2,
        "names": [
            "0",
            "1"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "sick": {
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
