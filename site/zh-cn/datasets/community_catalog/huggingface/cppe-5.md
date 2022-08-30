# cppe-5

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/cppe-5)
- [Huggingface](https://huggingface.co/datasets/cppe-5)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cppe-5')
```

- **说明**：

```
CPPE - 5 (Medical Personal Protective Equipment) is a new challenging dataset with the goal
to allow the study of subordinate categorization of medical personal protective equipments,
which is not possible with other popular data sets that focus on broad level categories.
```

- **许可**：未知
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 29
`'train'` | 1000

- **特征**：

```json
{
    "image_id": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "image": {
        "id": null,
        "_type": "Image"
    },
    "width": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "height": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "objects": {
        "feature": {
            "id": {
                "dtype": "int64",
                "id": null,
                "_type": "Value"
            },
            "area": {
                "dtype": "int64",
                "id": null,
                "_type": "Value"
            },
            "bbox": {
                "feature": {
                    "dtype": "float32",
                    "id": null,
                    "_type": "Value"
                },
                "length": 4,
                "id": null,
                "_type": "Sequence"
            },
            "category": {
                "num_classes": 5,
                "names": [
                    "Coverall",
                    "Face_Shield",
                    "Gloves",
                    "Goggles",
                    "Mask"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```
