# biwi_kinect_head_pose

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/biwi_kinect_head_pose)
- [Huggingface](https://huggingface.co/datasets/biwi_kinect_head_pose)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:biwi_kinect_head_pose')
```

- **说明**：

```
The Biwi Kinect Head Pose Database is acquired with the Microsoft Kinect sensor, a structured IR light device.It contains 15K images of 20 people with 6 females and 14 males where 4 people were recorded twice.
```

- **许可**：此数据库可用于非商业用途，例如大学研究和教育。
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 24

- **特征**：

```json
{
    "sequence_number": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "subject_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "rgb": {
        "feature": {
            "decode": true,
            "id": null,
            "_type": "Image"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "rgb_cal": {
        "intrisic_mat": {
            "shape": [
                3,
                3
            ],
            "dtype": "float64",
            "id": null,
            "_type": "Array2D"
        },
        "extrinsic_mat": {
            "rotation": {
                "shape": [
                    3,
                    3
                ],
                "dtype": "float64",
                "id": null,
                "_type": "Array2D"
            },
            "translation": {
                "feature": {
                    "dtype": "float64",
                    "id": null,
                    "_type": "Value"
                },
                "length": 3,
                "id": null,
                "_type": "Sequence"
            }
        }
    },
    "depth": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "depth_cal": {
        "intrisic_mat": {
            "shape": [
                3,
                3
            ],
            "dtype": "float64",
            "id": null,
            "_type": "Array2D"
        },
        "extrinsic_mat": {
            "rotation": {
                "shape": [
                    3,
                    3
                ],
                "dtype": "float64",
                "id": null,
                "_type": "Array2D"
            },
            "translation": {
                "feature": {
                    "dtype": "float64",
                    "id": null,
                    "_type": "Value"
                },
                "length": 3,
                "id": null,
                "_type": "Sequence"
            }
        }
    },
    "head_pose_gt": {
        "feature": {
            "center": {
                "feature": {
                    "dtype": "float64",
                    "id": null,
                    "_type": "Value"
                },
                "length": 3,
                "id": null,
                "_type": "Sequence"
            },
            "rotation": {
                "shape": [
                    3,
                    3
                ],
                "dtype": "float64",
                "id": null,
                "_type": "Array2D"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "head_template": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
