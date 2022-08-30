# coarse_discourse

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/coarse_discourse)
- [Huggingface](https://huggingface.co/datasets/coarse_discourse)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:coarse_discourse')
```

- **说明**：

```
dataset contains discourse annotation and relation on threads from reddit during 2016
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 116357

- **特征**：

```json
{
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "is_self_post": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "subreddit": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "majority_link": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "is_first_post": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "majority_type": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "id_post": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "post_depth": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "in_reply_to": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotations": {
        "feature": {
            "annotator": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "link_to_post": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "main_type": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```
