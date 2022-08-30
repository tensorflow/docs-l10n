# dialog_re

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/dialog_re)
- [Huggingface](https://huggingface.co/datasets/dialog_re)

## dialog_re

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:dialog_re/dialog_re')
```

- **说明**：

```
DialogRE is the first human-annotated dialogue based relation extraction (RE) dataset aiming
to support the prediction of relation(s) between two arguments that appear in a dialogue.
The dataset annotates all occurrences of 36 possible relation types that exist between pairs
of arguments in the 1,788 dialogues originating from the complete transcripts of Friends.
```

- **许可**：https://github.com/nlpdata/dialogre/blob/master/license.txt
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 357
`'train'` | 1073
`'validation'` | 358

- **特征**：

```json
{
    "dialog": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "relation_data": {
        "feature": {
            "x": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "y": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "x_type": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "y_type": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "r": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "rid": {
                "feature": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            },
            "t": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```
