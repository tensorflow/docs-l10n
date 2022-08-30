# hate_offensive

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/hate_offensive)
- [Huggingface](https://huggingface.co/datasets/hate_offensive)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:hate_offensive')
```

- **Description**:

```
This dataset contains annotated tweets for automated hate-speech recognition
```

- **许可**：无已知许可
- **Version**: 1.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 24783

- **特征**：

```json
{
    "total_annotation_count": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "hate_speech_annotations": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "offensive_language_annotations": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "neither_annotations": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 3,
        "names": [
            "hate-speech",
            "offensive-language",
            "neither"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "tweet": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
