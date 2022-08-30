# hate_speech_offensive

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/hate_speech_offensive)
- [Huggingface](https://huggingface.co/datasets/hate_speech_offensive)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:hate_speech_offensive')
```

- **Description**:

```
An annotated dataset for hate speech and offensive language detection on tweets.
```

- **许可**：MIT
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 24783

- **特征**：

```json
{
    "count": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "hate_speech_count": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "offensive_language_count": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "neither_count": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "class": {
        "num_classes": 3,
        "names": [
            "hate speech",
            "offensive language",
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
