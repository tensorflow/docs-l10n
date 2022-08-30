# hate_speech_portuguese

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/hate_speech_portuguese)
- [Huggingface](https://huggingface.co/datasets/hate_speech_portuguese)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:hate_speech_portuguese')
```

- **Description**:

```
Portuguese dataset for hate speech detection composed of 5,668 tweets with binary annotations (i.e. 'hate' vs. 'no-hate').
```

- **许可**：未知
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 5670

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 2,
        "names": [
            "no-hate",
            "hate"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "hatespeech_G1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_G1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "hatespeech_G2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_G2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "hatespeech_G3": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_G3": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
