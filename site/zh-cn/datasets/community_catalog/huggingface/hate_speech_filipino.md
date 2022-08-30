# hate_speech_filipino

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/hate_speech_filipino)
- [Huggingface](https://huggingface.co/datasets/hate_speech_filipino)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:hate_speech_filipino')
```

- **Description**:

```
Contains 10k tweets (training set) that are labeled as hate speech or non-hate speech. Released with 4,232 validation and 4,232 testing samples. Collected during the 2016 Philippine Presidential Elections.
```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 10000
`'train'` | 10000
`'validation'` | 4232

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
            "0",
            "1"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
