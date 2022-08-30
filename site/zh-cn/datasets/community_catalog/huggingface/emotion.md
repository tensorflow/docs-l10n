# emotion

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/emotion)
- [Huggingface](https://huggingface.co/datasets/emotion)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:emotion')
```

- **Description**:

```
Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise. For more detailed information please refer to the paper.
```

- **许可**：无已知许可
- **Version**: 0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2000
`'train'` | 16000
`'validation'` | 2000

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 6,
        "names": [
            "sadness",
            "joy",
            "love",
            "anger",
            "fear",
            "surprise"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
