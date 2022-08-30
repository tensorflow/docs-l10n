# danish_political_comments

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/danish_political_comments)
- [Huggingface](https://huggingface.co/datasets/danish_political_comments)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:danish_political_comments')
```

- **Description**:

```
The dataset consists of 9008 sentences that are labelled with fine-grained polarity in the range from -2 to 2 (negative to postive). The quality of the fine-grained is not cross validated and is therefore subject to uncertainties; however, the simple polarity has been cross validated and therefore is considered to be more correct.
```

- **许可**：无已知许可
- **版本**：0.9.1
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 9008

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "num_classes": 5,
        "names": [
            "2",
            "1",
            "0",
            "-1",
            "-2"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
