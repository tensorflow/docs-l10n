# id_panl_bppt

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/id_panl_bppt)
- [Huggingface](https://huggingface.co/datasets/id_panl_bppt)

## id_panl_bppt

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:id_panl_bppt/id_panl_bppt')
```

- **Description**:

```
Parallel Text Corpora for Multi-Domain Translation System created by BPPT (Indonesian Agency for the Assessment and
Application of Technology) for PAN Localization Project (A Regional Initiative to Develop Local Language Computing
Capacity in Asia). The dataset contains around 24K sentences divided in 4 difference topics (Economic, international,
Science and Technology and Sport).
```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 24021

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "translation": {
        "languages": [
            "en",
            "id"
        ],
        "id": null,
        "_type": "Translation"
    },
    "topic": {
        "num_classes": 4,
        "names": [
            "Economy",
            "International",
            "Science",
            "Sport"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
