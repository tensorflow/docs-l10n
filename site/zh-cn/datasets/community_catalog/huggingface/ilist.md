# ilist

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/ilist)
- [Huggingface](https://huggingface.co/datasets/ilist)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:ilist')
```

- **Description**:

```
This dataset is introduced in a task which aimed at identifying 5 closely-related languages of Indo-Aryan language family –
Hindi (also known as Khari Boli), Braj Bhasha, Awadhi, Bhojpuri, and Magahi.
```

- **许可**：无已知许可
- **Version**: 0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 9692
`'train'` | 70351
`'validation'` | 10329

- **特征**：

```json
{
    "language_id": {
        "num_classes": 5,
        "names": [
            "AWA",
            "BRA",
            "MAG",
            "BHO",
            "HIN"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
