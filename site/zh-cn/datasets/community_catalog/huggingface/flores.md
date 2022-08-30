# flores

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/flores)
- [Huggingface](https://huggingface.co/datasets/flores)

## neen

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:flores/neen')
```

- **Description**:

```
Evaluation datasets for low-resource machine translation: Nepali-English and Sinhala-English.
```

- **许可**：无已知许可
- **Version**: 1.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2836
`'validation'` | 2560

- **特征**：

```json
{
    "translation": {
        "languages": [
            "ne",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    }
}
```

## sien

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:flores/sien')
```

- **Description**:

```
Evaluation datasets for low-resource machine translation: Nepali-English and Sinhala-English.
```

- **许可**：无已知许可
- **Version**: 1.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2767
`'validation'` | 2899

- **特征**：

```json
{
    "translation": {
        "languages": [
            "si",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    }
}
```
