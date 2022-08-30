# giga_fren

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/giga_fren)
- [Huggingface](https://huggingface.co/datasets/giga_fren)

## en-fr

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:giga_fren/en-fr')
```

- **Description**:

```
Giga-word corpus for French-English from WMT2010 collected by Chris Callison-Burch
2 languages, total number of files: 452
total number of tokens: 1.43G
total number of sentence fragments: 47.55M
```

- **许可**：无已知许可
- **版本**：2.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 22519904

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
            "fr"
        ],
        "id": null,
        "_type": "Translation"
    }
}
```
