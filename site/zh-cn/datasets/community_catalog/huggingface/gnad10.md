# gnad10

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/gnad10)
- [Huggingface](https://huggingface.co/datasets/gnad10)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:gnad10')
```

- **Description**:

```
This dataset is intended to advance topic classification for German texts. A classifier that is efffective in
English may not be effective in German dataset because it has a higher inflection and longer compound words.
The 10kGNAD dataset contains 10273 German news articles from an Austrian online newspaper categorized into
9 categories. Article titles and text are concatenated together and authors are removed to avoid a keyword-like
classification on authors that write frequently about one category. This dataset can be used as a benchmark
for German topic classification.
```

- **许可**：无已知许可
- **Version**: 1.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1028
`'train'` | 9245

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 9,
        "names": [
            "Web",
            "Panorama",
            "International",
            "Wirtschaft",
            "Sport",
            "Inland",
            "Etat",
            "Wissenschaft",
            "Kultur"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
