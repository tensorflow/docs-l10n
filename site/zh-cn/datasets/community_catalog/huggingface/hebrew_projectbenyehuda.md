# hebrew_projectbenyehuda

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/hebrew_projectbenyehuda)
- [Huggingface](https://huggingface.co/datasets/hebrew_projectbenyehuda)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:hebrew_projectbenyehuda')
```

- **Description**:

```
This repository contains a dump of thousands of public domain works in Hebrew, from Project Ben-Yehuda, in plaintext UTF-8 files, with and without diacritics (nikkud). The metadata (pseudocatalogue.csv) file is a list of titles, authors, genres, and file paths, to help you process the dump.
All these works are in the public domain, so you are free to make any use of them, and do not need to ask for permission.
There are 10078 files, 3181136 lines
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 10078

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "authors": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "translators": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "original_language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "genre": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_edition": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
