# counter

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/counter)
- [Huggingface](https://huggingface.co/datasets/counter)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:counter')
```

- **Description**:

```
The COrpus of Urdu News TExt Reuse (COUNTER) corpus contains 1200 documents with real examples of text reuse from the field of journalism. It has been manually annotated at document level with three levels of reuse: wholly derived, partially derived and non derived.
```

- **许可**：该语料库在 Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License 下许可。
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 600

- **特征**：

```json
{
    "source": {
        "filename": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "headline": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "body": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "total_number_of_words": {
            "dtype": "int64",
            "id": null,
            "_type": "Value"
        },
        "total_number_of_sentences": {
            "dtype": "int64",
            "id": null,
            "_type": "Value"
        },
        "number_of_words_with_swr": {
            "dtype": "int64",
            "id": null,
            "_type": "Value"
        },
        "newspaper": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "newsdate": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "domain": {
            "num_classes": 5,
            "names": [
                "business",
                "sports",
                "national",
                "foreign",
                "showbiz"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "classification": {
            "num_classes": 3,
            "names": [
                "wholly_derived",
                "partially_derived",
                "not_derived"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        }
    },
    "derived": {
        "filename": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "headline": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "body": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "total_number_of_words": {
            "dtype": "int64",
            "id": null,
            "_type": "Value"
        },
        "total_number_of_sentences": {
            "dtype": "int64",
            "id": null,
            "_type": "Value"
        },
        "number_of_words_with_swr": {
            "dtype": "int64",
            "id": null,
            "_type": "Value"
        },
        "newspaper": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "newsdate": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "domain": {
            "num_classes": 5,
            "names": [
                "business",
                "sports",
                "national",
                "foreign",
                "showbiz"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "classification": {
            "num_classes": 3,
            "names": [
                "wholly_derived",
                "partially_derived",
                "not_derived"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        }
    }
}
```
