# cmu_hinglish_dog

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/cmu_hinglish_dog)
- [Huggingface](https://huggingface.co/datasets/cmu_hinglish_dog)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cmu_hinglish_dog')
```

- **说明**：

```
This is a collection of text conversations in Hinglish (code mixing between Hindi-English) and their corresponding English only versions. Can be used for Translating between the two.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 960
`'train'` | 8060
`'validation'` | 942

- **特征**：

```json
{
    "date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "docIdx": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "translation": {
        "languages": [
            "en",
            "hi_en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "uid": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "utcTimestamp": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "rating": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "status": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "uid1LogInTime": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "uid1LogOutTime": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "uid1response": {
        "response": {
            "feature": {
                "dtype": "int64",
                "id": null,
                "_type": "Value"
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        },
        "type": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    },
    "uid2response": {
        "response": {
            "feature": {
                "dtype": "int64",
                "id": null,
                "_type": "Value"
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        },
        "type": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    },
    "user2_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "whoSawDoc": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "wikiDocumentIdx": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    }
}
```
