# diplomacy_detection

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/diplomacy_detection)
- [Huggingface](https://huggingface.co/datasets/diplomacy_detection)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:diplomacy_detection')
```

- **说明**：

```
The Diplomacy dataset contains pairwise conversations annotated by the sender and the receiver for deception (and conversely truthfulness).   The 17,289 messages are gathered from 12 games.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 42
`'train'` | 189
`'validation'` | 21

- **特征**：

```json
{
    "messages": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "sender_labels": {
        "feature": {
            "num_classes": 2,
            "names": [
                "false",
                "true"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "receiver_labels": {
        "feature": {
            "num_classes": 3,
            "names": [
                "false",
                "true",
                "noannotation"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "speakers": {
        "feature": {
            "num_classes": 7,
            "names": [
                "italy",
                "turkey",
                "russia",
                "england",
                "austria",
                "germany",
                "france"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "receivers": {
        "feature": {
            "num_classes": 7,
            "names": [
                "italy",
                "turkey",
                "russia",
                "england",
                "austria",
                "germany",
                "france"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "absolute_message_index": {
        "feature": {
            "dtype": "int64",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "relative_message_index": {
        "feature": {
            "dtype": "int64",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "seasons": {
        "feature": {
            "num_classes": 6,
            "names": [
                "spring",
                "fall",
                "winter",
                "Spring",
                "Fall",
                "Winter"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "years": {
        "feature": {
            "num_classes": 18,
            "names": [
                "1901",
                "1902",
                "1903",
                "1904",
                "1905",
                "1906",
                "1907",
                "1908",
                "1909",
                "1910",
                "1911",
                "1912",
                "1913",
                "1914",
                "1915",
                "1916",
                "1917",
                "1918"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "game_score": {
        "feature": {
            "num_classes": 19,
            "names": [
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "game_score_delta": {
        "feature": {
            "num_classes": 37,
            "names": [
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "-1",
                "-2",
                "-3",
                "-4",
                "-5",
                "-6",
                "-7",
                "-8",
                "-9",
                "-10",
                "-11",
                "-12",
                "-13",
                "-14",
                "-15",
                "-16",
                "-17",
                "-18"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "players": {
        "feature": {
            "num_classes": 7,
            "names": [
                "italy",
                "turkey",
                "russia",
                "england",
                "austria",
                "germany",
                "france"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "game_id": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    }
}
```
