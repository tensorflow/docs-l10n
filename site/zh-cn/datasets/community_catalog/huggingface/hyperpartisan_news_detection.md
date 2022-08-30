# hyperpartisan_news_detection

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/hyperpartisan_news_detection)
- [Huggingface](https://huggingface.co/datasets/hyperpartisan_news_detection)

## byarticle

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:hyperpartisan_news_detection/byarticle')
```

- **Description**:

```
Hyperpartisan News Detection was a dataset created for PAN @ SemEval 2019 Task 4.
Given a news article text, decide whether it follows a hyperpartisan argumentation, i.e., whether it exhibits blind, prejudiced, or unreasoning allegiance to one party, faction, cause, or person.

There are 2 parts:
- byarticle: Labeled through crowdsourcing on an article basis. The data contains only articles for which a consensus among the crowdsourcing workers existed.
- bypublisher: Labeled by the overall bias of the publisher as provided by BuzzFeed journalists or MediaBiasFactCheck.com.
```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 645

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "hyperpartisan": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "published_at": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## bypublisher

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:hyperpartisan_news_detection/bypublisher')
```

- **Description**:

```
Hyperpartisan News Detection was a dataset created for PAN @ SemEval 2019 Task 4.
Given a news article text, decide whether it follows a hyperpartisan argumentation, i.e., whether it exhibits blind, prejudiced, or unreasoning allegiance to one party, faction, cause, or person.

There are 2 parts:
- byarticle: Labeled through crowdsourcing on an article basis. The data contains only articles for which a consensus among the crowdsourcing workers existed.
- bypublisher: Labeled by the overall bias of the publisher as provided by BuzzFeed journalists or MediaBiasFactCheck.com.
```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 600000
`'validation'` | 600000

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "hyperpartisan": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "published_at": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "bias": {
        "num_classes": 5,
        "names": [
            "right",
            "right-center",
            "least",
            "left-center",
            "left"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
