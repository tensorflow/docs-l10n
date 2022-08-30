# cedr

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/cedr)
- [Huggingface](https://huggingface.co/datasets/cedr)

## main

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cedr/main')
```

- **说明**：

```
This new dataset is designed to solve emotion recognition task for text data in Russian. The Corpus for Emotions Detecting in
Russian-language text sentences of different social sources (CEDR) contains 9410 sentences in Russian labeled for 5 emotion
categories. The data collected from different sources: posts of the LiveJournal social network, texts of the online news
agency Lenta.ru, and Twitter microblog posts. There are two variants of the corpus: main and enriched. The enriched variant
is include tokenization and lemmatization. Dataset with predefined train/test splits.
```

- **许可**：http://www.apache.org/licenses/LICENSE-2.0
- **版本**：0.1.1
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1882
`'train'` | 7528

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "labels": {
        "feature": {
            "num_classes": 5,
            "names": [
                "joy",
                "sadness",
                "surprise",
                "fear",
                "anger"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## enriched

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cedr/enriched')
```

- **说明**：

```
This new dataset is designed to solve emotion recognition task for text data in Russian. The Corpus for Emotions Detecting in
Russian-language text sentences of different social sources (CEDR) contains 9410 sentences in Russian labeled for 5 emotion
categories. The data collected from different sources: posts of the LiveJournal social network, texts of the online news
agency Lenta.ru, and Twitter microblog posts. There are two variants of the corpus: main and enriched. The enriched variant
is include tokenization and lemmatization. Dataset with predefined train/test splits.
```

- **许可**：http://www.apache.org/licenses/LICENSE-2.0
- **版本**：0.1.1
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1882
`'train'` | 7528

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "labels": {
        "feature": {
            "num_classes": 5,
            "names": [
                "joy",
                "sadness",
                "surprise",
                "fear",
                "anger"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sentences": [
        [
            {
                "forma": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "lemma": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            }
        ]
    ]
}
```
