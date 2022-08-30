# cord19

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/cord19)
- [Huggingface](https://huggingface.co/datasets/cord19)

## metadata

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cord19/metadata')
```

- **说明**：

```
The Covid-19 Open Research Dataset (CORD-19) is a growing resource of scientific papers on Covid-19 and related
historical coronavirus research. CORD-19 is designed to facilitate the development of text mining and information
retrieval systems over its rich collection of metadata and structured full text papers. Since its release, CORD-19
has been downloaded over 75K times and has served as the basis of many Covid-19 text mining and discovery systems.

The dataset itself isn't defining a specific task, but there is a Kaggle challenge that define 17 open research
questions to be solved with the dataset: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 368618

- **特征**：

```json
{
    "cord_uid": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sha": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_x": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "doi": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "abstract": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "publish_time": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "authors": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "journal": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## fulltext

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cord19/fulltext')
```

- **说明**：

```
The Covid-19 Open Research Dataset (CORD-19) is a growing resource of scientific papers on Covid-19 and related
historical coronavirus research. CORD-19 is designed to facilitate the development of text mining and information
retrieval systems over its rich collection of metadata and structured full text papers. Since its release, CORD-19
has been downloaded over 75K times and has served as the basis of many Covid-19 text mining and discovery systems.

The dataset itself isn't defining a specific task, but there is a Kaggle challenge that define 17 open research
questions to be solved with the dataset: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 368618

- **特征**：

```json
{
    "cord_uid": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sha": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_x": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "doi": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "abstract": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "publish_time": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "authors": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "journal": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "fulltext": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## embeddings

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cord19/embeddings')
```

- **说明**：

```
The Covid-19 Open Research Dataset (CORD-19) is a growing resource of scientific papers on Covid-19 and related
historical coronavirus research. CORD-19 is designed to facilitate the development of text mining and information
retrieval systems over its rich collection of metadata and structured full text papers. Since its release, CORD-19
has been downloaded over 75K times and has served as the basis of many Covid-19 text mining and discovery systems.

The dataset itself isn't defining a specific task, but there is a Kaggle challenge that define 17 open research
questions to be solved with the dataset: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 368618

- **特征**：

```json
{
    "cord_uid": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sha": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_x": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "doi": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "abstract": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "publish_time": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "authors": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "journal": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "doc_embeddings": {
        "feature": {
            "dtype": "float64",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```
