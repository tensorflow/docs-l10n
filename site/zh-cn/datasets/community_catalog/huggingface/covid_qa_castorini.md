# covid_qa_castorini

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/covid_qa_castorini)
- [Huggingface](https://huggingface.co/datasets/covid_qa_castorini)

## covid_qa_deepset

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:covid_qa_castorini/covid_qa_deepset')
```

- **说明**：

```
COVID-QA is a Question Answering dataset consisting of 2,019 question/answer pairs annotated by volunteer biomedical experts on scientific articles related to COVID-19.
```

- **许可**：Apache License 2.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2019

- **特征**：

```json
{
    "document_id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "context": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "is_impossible": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "answers": {
        "feature": {
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "answer_start": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## covidqa

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:covid_qa_castorini/covidqa')
```

- **说明**：

```
CovidQA is the beginnings of a question answering dataset specifically designed for COVID-19, built by hand from knowledge gathered from Kaggle's COVID-19 Open Research Dataset Challenge.
```

- **许可**：Apache License 2.0
- **版本**：0.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 27

- **特征**：

```json
{
    "category_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question_query": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "keyword_query": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "answers": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "title": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "exact_answer": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## covid_qa_castorini

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:covid_qa_castorini/covid_qa_castorini')
```

- **说明**：

```
CovidQA is the beginnings of a question answering dataset specifically designed for COVID-19, built by hand from knowledge gathered from Kaggle's COVID-19 Open Research Dataset Challenge.
```

- **许可**：Apache License 2.0
- **版本**：0.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 27

- **特征**：

```json
{
    "category_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question_query": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "keyword_query": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "answers": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "title": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "exact_answer": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```
