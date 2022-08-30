# covid_qa_deepset

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/covid_qa_deepset)
- [Huggingface](https://huggingface.co/datasets/covid_qa_deepset)

## covid_qa_deepset

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:covid_qa_deepset/covid_qa_deepset')
```

- **Description**:

```
COVID-QA is a Question Answering dataset consisting of 2,019 question/answer pairs annotated by volunteer biomedical experts on scientific articles related to COVID-19.
```

- **许可**：Apache License 2.0
- **Version**: 1.0.0
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
