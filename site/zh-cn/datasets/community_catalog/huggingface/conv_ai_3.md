# conv_ai_3

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/conv_ai_3)
- [Huggingface](https://huggingface.co/datasets/conv_ai_3)

## conv_ai_3

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:conv_ai_3/conv_ai_3')
```

- **说明**：

```
The Conv AI 3 challenge is organized as part of the Search-oriented Conversational AI (SCAI) EMNLP workshop in 2020. The main aim of the conversational systems is to return an appropriate answer in response to the user requests. However, some user requests might be ambiguous. In Information Retrieval (IR) settings such a situation is handled mainly through the diversification of search result page. It is however much more challenging in dialogue settings. Hence, we aim to study the following situation for dialogue settings:
- a user is asking an ambiguous question (where ambiguous question is a question to which one can return > 1 possible answers)
- the system must identify that the question is ambiguous, and, instead of trying to answer it directly, ask a good clarifying question.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 9176
`'validation'` | 2313

- **特征**：

```json
{
    "topic_id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "initial_request": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "topic_desc": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "clarification_need": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "facet_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "facet_desc": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "answer": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
