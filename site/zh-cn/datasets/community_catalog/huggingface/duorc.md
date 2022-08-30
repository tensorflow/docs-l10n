# duorc

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/duorc)
- [Huggingface](https://huggingface.co/datasets/duorc)

## SelfRC

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:duorc/SelfRC')
```

- **说明**：

```
DuoRC contains 186,089 unique question-answer pairs created from a collection of 7680 pairs of movie plots where each pair in the collection reflects two versions of the same movie.
```

- **许可**：https://raw.githubusercontent.com/duorc/duorc/master/LICENSE
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 12559
`'train'` | 60721
`'validation'` | 12961

- **特征**：

```json
{
    "plot_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "plot": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
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
    "answers": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "no_answer": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    }
}
```

## ParaphraseRC

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:duorc/ParaphraseRC')
```

- **说明**：

```
DuoRC contains 186,089 unique question-answer pairs created from a collection of 7680 pairs of movie plots where each pair in the collection reflects two versions of the same movie.
```

- **许可**：https://raw.githubusercontent.com/duorc/duorc/master/LICENSE
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 15857
`'train'` | 69524
`'validation'` | 15591

- **特征**：

```json
{
    "plot_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "plot": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
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
    "answers": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "no_answer": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    }
}
```
