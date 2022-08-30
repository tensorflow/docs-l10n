# circa

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/circa)
- [Huggingface](https://huggingface.co/datasets/circa)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:circa')
```

- **说明**：

```
The Circa (meaning ‘approximately’) dataset aims to help machine learning systems
to solve the problem of interpreting indirect answers to polar questions.

The dataset contains pairs of yes/no questions and indirect answers, together with
annotations for the interpretation of the answer. The data is collected in 10
different social conversational situations (eg. food preferences of a friend).

NOTE: There might be missing labels in the dataset and we have replaced them with -1.
The original dataset contains no train/dev/test splits.
```

- **许可**：Creative Commons Attribution 4.0 License
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 34268

- **特征**：

```json
{
    "context": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question-X": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "canquestion-X": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "answer-Y": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "judgements": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "goldstandard1": {
        "num_classes": 8,
        "names": [
            "Yes",
            "No",
            "In the middle, neither yes nor no",
            "Probably yes / sometimes yes",
            "Probably no",
            "Yes, subject to some conditions",
            "Other",
            "I am not sure how X will interpret Y\u2019s answer"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "goldstandard2": {
        "num_classes": 5,
        "names": [
            "Yes",
            "No",
            "In the middle, neither yes nor no",
            "Yes, subject to some conditions",
            "Other"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
