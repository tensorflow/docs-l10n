# codah

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/codah)
- [Huggingface](https://huggingface.co/datasets/codah)

## codah

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:codah/codah')
```

- **说明**：

```
The COmmonsense Dataset Adversarially-authored by Humans (CODAH) is an evaluation set for commonsense question-answering in the sentence completion style of SWAG. As opposed to other automatically generated NLI datasets, CODAH is adversarially constructed by humans who can view feedback from a pre-trained model and use this information to design challenging commonsense questions. Our experimental results show that CODAH questions present a complementary extension to the SWAG dataset, testing additional modes of common sense.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2776

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "question_category": {
        "num_classes": 6,
        "names": [
            "Idioms",
            "Reference",
            "Polysemy",
            "Negation",
            "Quantitative",
            "Others"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "question_propmt": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "candidate_answers": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "correct_answer_idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    }
}
```

## fold_0

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:codah/fold_0')
```

- **说明**：

```
The COmmonsense Dataset Adversarially-authored by Humans (CODAH) is an evaluation set for commonsense question-answering in the sentence completion style of SWAG. As opposed to other automatically generated NLI datasets, CODAH is adversarially constructed by humans who can view feedback from a pre-trained model and use this information to design challenging commonsense questions. Our experimental results show that CODAH questions present a complementary extension to the SWAG dataset, testing additional modes of common sense.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 555
`'train'` | 1665
`'validation'` | 556

- **特征**:

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "question_category": {
        "num_classes": 6,
        "names": [
            "Idioms",
            "Reference",
            "Polysemy",
            "Negation",
            "Quantitative",
            "Others"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "question_propmt": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "candidate_answers": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "correct_answer_idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    }
}
```

## fold_1

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:codah/fold_1')
```

- **说明**：

```
The COmmonsense Dataset Adversarially-authored by Humans (CODAH) is an evaluation set for commonsense question-answering in the sentence completion style of SWAG. As opposed to other automatically generated NLI datasets, CODAH is adversarially constructed by humans who can view feedback from a pre-trained model and use this information to design challenging commonsense questions. Our experimental results show that CODAH questions present a complementary extension to the SWAG dataset, testing additional modes of common sense.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 555
`'train'` | 1665
`'validation'` | 556

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "question_category": {
        "num_classes": 6,
        "names": [
            "Idioms",
            "Reference",
            "Polysemy",
            "Negation",
            "Quantitative",
            "Others"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "question_propmt": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "candidate_answers": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "correct_answer_idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    }
}
```

## fold_2

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:codah/fold_2')
```

- **说明**：

```
The COmmonsense Dataset Adversarially-authored by Humans (CODAH) is an evaluation set for commonsense question-answering in the sentence completion style of SWAG. As opposed to other automatically generated NLI datasets, CODAH is adversarially constructed by humans who can view feedback from a pre-trained model and use this information to design challenging commonsense questions. Our experimental results show that CODAH questions present a complementary extension to the SWAG dataset, testing additional modes of common sense.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 555
`'train'` | 1665
`'validation'` | 556

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "question_category": {
        "num_classes": 6,
        "names": [
            "Idioms",
            "Reference",
            "Polysemy",
            "Negation",
            "Quantitative",
            "Others"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "question_propmt": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "candidate_answers": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "correct_answer_idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    }
}
```

## fold_3

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:codah/fold_3')
```

- **说明**：

```
The COmmonsense Dataset Adversarially-authored by Humans (CODAH) is an evaluation set for commonsense question-answering in the sentence completion style of SWAG. As opposed to other automatically generated NLI datasets, CODAH is adversarially constructed by humans who can view feedback from a pre-trained model and use this information to design challenging commonsense questions. Our experimental results show that CODAH questions present a complementary extension to the SWAG dataset, testing additional modes of common sense.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 555
`'train'` | 1665
`'validation'` | 556

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "question_category": {
        "num_classes": 6,
        "names": [
            "Idioms",
            "Reference",
            "Polysemy",
            "Negation",
            "Quantitative",
            "Others"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "question_propmt": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "candidate_answers": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "correct_answer_idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    }
}
```

## fold_4

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:codah/fold_4')
```

- **说明**：

```
The COmmonsense Dataset Adversarially-authored by Humans (CODAH) is an evaluation set for commonsense question-answering in the sentence completion style of SWAG. As opposed to other automatically generated NLI datasets, CODAH is adversarially constructed by humans who can view feedback from a pre-trained model and use this information to design challenging commonsense questions. Our experimental results show that CODAH questions present a complementary extension to the SWAG dataset, testing additional modes of common sense.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 556
`'train'` | 1665
`'validation'` | 555

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "question_category": {
        "num_classes": 6,
        "names": [
            "Idioms",
            "Reference",
            "Polysemy",
            "Negation",
            "Quantitative",
            "Others"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "question_propmt": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "candidate_answers": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "correct_answer_idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    }
}
```
