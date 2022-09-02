# hendrycks_test

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/hendrycks_test)
- [Huggingface](https://huggingface.co/datasets/hendrycks_test)

## abstract_algebra

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/abstract_algebra')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 100
`'validation'` | 11

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## anatomy

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/anatomy')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 135
`'validation'` | 14

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## astronomy

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/astronomy')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 152
`'validation'` | 16

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## business_ethics

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/business_ethics')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 100
`'validation'` | 11

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## clinical_knowledge

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/clinical_knowledge')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 265
`'validation'` | 29

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## college_biology

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/college_biology')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 144
`'validation'` | 16

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## college_chemistry

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/college_chemistry')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 100
`'validation'` | 8

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## college_computer_science

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/college_computer_science')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 100
`'validation'` | 11

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## college_mathematics

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/college_mathematics')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 100
`'validation'` | 11

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## college_medicine

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/college_medicine')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 173
`'validation'` | 22

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## college_physics

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/college_physics')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 102
`'validation'` | 11

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## computer_security

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/computer_security')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 100
`'validation'` | 11

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## conceptual_physics

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/conceptual_physics')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 235
`'validation'` | 26

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## econometrics

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/econometrics')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 114
`'validation'` | 12

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## electrical_engineering

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/electrical_engineering')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 145
`'validation'` | 16

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## elementary_mathematics

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/elementary_mathematics')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 378
`'validation'` | 41

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## formal_logic

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/formal_logic')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 126
`'validation'` | 14

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## global_facts

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/global_facts')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 100
`'validation'` | 10

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## high_school_biology

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/high_school_biology')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 310
`'validation'` | 32

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## high_school_chemistry

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/high_school_chemistry')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 203
`'validation'` | 22

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## high_school_computer_science

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/high_school_computer_science')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 100
`'validation'` | 9

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## high_school_european_history

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/high_school_european_history')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 165
`'validation'` | 18

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## high_school_geography

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/high_school_geography')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 198
`'validation'` | 22

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## high_school_government_and_politics

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/high_school_government_and_politics')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 193
`'validation'` | 21

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## high_school_macroeconomics

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/high_school_macroeconomics')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 390
`'validation'` | 43

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## high_school_mathematics

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/high_school_mathematics')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 270
`'validation'` | 29

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## high_school_microeconomics

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/high_school_microeconomics')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 238
`'validation'` | 26

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## high_school_physics

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/high_school_physics')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 151
`'validation'` | 17

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## high_school_psychology

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/high_school_psychology')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 545
`'validation'` | 60

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## high_school_statistics

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/high_school_statistics')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 216
`'validation'` | 23

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## high_school_us_history

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/high_school_us_history')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 204
`'validation'` | 22

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## high_school_world_history

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/high_school_world_history')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 237
`'validation'` | 26

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## human_aging

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/human_aging')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 223
`'validation'` | 23

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## human_sexuality

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/human_sexuality')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 131
`'validation'` | 12

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## international_law

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/international_law')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 121
`'validation'` | 13

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## jurisprudence

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/jurisprudence')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 108
`'validation'` | 11

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## logical_fallacies

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/logical_fallacies')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 163
`'validation'` | 18

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## machine_learning

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/machine_learning')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 112
`'validation'` | 11

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## management

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/management')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 103
`'validation'` | 11

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## marketing

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/marketing')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 234
`'validation'` | 25

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## medical_genetics

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/medical_genetics')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 100
`'validation'` | 11

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## miscellaneous

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/miscellaneous')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 783
`'validation'` | 86

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## moral_disputes

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/moral_disputes')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 346
`'validation'` | 38

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## moral_scenarios

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/moral_scenarios')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 895
`'validation'` | 100

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## nutrition

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/nutrition')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 306
`'validation'` | 33

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## philosophy

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/philosophy')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 311
`'validation'` | 34

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## prehistory

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/prehistory')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 324
`'validation'` | 35

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## professional_accounting

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/professional_accounting')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 282
`'validation'` | 31

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## professional_law

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/professional_law')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 1534
`'validation'` | 170

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## professional_medicine

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/professional_medicine')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 272
`'validation'` | 31

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## professional_psychology

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/professional_psychology')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 612
`'validation'` | 69

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## public_relations

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/public_relations')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 110
`'validation'` | 12

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## security_studies

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/security_studies')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 245
`'validation'` | 27

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## sociology

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/sociology')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 201
`'validation'` | 22

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## us_foreign_policy

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/us_foreign_policy')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 100
`'validation'` | 11

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## virology

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/virology')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 166
`'validation'` | 18

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## world_religions

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hendrycks_test/world_religions')
```

- **说明**：

```
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'auxiliary_train'` | 99842
`'dev'` | 5
`'test'` | 171
`'validation'` | 19

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "choices": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "num_classes": 4,
        "names": [
            "A",
            "B",
            "C",
            "D"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
