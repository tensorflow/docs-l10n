# babi_qa

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/babi_qa)
- [Huggingface](https://huggingface.co/datasets/babi_qa)

## en-qa1

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa1')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-qa2

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa2')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-qa3

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa3')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-qa4

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa4')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**:

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 1000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-qa5

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa5')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-qa6

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa6')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-qa7

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa7')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-qa8

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa8')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-qa9

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa9')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-qa10

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa10')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-qa11

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa11')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-qa12

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa12')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-qa13

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa13')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-qa14

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa14')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-qa15

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa15')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 250
`'train'` | 250

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-qa16

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa16')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 1000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-qa17

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa17')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 125
`'train'` | 125

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-qa18

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa18')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 199
`'train'` | 198

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-qa19

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa19')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 1000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-qa20

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-qa20')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 93
`'train'` | 94

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa1

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa1')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa2

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa2')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa3

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa3')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 167
`'train'` | 167

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa4

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa4')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 1000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa5

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa5')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa6

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa6')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa7

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa7')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa8

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa8')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa9

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa9')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa10

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa10')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa11

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa11')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa12

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa12')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa13

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa13')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 125
`'train'` | 125

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa14

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa14')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa15

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa15')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 250
`'train'` | 250

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa16

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa16')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 1000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa17

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa17')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 125
`'train'` | 125

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa18

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa18')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 198
`'train'` | 198

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa19

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa19')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 1000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-qa20

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-qa20')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 94
`'train'` | 93

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa1

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa1')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa2

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa2')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa3

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa3')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa4

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa4')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 10000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa5

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa5')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa6

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa6')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa7

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa7')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa8

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa8')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa9

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa9')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa10

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa10')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa11

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa11')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa12

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa12')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa13

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa13')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa14

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa14')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa15

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa15')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 250
`'train'` | 2500

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa16

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa16')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 10000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa17

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa17')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 125
`'train'` | 1250

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa18

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa18')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 199
`'train'` | 1978

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa19

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa19')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 10000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-10k-qa20

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-10k-qa20')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 93
`'train'` | 933

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa1

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa1')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 180
`'validation'` | 20

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa2

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa2')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 180
`'validation'` | 20

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa3

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa3')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 180
`'validation'` | 20

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa4

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa4')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 900
`'validation'` | 100

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa5

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa5')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 180
`'validation'` | 20

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa6

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa6')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 180
`'validation'` | 20

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa7

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa7')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 180
`'validation'` | 20

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa8

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa8')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 180
`'validation'` | 20

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa9

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa9')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 180
`'validation'` | 20

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa10

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa10')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 180
`'validation'` | 20

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa11

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa11')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 180
`'validation'` | 20

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa12

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa12')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 180
`'validation'` | 20

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa13

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa13')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 180
`'validation'` | 20

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa14

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa14')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 180
`'validation'` | 20

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa15

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa15')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 250
`'train'` | 225
`'validation'` | 25

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa16

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa16')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 900
`'validation'` | 100

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa17

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa17')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 125
`'train'` | 113
`'validation'` | 12

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa18

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa18')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 199
`'train'` | 179
`'validation'` | 19

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa19

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa19')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 900
`'validation'` | 100

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-qa20

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-qa20')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 93
`'train'` | 85
`'validation'` | 9

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa1

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa1')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 1800
`'validation'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa2

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa2')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 1800
`'validation'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa3

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa3')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 1800
`'validation'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa4

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa4')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 9000
`'validation'` | 1000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa5

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa5')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 1800
`'validation'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa6

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa6')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 1800
`'validation'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa7

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa7')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 1800
`'validation'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa8

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa8')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 1800
`'validation'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa9

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa9')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 1800
`'validation'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa10

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa10')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 1800
`'validation'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa11

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa11')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 1800
`'validation'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa12

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa12')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 1800
`'validation'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa13

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa13')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 1800
`'validation'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa14

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa14')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 1800
`'validation'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa15

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa15')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 250
`'train'` | 2250
`'validation'` | 250

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa16

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa16')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 9000
`'validation'` | 1000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa17

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa17')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 125
`'train'` | 1125
`'validation'` | 125

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa18

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa18')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 199
`'train'` | 1781
`'validation'` | 197

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa19

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa19')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 9000
`'validation'` | 1000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## en-valid-10k-qa20

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/en-valid-10k-qa20')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 93
`'train'` | 840
`'validation'` | 93

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa1

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa1')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa2

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa2')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa3

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa3')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 167
`'train'` | 1667

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa4

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa4')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 10000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa5

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa5')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa6

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa6')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa7

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa7')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa8

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa8')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa9

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa9')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa10

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa10')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa11

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa11')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa12

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa12')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa13

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa13')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 125
`'train'` | 1250

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa14

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa14')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa15

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa15')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 250
`'train'` | 2500

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa16

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa16')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 10000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa17

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa17')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 125
`'train'` | 1250

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa18

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa18')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 198
`'train'` | 1977

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa19

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa19')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 10000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## hn-10k-qa20

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/hn-10k-qa20')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 94
`'train'` | 934

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa1

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa1')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa2

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa2')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa3

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa3')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa4

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa4')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 1000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa5

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa5')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa6

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa6')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa7

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa7')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa8

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa8')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa9

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa9')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa10

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa10')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa11

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa11')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa12

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa12')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa13

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa13')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa14

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa14')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 200

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa15

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa15')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 250
`'train'` | 250

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa16

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa16')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 1000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa17

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa17')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 125
`'train'` | 125

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa18

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa18')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 199
`'train'` | 198

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa19

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa19')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 1000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-qa20

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-qa20')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 93
`'train'` | 94

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa1

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa1')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa2

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa2')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa3

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa3')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa4

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa4')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 10000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa5

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa5')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa6

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa6')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa7

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa7')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa8

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa8')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa9

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa9')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa10

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa10')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa11

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa11')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa12

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa12')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa13

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa13')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa14

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa14')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 200
`'train'` | 2000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa15

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa15')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 250
`'train'` | 2500

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa16

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa16')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 10000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa17

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa17')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 125
`'train'` | 1250

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa18

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa18')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 199
`'train'` | 1978

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa19

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa19')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 10000

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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

## shuffled-10k-qa20

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:babi_qa/shuffled-10k-qa20')
```

- **说明**：

```
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
```

- **许可**：Creative Commons Attribution 3.0 License
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 93
`'train'` | 933

- **特征**：

```json
{
    "story": {
        "feature": {
            "id": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "type": {
                "num_classes": 2,
                "names": [
                    "context",
                    "question"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "supporting_ids": {
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
