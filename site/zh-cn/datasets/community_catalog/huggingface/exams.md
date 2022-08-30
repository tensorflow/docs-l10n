# exams

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/exams)
- [Huggingface](https://huggingface.co/datasets/exams)

## alignments

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/alignments')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'full'` | 10834

- **特征**：

```json
{
    "source_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target_id_list": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## multilingual

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/multilingual')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 13510
`'train'` | 7961
`'validation'` | 2672

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## multilingual_with_para

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/multilingual_with_para')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 13510
`'train'` | 7961
`'validation'` | 2672

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_test

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_test')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 19736

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_with_para_test

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_with_para_test')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 19736

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_bg

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_bg')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2344
`'validation'` | 593

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_with_para_bg

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_with_para_bg')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2344
`'validation'` | 593

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_hr

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_hr')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2341
`'validation'` | 538

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_with_para_hr

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_with_para_hr')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2341
`'validation'` | 538

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_hu

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_hu')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1731
`'validation'` | 536

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_with_para_hu

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_with_para_hu')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1731
`'validation'` | 536

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_it

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_it')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1010
`'validation'` | 246

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_with_para_it

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_with_para_it')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1010
`'validation'` | 246

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_mk

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_mk')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1665
`'validation'` | 410

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_with_para_mk

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_with_para_mk')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1665
`'validation'` | 410

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_pl

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_pl')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1577
`'validation'` | 394

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_with_para_pl

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_with_para_pl')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1577
`'validation'` | 394

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_pt

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_pt')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 740
`'validation'` | 184

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_with_para_pt

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_with_para_pt')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 740
`'validation'` | 184

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_sq

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_sq')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1194
`'validation'` | 311

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_with_para_sq

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_with_para_sq')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1194
`'validation'` | 311

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_sr

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_sr')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1323
`'validation'` | 314

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_with_para_sr

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_with_para_sr')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1323
`'validation'` | 314

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_tr

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_tr')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1571
`'validation'` | 393

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_with_para_tr

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_with_para_tr')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1571
`'validation'` | 393

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_vi

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_vi')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1955
`'validation'` | 488

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## crosslingual_with_para_vi

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:exams/crosslingual_with_para_vi')
```

- **说明**：

```
EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations.
It consists of more than 24,000 high-quality high school exam questions in 16 languages,
covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1955
`'validation'` | 488

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "stem": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "choices": {
            "feature": {
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "label": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "para": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        }
    },
    "answerKey": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "info": {
        "grade": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "subject": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "language": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```
