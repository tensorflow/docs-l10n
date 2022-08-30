# bprec

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/bprec)
- [Huggingface](https://huggingface.co/datasets/bprec)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bprec')
```

- **说明**：

```
Dataset consisting of Polish language texts annotated to recognize brand-product relations.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'banking'` | 561
`'cosmetics'` | 2384
`'electro'` | 382
`'tele'` | 2391

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "ner": {
        "feature": {
            "source": {
                "from": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "to": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "type": {
                    "num_classes": 10,
                    "names": [
                        "PRODUCT_NAME",
                        "PRODUCT_NAME_IMP",
                        "PRODUCT_NO_BRAND",
                        "BRAND_NAME",
                        "BRAND_NAME_IMP",
                        "VERSION",
                        "PRODUCT_ADJ",
                        "BRAND_ADJ",
                        "LOCATION",
                        "LOCATION_IMP"
                    ],
                    "names_file": null,
                    "id": null,
                    "_type": "ClassLabel"
                }
            },
            "target": {
                "from": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "to": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "type": {
                    "num_classes": 10,
                    "names": [
                        "PRODUCT_NAME",
                        "PRODUCT_NAME_IMP",
                        "PRODUCT_NO_BRAND",
                        "BRAND_NAME",
                        "BRAND_NAME_IMP",
                        "VERSION",
                        "PRODUCT_ADJ",
                        "BRAND_ADJ",
                        "LOCATION",
                        "LOCATION_IMP"
                    ],
                    "names_file": null,
                    "id": null,
                    "_type": "ClassLabel"
                }
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## all

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bprec/all')
```

- **说明**：

```
Dataset consisting of Polish language texts annotated to recognize brand-product relations.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 5718

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "ner": {
        "feature": {
            "source": {
                "from": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "to": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "type": {
                    "num_classes": 10,
                    "names": [
                        "PRODUCT_NAME",
                        "PRODUCT_NAME_IMP",
                        "PRODUCT_NO_BRAND",
                        "BRAND_NAME",
                        "BRAND_NAME_IMP",
                        "VERSION",
                        "PRODUCT_ADJ",
                        "BRAND_ADJ",
                        "LOCATION",
                        "LOCATION_IMP"
                    ],
                    "names_file": null,
                    "id": null,
                    "_type": "ClassLabel"
                }
            },
            "target": {
                "from": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "to": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "type": {
                    "num_classes": 10,
                    "names": [
                        "PRODUCT_NAME",
                        "PRODUCT_NAME_IMP",
                        "PRODUCT_NO_BRAND",
                        "BRAND_NAME",
                        "BRAND_NAME_IMP",
                        "VERSION",
                        "PRODUCT_ADJ",
                        "BRAND_ADJ",
                        "LOCATION",
                        "LOCATION_IMP"
                    ],
                    "names_file": null,
                    "id": null,
                    "_type": "ClassLabel"
                }
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## tele

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bprec/tele')
```

- **说明**：

```
Dataset consisting of Polish language texts annotated to recognize brand-product relations.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2391

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "ner": {
        "feature": {
            "source": {
                "from": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "to": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "type": {
                    "num_classes": 10,
                    "names": [
                        "PRODUCT_NAME",
                        "PRODUCT_NAME_IMP",
                        "PRODUCT_NO_BRAND",
                        "BRAND_NAME",
                        "BRAND_NAME_IMP",
                        "VERSION",
                        "PRODUCT_ADJ",
                        "BRAND_ADJ",
                        "LOCATION",
                        "LOCATION_IMP"
                    ],
                    "names_file": null,
                    "id": null,
                    "_type": "ClassLabel"
                }
            },
            "target": {
                "from": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "to": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "type": {
                    "num_classes": 10,
                    "names": [
                        "PRODUCT_NAME",
                        "PRODUCT_NAME_IMP",
                        "PRODUCT_NO_BRAND",
                        "BRAND_NAME",
                        "BRAND_NAME_IMP",
                        "VERSION",
                        "PRODUCT_ADJ",
                        "BRAND_ADJ",
                        "LOCATION",
                        "LOCATION_IMP"
                    ],
                    "names_file": null,
                    "id": null,
                    "_type": "ClassLabel"
                }
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## electro

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bprec/electro')
```

- **说明**：

```
Dataset consisting of Polish language texts annotated to recognize brand-product relations.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 382

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "ner": {
        "feature": {
            "source": {
                "from": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "to": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "type": {
                    "num_classes": 10,
                    "names": [
                        "PRODUCT_NAME",
                        "PRODUCT_NAME_IMP",
                        "PRODUCT_NO_BRAND",
                        "BRAND_NAME",
                        "BRAND_NAME_IMP",
                        "VERSION",
                        "PRODUCT_ADJ",
                        "BRAND_ADJ",
                        "LOCATION",
                        "LOCATION_IMP"
                    ],
                    "names_file": null,
                    "id": null,
                    "_type": "ClassLabel"
                }
            },
            "target": {
                "from": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "to": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "type": {
                    "num_classes": 10,
                    "names": [
                        "PRODUCT_NAME",
                        "PRODUCT_NAME_IMP",
                        "PRODUCT_NO_BRAND",
                        "BRAND_NAME",
                        "BRAND_NAME_IMP",
                        "VERSION",
                        "PRODUCT_ADJ",
                        "BRAND_ADJ",
                        "LOCATION",
                        "LOCATION_IMP"
                    ],
                    "names_file": null,
                    "id": null,
                    "_type": "ClassLabel"
                }
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## cosmetics

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bprec/cosmetics')
```

- **说明**：

```
Dataset consisting of Polish language texts annotated to recognize brand-product relations.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2384

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "ner": {
        "feature": {
            "source": {
                "from": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "to": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "type": {
                    "num_classes": 10,
                    "names": [
                        "PRODUCT_NAME",
                        "PRODUCT_NAME_IMP",
                        "PRODUCT_NO_BRAND",
                        "BRAND_NAME",
                        "BRAND_NAME_IMP",
                        "VERSION",
                        "PRODUCT_ADJ",
                        "BRAND_ADJ",
                        "LOCATION",
                        "LOCATION_IMP"
                    ],
                    "names_file": null,
                    "id": null,
                    "_type": "ClassLabel"
                }
            },
            "target": {
                "from": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "to": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "type": {
                    "num_classes": 10,
                    "names": [
                        "PRODUCT_NAME",
                        "PRODUCT_NAME_IMP",
                        "PRODUCT_NO_BRAND",
                        "BRAND_NAME",
                        "BRAND_NAME_IMP",
                        "VERSION",
                        "PRODUCT_ADJ",
                        "BRAND_ADJ",
                        "LOCATION",
                        "LOCATION_IMP"
                    ],
                    "names_file": null,
                    "id": null,
                    "_type": "ClassLabel"
                }
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## banking

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bprec/banking')
```

- **说明**：

```
Dataset consisting of Polish language texts annotated to recognize brand-product relations.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 561

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "ner": {
        "feature": {
            "source": {
                "from": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "to": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "type": {
                    "num_classes": 10,
                    "names": [
                        "PRODUCT_NAME",
                        "PRODUCT_NAME_IMP",
                        "PRODUCT_NO_BRAND",
                        "BRAND_NAME",
                        "BRAND_NAME_IMP",
                        "VERSION",
                        "PRODUCT_ADJ",
                        "BRAND_ADJ",
                        "LOCATION",
                        "LOCATION_IMP"
                    ],
                    "names_file": null,
                    "id": null,
                    "_type": "ClassLabel"
                }
            },
            "target": {
                "from": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "text": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "to": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "type": {
                    "num_classes": 10,
                    "names": [
                        "PRODUCT_NAME",
                        "PRODUCT_NAME_IMP",
                        "PRODUCT_NO_BRAND",
                        "BRAND_NAME",
                        "BRAND_NAME_IMP",
                        "VERSION",
                        "PRODUCT_ADJ",
                        "BRAND_ADJ",
                        "LOCATION",
                        "LOCATION_IMP"
                    ],
                    "names_file": null,
                    "id": null,
                    "_type": "ClassLabel"
                }
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```
