# arabic_pos_dialect

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/arabic_pos_dialect)
- [Huggingface](https://huggingface.co/datasets/arabic_pos_dialect)

## egy

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:arabic_pos_dialect/egy')
```

- **说明**：

```
The Dialectal Arabic Datasets contain four dialects of Arabic, Etyptian (EGY), Levantine (LEV), Gulf (GLF), and Maghrebi (MGR). Each dataset consists of a set of 350 manually segmented and POS tagged tweets.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 350

- **特征**：

```json
{
    "fold": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "subfold": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "words": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "segments": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "pos_tags": {
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

## lev

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:arabic_pos_dialect/lev')
```

- **说明**：

```
The Dialectal Arabic Datasets contain four dialects of Arabic, Etyptian (EGY), Levantine (LEV), Gulf (GLF), and Maghrebi (MGR). Each dataset consists of a set of 350 manually segmented and POS tagged tweets.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 350

- **特征**：

```json
{
    "fold": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "subfold": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "words": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "segments": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "pos_tags": {
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

## glf

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:arabic_pos_dialect/glf')
```

- **说明**：

```
The Dialectal Arabic Datasets contain four dialects of Arabic, Etyptian (EGY), Levantine (LEV), Gulf (GLF), and Maghrebi (MGR). Each dataset consists of a set of 350 manually segmented and POS tagged tweets.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 350

- **特征**：

```json
{
    "fold": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "subfold": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "words": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "segments": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "pos_tags": {
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

## mgr

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:arabic_pos_dialect/mgr')
```

- **说明**：

```
The Dialectal Arabic Datasets contain four dialects of Arabic, Etyptian (EGY), Levantine (LEV), Gulf (GLF), and Maghrebi (MGR). Each dataset consists of a set of 350 manually segmented and POS tagged tweets.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 350

- **特征**：

```json
{
    "fold": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "subfold": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "words": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "segments": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "pos_tags": {
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
