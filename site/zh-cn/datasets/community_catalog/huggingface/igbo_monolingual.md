# igbo_monolingual

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/igbo_monolingual)
- [Huggingface](https://huggingface.co/datasets/igbo_monolingual)

## eze_goes_to_school

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:igbo_monolingual/eze_goes_to_school')
```

- **说明**：

```
A dataset is a collection of Monolingual Igbo sentences.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1

- **特征**：

```json
{
    "format": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "chapters": {
        "feature": {
            "title": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "content": {
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

## bbc-igbo

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:igbo_monolingual/bbc-igbo')
```

- **说明**：

```
A dataset is a collection of Monolingual Igbo sentences.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1297

- **特征**：

```json
{
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "description": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "content": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "tags": {
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

## igbo-radio

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:igbo_monolingual/igbo-radio')
```

- **说明**：

```
A dataset is a collection of Monolingual Igbo sentences.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 440

- **特征**：

```json
{
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "author": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "description": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "content": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## jw-ot-igbo

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:igbo_monolingual/jw-ot-igbo')
```

- **说明**：

```
A dataset is a collection of Monolingual Igbo sentences.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 39

- **特征**：

```json
{
    "format": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "chapters": {
        "feature": {
            "title": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "content": {
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

## jw-nt-igbo

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:igbo_monolingual/jw-nt-igbo')
```

- **说明**：

```
A dataset is a collection of Monolingual Igbo sentences.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 27

- **特征**：

```json
{
    "format": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "chapters": {
        "feature": {
            "title": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "content": {
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

## jw-books

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:igbo_monolingual/jw-books')
```

- **说明**：

```
A dataset is a collection of Monolingual Igbo sentences.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 48

- **特征**：

```json
{
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "content": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "format": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## jw-teta

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:igbo_monolingual/jw-teta')
```

- **说明**：

```
A dataset is a collection of Monolingual Igbo sentences.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 37

- **特征**：

```json
{
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "content": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "format": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## jw-ulo_nche

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:igbo_monolingual/jw-ulo_nche')
```

- **说明**：

```
A dataset is a collection of Monolingual Igbo sentences.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 55

- **特征**：

```json
{
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "content": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "format": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## jw-ulo_nche_naamu

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:igbo_monolingual/jw-ulo_nche_naamu')
```

- **说明**：

```
A dataset is a collection of Monolingual Igbo sentences.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 88

- **特征**：

```json
{
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "content": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "format": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
