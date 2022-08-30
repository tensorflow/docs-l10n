# cbt

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/cbt)
- [Huggingface](https://huggingface.co/datasets/cbt)

## raw

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cbt/raw')
```

- **说明**：

```
The Children’s Book Test (CBT) is designed to measure directly
how well language models can exploit wider linguistic context.
The CBT is built from books that are freely available.
```

- **许可**：GNU Free Documentation License v1.3
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 5
`'train'` | 98
`'validation'` | 5

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
    }
}
```

## V

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cbt/V')
```

- **说明**：

```
The Children’s Book Test (CBT) is designed to measure directly
how well language models can exploit wider linguistic context.
The CBT is built from books that are freely available.
```

- **许可**：GNU Free Documentation License v1.3
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2500
`'train'` | 105825
`'validation'` | 2000

- **特征**：

```json
{
    "sentences": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
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
    },
    "options": {
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

## P

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cbt/P')
```

- **说明**：

```
The Children’s Book Test (CBT) is designed to measure directly
how well language models can exploit wider linguistic context.
The CBT is built from books that are freely available.
```

- **许可**：GNU Free Documentation License v1.3
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2500
`'train'` | 334030
`'validation'` | 2000

- **特征**：

```json
{
    "sentences": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
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
    },
    "options": {
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

## NE

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cbt/NE')
```

- **说明**：

```
The Children’s Book Test (CBT) is designed to measure directly
how well language models can exploit wider linguistic context.
The CBT is built from books that are freely available.
```

- **许可**：GNU Free Documentation License v1.3
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2500
`'train'` | 108719
`'validation'` | 2000

- **特征**：

```json
{
    "sentences": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
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
    },
    "options": {
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

## CN

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cbt/CN')
```

- **说明**：

```
The Children’s Book Test (CBT) is designed to measure directly
how well language models can exploit wider linguistic context.
The CBT is built from books that are freely available.
```

- **许可**：GNU Free Documentation License v1.3
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2500
`'train'` | 120769
`'validation'` | 2000

- **特征**：

```json
{
    "sentences": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
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
    },
    "options": {
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
