# break_data

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/break_data)
- [Huggingface](https://huggingface.co/datasets/break_data)

## QDMR-high-level

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:break_data/QDMR-high-level')
```

- **说明**：

```
Break is a human annotated dataset of natural language questions and their Question Decomposition Meaning Representations
(QDMRs). Break consists of 83,978 examples sampled from 10 question answering datasets over text, images and databases.
This repository contains the Break dataset along with information on the exact data format.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 3195
`'train'` | 17503
`'validation'` | 3130

- **特征**：

```json
{
    "question_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question_text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "decomposition": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "operators": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "split": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## QDMR-high-level-lexicon

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:break_data/QDMR-high-level-lexicon')
```

- **说明**：

```
Break is a human annotated dataset of natural language questions and their Question Decomposition Meaning Representations
(QDMRs). Break consists of 83,978 examples sampled from 10 question answering datasets over text, images and databases.
This repository contains the Break dataset along with information on the exact data format.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 3195
`'train'` | 17503
`'validation'` | 3130

- **特征**：

```json
{
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "allowed_tokens": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## QDMR

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:break_data/QDMR')
```

- **说明**：

```
Break is a human annotated dataset of natural language questions and their Question Decomposition Meaning Representations
(QDMRs). Break consists of 83,978 examples sampled from 10 question answering datasets over text, images and databases.
This repository contains the Break dataset along with information on the exact data format.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 8069
`'train'` | 44321
`'validation'` | 7760

- **特征**：

```json
{
    "question_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question_text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "decomposition": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "operators": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "split": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## QDMR-lexicon

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:break_data/QDMR-lexicon')
```

- **说明**：

```
Break is a human annotated dataset of natural language questions and their Question Decomposition Meaning Representations
(QDMRs). Break consists of 83,978 examples sampled from 10 question answering datasets over text, images and databases.
This repository contains the Break dataset along with information on the exact data format.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 8069
`'train'` | 44321
`'validation'` | 7760

- **特征**：

```json
{
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "allowed_tokens": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## logical-forms

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:break_data/logical-forms')
```

- **说明**：

```
Break is a human annotated dataset of natural language questions and their Question Decomposition Meaning Representations
(QDMRs). Break consists of 83,978 examples sampled from 10 question answering datasets over text, images and databases.
This repository contains the Break dataset along with information on the exact data format.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 8006
`'train'` | 44098
`'validation'` | 7719

- **特征**：

```json
{
    "question_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question_text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "decomposition": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "operators": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "split": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "program": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
