# cfq

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/cfq)
- [Huggingface](https://huggingface.co/datasets/cfq)

## mcd1

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cfq/mcd1')
```

- **说明**：

```
The CFQ dataset (and it's splits) for measuring compositional generalization.

See https://arxiv.org/abs/1912.09713.pdf for background.

Example usage:
data = datasets.load_dataset('cfq/mcd1')
```

- **许可**：无已知许可
- **版本**：1.0.1
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 11968
`'train'` | 95743

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "query": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## mcd2

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cfq/mcd2')
```

- **说明**：

```
The CFQ dataset (and it's splits) for measuring compositional generalization.

See https://arxiv.org/abs/1912.09713.pdf for background.

Example usage:
data = datasets.load_dataset('cfq/mcd1')
```

- **许可**：无已知许可
- **版本**：1.0.1
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 11968
`'train'` | 95743

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "query": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## mcd3

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cfq/mcd3')
```

- **说明**：

```
The CFQ dataset (and it's splits) for measuring compositional generalization.

See https://arxiv.org/abs/1912.09713.pdf for background.

Example usage:
data = datasets.load_dataset('cfq/mcd1')
```

- **许可**：无已知许可
- **版本**：1.0.1
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 11968
`'train'` | 95743

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "query": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## question_complexity_split

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cfq/question_complexity_split')
```

- **说明**：

```
The CFQ dataset (and it's splits) for measuring compositional generalization.

See https://arxiv.org/abs/1912.09713.pdf for background.

Example usage:
data = datasets.load_dataset('cfq/mcd1')
```

- **许可**：无已知许可
- **版本**：1.0.1
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 10340
`'train'` | 98999

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "query": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## question_pattern_split

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cfq/question_pattern_split')
```

- **说明**：

```
The CFQ dataset (and it's splits) for measuring compositional generalization.

See https://arxiv.org/abs/1912.09713.pdf for background.

Example usage:
data = datasets.load_dataset('cfq/mcd1')
```

- **许可**：无已知许可
- **版本**：1.0.1
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 11909
`'train'` | 95654

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "query": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## query_complexity_split

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cfq/query_complexity_split')
```

- **说明**：

```
The CFQ dataset (and it's splits) for measuring compositional generalization.

See https://arxiv.org/abs/1912.09713.pdf for background.

Example usage:
data = datasets.load_dataset('cfq/mcd1')
```

- **许可**：无已知许可
- **版本**：1.0.1
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 9512
`'train'` | 100654

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "query": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## query_pattern_split

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cfq/query_pattern_split')
```

- **说明**：

```
The CFQ dataset (and it's splits) for measuring compositional generalization.

See https://arxiv.org/abs/1912.09713.pdf for background.

Example usage:
data = datasets.load_dataset('cfq/mcd1')
```

- **许可**：无已知许可
- **版本**：1.0.1
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 12589
`'train'` | 94600

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "query": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## random_split

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cfq/random_split')
```

- **说明**：

```
The CFQ dataset (and it's splits) for measuring compositional generalization.

See https://arxiv.org/abs/1912.09713.pdf for background.

Example usage:
data = datasets.load_dataset('cfq/mcd1')
```

- **许可**：无已知许可
- **版本**：1.0.1
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 11967
`'train'` | 95744

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "query": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
