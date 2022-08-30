# gsm8k

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/gsm8k)
- [Huggingface](https://huggingface.co/datasets/gsm8k)

## main

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gsm8k/main')
```

- **说明**：

```
GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality
linguistically diverse grade school math word problems. The
dataset was created to support the task of question answering
on basic mathematical problems that require multi-step reasoning.
```

- **许可**：MIT
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1319
`'train'` | 7473

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "answer": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## socratic

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gsm8k/socratic')
```

- **说明**：

```
GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality
linguistically diverse grade school math word problems. The
dataset was created to support the task of question answering
on basic mathematical problems that require multi-step reasoning.
```

- **许可**：MIT
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1319
`'train'` | 7473

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "answer": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
