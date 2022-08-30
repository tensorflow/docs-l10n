# boolq

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/boolq)
- [Huggingface](https://huggingface.co/datasets/boolq)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:boolq')
```

- **说明**：

```
BoolQ is a question answering dataset for yes/no questions containing 15942 examples. These questions are naturally
occurring ---they are generated in unprompted and unconstrained settings.
Each example is a triplet of (question, passage, answer), with the title of the page as optional additional context.
The text-pair classification setup is similar to existing natural language inference tasks.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 9427
`'validation'` | 3270

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "answer": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "passage": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
