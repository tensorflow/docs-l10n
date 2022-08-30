# hans

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/hans)
- [Huggingface](https://huggingface.co/datasets/hans)

## plain_text

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hans/plain_text')
```

- **说明**：

```
The HANS dataset is an NLI evaluation set that tests specific hypotheses about invalid heuristics that NLI models are likely to learn.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 30000
`'validation'` | 30000

- **特征**：

```json
{
    "premise": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "hypothesis": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 2,
        "names": [
            "entailment",
            "non-entailment"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "parse_premise": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "parse_hypothesis": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "binary_parse_premise": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "binary_parse_hypothesis": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "heuristic": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "subcase": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "template": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
