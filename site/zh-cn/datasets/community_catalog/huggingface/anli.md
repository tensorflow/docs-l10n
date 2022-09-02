# anli

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/anli)
- [Huggingface](https://huggingface.co/datasets/anli)

## plain_text

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:anli/plain_text')
```

- **说明**：

```
The Adversarial Natural Language Inference (ANLI) is a new large-scale NLI benchmark dataset,
The dataset is collected via an iterative, adversarial human-and-model-in-the-loop procedure.
ANLI is much more difficult than its predecessors including SNLI and MNLI.
It contains three rounds. Each round has train/dev/test splits.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'dev_r1'` | 1000
`'dev_r2'` | 1000
`'dev_r3'` | 1200
`'test_r1'` | 1000
`'test_r2'` | 1000
`'test_r3'` | 1200
`'train_r1'` | 16946
`'train_r2'` | 45460
`'train_r3'` | 100459

- **特征**：

```json
{
    "uid": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
        "num_classes": 3,
        "names": [
            "entailment",
            "neutral",
            "contradiction"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "reason": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
