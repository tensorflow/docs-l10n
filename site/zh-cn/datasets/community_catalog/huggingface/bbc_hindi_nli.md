# bbc_hindi_nli

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/bbc_hindi_nli)
- [Huggingface](https://huggingface.co/datasets/bbc_hindi_nli)

## bbc hindi nli

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bbc_hindi_nli/bbc hindi nli')
```

- **说明**：

```
This dataset is used to train models for Natural Language Inference Tasks in Low-Resource Languages like Hindi.
```

- **许可**：MIT 许可

- **版本**：1.0.0

- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2592
`'train'` | 15552
`'validation'` | 2580

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
            "not-entailment",
            "entailment"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "topic": {
        "num_classes": 6,
        "names": [
            "india",
            "news",
            "international",
            "entertainment",
            "sport",
            "science"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
