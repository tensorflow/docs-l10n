# assin2

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/assin2)
- [Huggingface](https://huggingface.co/datasets/assin2)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:assin2')
```

- **说明**：

```
The ASSIN 2 corpus is composed of rather simple sentences. Following the procedures of SemEval 2014 Task 1.
The training and validation data are composed, respectively, of 6,500 and 500 sentence pairs in Brazilian Portuguese,
annotated for entailment and semantic similarity. Semantic similarity values range from 1 to 5, and text entailment
classes are either entailment or none. The test data are composed of approximately 3,000 sentence pairs with the same
annotation. All data were manually annotated.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2448
`'train'` | 6500
`'validation'` | 500

- **特征**：

```json
{
    "sentence_pair_id": {
        "dtype": "int64",
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
    "relatedness_score": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "entailment_judgment": {
        "num_classes": 2,
        "names": [
            "NONE",
            "ENTAILMENT"
        ],
        "id": null,
        "_type": "ClassLabel"
    }
}
```
