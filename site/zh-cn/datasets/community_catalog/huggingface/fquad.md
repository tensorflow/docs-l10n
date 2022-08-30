# fquad

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/fquad)
- [Huggingface](https://huggingface.co/datasets/fquad)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:fquad')
```

- **说明**：

```
FQuAD: French Question Answering Dataset
We introduce FQuAD, a native French Question Answering Dataset. FQuAD contains 25,000+ question and answer pairs.
Finetuning CamemBERT on FQuAD yields a F1 score of 88% and an exact match of 77.9%.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 4921
`'validation'` | 768

- **特征**：

```json
{
    "context": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "questions": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answers": {
        "feature": {
            "texts": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "answers_starts": {
                "dtype": "int32",
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
