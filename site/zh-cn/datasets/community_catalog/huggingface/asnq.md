# asnq

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/asnq)
- [Huggingface](https://huggingface.co/datasets/asnq)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:asnq')
```

- **说明**：

```
ASNQ is a dataset for answer sentence selection derived from
Google's Natural Questions (NQ) dataset (Kwiatkowski et al. 2019).

Each example contains a question, candidate sentence, label indicating whether or not
the sentence answers the question, and two additional features --
sentence_in_long_answer and short_answer_in_sentence indicating whether ot not the
candidate sentence is contained in the long_answer and if the short_answer is in the candidate sentence.

For more details please see
https://arxiv.org/pdf/1911.04118.pdf

and

https://research.google/pubs/pub47761/
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 20377568
`'validation'` | 930062

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 2,
        "names": [
            "neg",
            "pos"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "sentence_in_long_answer": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "short_answer_in_sentence": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    }
}
```
