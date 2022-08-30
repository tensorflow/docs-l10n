# drop

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/drop)
- [Huggingface](https://huggingface.co/datasets/drop)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:drop')
```

- **说明**：

```
DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs.
. DROP is a crowdsourced, adversarially-created, 96k-question benchmark, in which a system must resolve references in a
question, perhaps to multiple input positions, and perform discrete operations over them (such as addition, counting, or
 sorting). These operations require a much more comprehensive understanding of the content of paragraphs than what was
 necessary for prior datasets.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 77400
`'validation'` | 9535

- **特征**：

```json
{
    "section_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "query_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "passage": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "answers_spans": {
        "feature": {
            "spans": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "types": {
                "dtype": "string",
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
