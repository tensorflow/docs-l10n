# common_gen

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/common_gen)
- [Huggingface](https://huggingface.co/datasets/common_gen)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_gen')
```

- **说明**：

```
CommonGen is a constrained text generation task, associated with a benchmark dataset,
to explicitly test machines for the ability of generative commonsense reasoning. Given
a set of common concepts; the task is to generate a coherent sentence describing an
everyday scenario using these concepts.

CommonGen is challenging because it inherently requires 1) relational reasoning using
background commonsense knowledge, and 2) compositional generalization ability to work
on unseen concept combinations. Our dataset, constructed through a combination of
crowd-sourcing from AMT and existing caption corpora, consists of 30k concept-sets and
50k sentences in total.
```

- **许可**：无已知许可
- **版本**：2020.5.30
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1497
`'train'` | 67389
`'validation'` | 4018

- **特征**：

```json
{
    "concept_set_idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "concepts": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
