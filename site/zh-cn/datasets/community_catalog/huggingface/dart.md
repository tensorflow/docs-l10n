# dart

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/dart)
- [Huggingface](https://huggingface.co/datasets/dart)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:dart')
```

- **说明**：

```
DART is a large and open-domain structured DAta Record to Text generation corpus with high-quality
sentence annotations with each input being a set of entity-relation triples following a tree-structured ontology.
It consists of 82191 examples across different domains with each input being a semantic RDF triple set derived
from data records in tables and the tree ontology of table schema, annotated with sentence description that
covers all facts in the triple set.

DART is released in the following paper where you can find more details and baseline results:
https://arxiv.org/abs/2007.02871
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 6959
`'train'` | 30526
`'validation'` | 2768

- **特征**：

```json
{
    "tripleset": {
        "feature": {
            "feature": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "length": -1,
            "id": null,
            "_type": "Sequence"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "subtree_was_extended": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "annotations": {
        "feature": {
            "source": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "text": {
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
