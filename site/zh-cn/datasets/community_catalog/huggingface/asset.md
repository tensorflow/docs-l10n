# asset

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/asset)
- [Huggingface](https://huggingface.co/datasets/asset)

## simplification

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:asset/simplification')
```

- **说明**：

```
ASSET is a dataset for evaluating Sentence Simplification systems with multiple rewriting transformations,
as described in "ASSET: A Dataset for Tuning and Evaluation of Sentence Simplification Models with Multiple Rewriting Transformations".
The corpus is composed of 2000 validation and 359 test original sentences that were each simplified 10 times by different annotators.
The corpus also contains human judgments of meaning preservation, fluency and simplicity for the outputs of several automatic text simplification systems.
```

- **许可**：Creative Common Attribution-NonCommercial 4.0 International
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 359
`'validation'` | 2000

- **特征**：

```json
{
    "original": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "simplifications": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## ratings

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:asset/ratings')
```

- **说明**：

```
ASSET is a dataset for evaluating Sentence Simplification systems with multiple rewriting transformations,
as described in "ASSET: A Dataset for Tuning and Evaluation of Sentence Simplification Models with Multiple Rewriting Transformations".
The corpus is composed of 2000 validation and 359 test original sentences that were each simplified 10 times by different annotators.
The corpus also contains human judgments of meaning preservation, fluency and simplicity for the outputs of several automatic text simplification systems.
```

- **许可**：Creative Common Attribution-NonCommercial 4.0 International
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'full'` | 4500

- **特征**：

```json
{
    "original": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "simplification": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "original_sentence_id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "aspect": {
        "num_classes": 3,
        "names": [
            "meaning",
            "fluency",
            "simplicity"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "worker_id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    }
}
```
