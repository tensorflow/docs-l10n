# glucose

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/glucose)
- [Huggingface](https://huggingface.co/datasets/glucose)

## glucose

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:glucose/glucose')
```

- **说明**：

```
When humans read or listen, they make implicit commonsense inferences that frame their understanding of what happened and why. As a step toward AI systems that can build similar mental models, we introduce GLUCOSE, a large-scale dataset of implicit commonsense causal knowledge, encoded as causal mini-theories about the world, each grounded in a narrative context.
```

- **许可**：Creative Commons Attribution-NonCommercial 4.0 International Public License
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 500
`'train'` | 65522

- **特征**：

```json
{
    "experiment_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "story_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "worker_id": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "worker_ids": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "submission_time_normalized": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "worker_quality_assessment": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "selected_sentence_index": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "story": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "selected_sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "number_filled_in": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "1_specificNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "1_specificStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "1_generalNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "1_generalStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "2_specificNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "2_specificStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "2_generalNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "2_generalStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "3_specificNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "3_specificStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "3_generalNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "3_generalStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "4_specificNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "4_specificStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "4_generalNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "4_generalStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "5_specificNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "5_specificStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "5_generalNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "5_generalStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "6_specificNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "6_specificStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "6_generalNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "6_generalStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "7_specificNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "7_specificStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "7_generalNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "7_generalStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "8_specificNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "8_specificStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "8_generalNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "8_generalStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "9_specificNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "9_specificStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "9_generalNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "9_generalStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "10_specificNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "10_specificStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "10_generalNL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "10_generalStructured": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
