# cs_restaurants

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/cs_restaurants)
- [Huggingface](https://huggingface.co/datasets/cs_restaurants)

## CSRestaurants

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cs_restaurants/CSRestaurants')
```

- **说明**：

```
This is a dataset for NLG in task-oriented spoken dialogue systems with Czech as the target language. It originated as
a translation of the English San Francisco Restaurants dataset by Wen et al. (2015).
```

- **许可**：Creative Commons 4.0 BY-SA
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 842
`'train'` | 3569
`'validation'` | 781

- **特征**：

```json
{
    "dialogue_act": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "delexicalized_dialogue_act": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "delexicalized_text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
