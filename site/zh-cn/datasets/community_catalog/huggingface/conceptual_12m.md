# conceptual_12m

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/conceptual_12m)
- [Huggingface](https://huggingface.co/datasets/conceptual_12m)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:conceptual_12m')
```

- **Description**:

```
Conceptual 12M is a large-scale dataset of 12 million
image-text pairs specifically meant to be used for visionand-language pre-training.
Its data collection pipeline is a relaxed version of the one used in Conceptual Captions 3M.
```

- **许可**：数据集可以免费用于任何目的，但我们将感谢您承认 Google LLC（“Google”）作为数据源。数据集按“原样”提供，不提供任何明示或暗示的保证。对于因使用数据集而导致的任何直接或间接损害，Google 不承担任何责任。

- **Version**: 0.0.0

- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 12423374

- **特征**：

```json
{
    "image_url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "caption": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
