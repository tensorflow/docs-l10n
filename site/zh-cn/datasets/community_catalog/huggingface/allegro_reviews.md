# allegro_reviews

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/allegro_reviews)
- [Huggingface](https://huggingface.co/datasets/allegro_reviews)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:allegro_reviews')
```

- **Description**:

```
Allegro Reviews is a sentiment analysis dataset, consisting of 11,588 product reviews written in Polish and extracted
from Allegro.pl - a popular e-commerce marketplace. Each review contains at least 50 words and has a rating on a scale
from one (negative review) to five (positive review).

We recommend using the provided train/dev/test split. The ratings for the test set reviews are kept hidden.
You can evaluate your model using the online evaluation tool available on klejbenchmark.com.
```

- **许可**：CC BY-SA 4.0
- **Version**: 1.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1006
`'train'` | 9577
`'validation'` | 1002

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "rating": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    }
}
```
