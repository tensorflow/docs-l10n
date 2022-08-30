# app_reviews

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/app_reviews)
- [Huggingface](https://huggingface.co/datasets/app_reviews)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:app_reviews')
```

- **说明**：

```
It is a large dataset of Android applications belonging to 23 differentapps categories, which provides an overview of the types of feedback users report on the apps and documents the evolution of the related code metrics. The dataset contains about 395 applications of the F-Droid repository, including around 600 versions, 280,000 user reviews (extracted with specific text mining approaches)
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 288065

- **特征**：

```json
{
    "package_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star": {
        "dtype": "int8",
        "id": null,
        "_type": "Value"
    }
}
```
