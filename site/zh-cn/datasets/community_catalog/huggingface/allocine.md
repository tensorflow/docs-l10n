# allocine

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/allocine)
- [Huggingface](https://huggingface.co/datasets/allocine)

## allocine

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:allocine/allocine')
```

- **说明**：

```
Allocine Dataset: A Large-Scale French Movie Reviews Dataset.
 This is a dataset for binary sentiment classification, made of user reviews scraped from Allocine.fr.
 It contains 100k positive and 100k negative reviews divided into 3 balanced splits: train (160k reviews), val (20k) and test (20k).
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 20000
`'train'` | 160000
`'validation'` | 20000

- **特征**：

```json
{
    "review": {
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
    }
}
```
