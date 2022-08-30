# dbrd

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/dbrd)
- [Huggingface](https://huggingface.co/datasets/dbrd)

## plain_text

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:dbrd/plain_text')
```

- **说明**：

```
The Dutch Book Review Dataset (DBRD) contains over 110k book reviews of which 22k have associated binary sentiment polarity labels. It is intended as a benchmark for sentiment classification in Dutch and created due to a lack of annotated datasets in Dutch that are suitable for this task.
```

- **许可**：无已知许可
- **版本**：3.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2224
`'train'` | 20028
`'unsupervised'` | 96264

- **特征**：

```json
{
    "text": {
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
