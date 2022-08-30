# billsum

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/billsum)
- [Huggingface](https://huggingface.co/datasets/billsum)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:billsum')
```

- **说明**：

```
BillSum, summarization of US Congressional and California state bills.

There are several features:
  - text: bill text.
  - summary: summary of the bills.
  - title: title of the bills.
features for us bills. ca bills does not have.
  - text_len: number of chars in text.
  - sum_len: number of chars in summary.
```

- **许可**：无已知许可
- **版本**：3.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'ca_test'` | 1237
`'test'` | 3269
`'train'` | 18949

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "summary": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
