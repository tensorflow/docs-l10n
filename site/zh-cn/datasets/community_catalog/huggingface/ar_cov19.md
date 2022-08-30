# ar_cov19

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/ar_cov19)
- [Huggingface](https://huggingface.co/datasets/ar_cov19)

## ar_cov19

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:ar_cov19/ar_cov19')
```

- **Description**:

```
ArCOV-19 is an Arabic COVID-19 Twitter dataset that covers the period from 27th of January till 30th of April 2020. ArCOV-19 is designed to enable research under several domains including natural language processing, information retrieval, and social computing, among others
```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1090591

- **特征**：

```json
{
    "tweetID": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    }
}
```
