# google_wellformed_query

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/google_wellformed_query)
- [Huggingface](https://huggingface.co/datasets/google_wellformed_query)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:google_wellformed_query')
```

- **说明**：

```
Google's query wellformedness dataset was created by crowdsourcing well-formedness annotations for 25,100 queries from the Paralex corpus. Every query was annotated by five raters each with 1/0 rating of whether or not the query is well-formed.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 3850
`'train'` | 17500
`'validation'` | 3750

- **特征**：

```json
{
    "rating": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "content": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
