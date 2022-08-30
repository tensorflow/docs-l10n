# eth_py150_open

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/eth_py150_open)
- [Huggingface](https://huggingface.co/datasets/eth_py150_open)

## eth_py150_open

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:eth_py150_open/eth_py150_open')
```

- **Description**:

```
A redistributable subset of the ETH Py150 corpus, introduced in the ICML 2020 paper 'Learning and Evaluating Contextual Embedding of Source Code'
```

- **许可**：Apache License, Version 2.0
- **Version**: 1.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 41457
`'train'` | 74749
`'validation'` | 8302

- **特征**：

```json
{
    "filepath": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "license": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
