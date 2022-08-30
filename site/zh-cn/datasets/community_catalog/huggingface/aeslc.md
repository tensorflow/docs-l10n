# aeslc

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/aeslc)
- [Huggingface](https://huggingface.co/datasets/aeslc)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:aeslc')
```

- **Description**:

```
A collection of email messages of employees in the Enron Corporation.

There are two features:
  - email_body: email body text.
  - subject_line: email subject text.
```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1906
`'train'` | 14436
`'validation'` | 1960

- **特征**：

```json
{
    "email_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "subject_line": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
