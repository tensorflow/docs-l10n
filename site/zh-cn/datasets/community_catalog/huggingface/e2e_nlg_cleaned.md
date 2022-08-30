# e2e_nlg_cleaned

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/e2e_nlg_cleaned)
- [Huggingface](https://huggingface.co/datasets/e2e_nlg_cleaned)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:e2e_nlg_cleaned')
```

- **Description**:

```
An update release of E2E NLG Challenge data with cleaned MRs and scripts, accompanying the following paper:

Ondřej Dušek, David M. Howcroft, and Verena Rieser (2019): Semantic Noise Matters for Neural Natural Language Generation. In INLG, Tokyo, Japan.
```

- **许可**：无已知许可
- **Version**: 0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 4693
`'train'` | 33525
`'validation'` | 4299

- **特征**：

```json
{
    "meaning_representation": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "human_reference": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
