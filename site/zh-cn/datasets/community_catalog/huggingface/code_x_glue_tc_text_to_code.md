# code_x_glue_tc_text_to_code

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/code_x_glue_tc_text_to_code)
- [Huggingface](https://huggingface.co/datasets/code_x_glue_tc_text_to_code)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:code_x_glue_tc_text_to_code')
```

- **Description**:

```
CodeXGLUE text-to-code dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code

We use concode dataset which is a widely used code generation dataset from Iyer's EMNLP 2018 paper Mapping Language to Code in Programmatic Context. See paper for details.
```

- **许可**：无已知许可
- **Version**: 0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2000
`'train'` | 100000
`'validation'` | 2000

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "nl": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "code": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
