# code_x_glue_cc_clone_detection_poj104

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/code_x_glue_cc_clone_detection_poj104)
- [Huggingface](https://huggingface.co/datasets/code_x_glue_cc_clone_detection_poj104)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:code_x_glue_cc_clone_detection_poj104')
```

- **说明**：

```
CodeXGLUE Clone-detection-POJ-104 dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-POJ-104

Given a code and a collection of candidates as the input, the task is to return Top K codes with the same semantic. Models are evaluated by MAP score.
We use POJ-104 dataset on this task.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 12000
`'train'` | 32000
`'validation'` | 8000

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "code": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
