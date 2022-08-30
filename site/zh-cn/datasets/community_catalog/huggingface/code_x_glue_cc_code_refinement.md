# code_x_glue_cc_code_refinement

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/code_x_glue_cc_code_refinement)
- [Huggingface](https://huggingface.co/datasets/code_x_glue_cc_code_refinement)

## medium

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:code_x_glue_cc_code_refinement/medium')
```

- **说明**：

```
CodeXGLUE code-refinement dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-refinement

We use the dataset released by this paper(https://arxiv.org/pdf/1812.08693.pdf). The source side is a Java function with bugs and the target side is the refined one. All the function and variable names are normalized. Their dataset contains two subsets ( i.e.small and medium) based on the function length.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 6545
`'train'` | 52364
`'validation'` | 6546

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "buggy": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "fixed": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## small

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:code_x_glue_cc_code_refinement/small')
```

- **说明**：

```
CodeXGLUE code-refinement dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-refinement

We use the dataset released by this paper(https://arxiv.org/pdf/1812.08693.pdf). The source side is a Java function with bugs and the target side is the refined one. All the function and variable names are normalized. Their dataset contains two subsets ( i.e.small and medium) based on the function length.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 5835
`'train'` | 46680
`'validation'` | 5835

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "buggy": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "fixed": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
