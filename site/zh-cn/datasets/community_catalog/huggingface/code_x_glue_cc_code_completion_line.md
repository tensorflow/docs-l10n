# code_x_glue_cc_code_completion_line

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/code_x_glue_cc_code_completion_line)
- [Huggingface](https://huggingface.co/datasets/code_x_glue_cc_code_completion_line)

## java

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:code_x_glue_cc_code_completion_line/java')
```

- **说明**：

```
CodeXGLUE CodeCompletion-line dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-line

Complete the unfinished line given previous context. Models are evaluated by exact match and edit similarity.
We propose line completion task to test model's ability to autocomplete a line. Majority code completion systems behave well in token level completion, but fail in completing an unfinished line like a method call with specific parameters, a function signature, a loop condition, a variable definition and so on. When a software develop finish one or more tokens of the current line, the line level completion model is expected to generate the entire line of syntactically correct code.
Line level code completion task shares the train/dev dataset with token level completion. After training a model on CodeCompletion-token, you could directly use it to test on line-level completion.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 3000

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "input": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gt": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## python

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:code_x_glue_cc_code_completion_line/python')
```

- **说明**：

```
CodeXGLUE CodeCompletion-line dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-line

Complete the unfinished line given previous context. Models are evaluated by exact match and edit similarity.
We propose line completion task to test model's ability to autocomplete a line. Majority code completion systems behave well in token level completion, but fail in completing an unfinished line like a method call with specific parameters, a function signature, a loop condition, a variable definition and so on. When a software develop finish one or more tokens of the current line, the line level completion model is expected to generate the entire line of syntactically correct code.
Line level code completion task shares the train/dev dataset with token level completion. After training a model on CodeCompletion-token, you could directly use it to test on line-level completion.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 10000

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "input": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gt": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
