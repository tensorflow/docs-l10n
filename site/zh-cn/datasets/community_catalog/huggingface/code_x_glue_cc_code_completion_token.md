# code_x_glue_cc_code_completion_token

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/code_x_glue_cc_code_completion_token)
- [Huggingface](https://huggingface.co/datasets/code_x_glue_cc_code_completion_token)

## java

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:code_x_glue_cc_code_completion_token/java')
```

- **说明**：

```
CodeXGLUE CodeCompletion-token dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-token

Predict next code token given context of previous tokens. Models are evaluated by token level accuracy.
Code completion is a one of the most widely used features in software development through IDEs. An effective code completion tool could improve software developers' productivity. We provide code completion evaluation tasks in two granularities -- token level and line level. Here we introduce token level code completion. Token level task is analogous to language modeling. Models should have be able to predict the next token in arbitary types.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 8268
`'train'` | 12934
`'validation'` | 7189

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "code": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## python

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:code_x_glue_cc_code_completion_token/python')
```

- **说明**：

```
CodeXGLUE CodeCompletion-token dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-token

Predict next code token given context of previous tokens. Models are evaluated by token level accuracy.
Code completion is a one of the most widely used features in software development through IDEs. An effective code completion tool could improve software developers' productivity. We provide code completion evaluation tasks in two granularities -- token level and line level. Here we introduce token level code completion. Token level task is analogous to language modeling. Models should have be able to predict the next token in arbitary types.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 50000
`'train'` | 100000

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "code": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```
