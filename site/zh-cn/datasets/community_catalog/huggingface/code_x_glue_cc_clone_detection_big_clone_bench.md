# code_x_glue_cc_clone_detection_big_clone_bench

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/code_x_glue_cc_clone_detection_big_clone_bench)
- [Huggingface](https://huggingface.co/datasets/code_x_glue_cc_clone_detection_big_clone_bench)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:code_x_glue_cc_clone_detection_big_clone_bench')
```

- **说明**：

```
CodeXGLUE Clone-detection-BigCloneBench dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench

Given two codes as the input, the task is to do binary classification (0/1), where 1 stands for semantic equivalence and 0 for others. Models are evaluated by F1 score.
The dataset we use is BigCloneBench and filtered following the paper Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 415416
`'train'` | 901028
`'validation'` | 415416

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "id1": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "id2": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "func1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    }
}
```
