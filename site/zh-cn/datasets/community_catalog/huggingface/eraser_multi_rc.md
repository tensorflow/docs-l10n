# eraser_multi_rc

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/eraser_multi_rc)
- [Huggingface](https://huggingface.co/datasets/eraser_multi_rc)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:eraser_multi_rc')
```

- **说明**：

```
Eraser Multi RC is a dataset for queries over multi-line passages, along with
answers and a rationalte. Each example in this dataset has the following 5 parts
1. A Mutli-line Passage
2. A Query about the passage
3. An Answer to the query
4. A Classification as to whether the answer is right or wrong
5. An Explanation justifying the classification
```

- **许可**：无已知许可
- **版本**：0.1.1
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 4848
`'train'` | 24029
`'validation'` | 3214

- **特征**：

```json
{
    "passage": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "query_and_answer": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 2,
        "names": [
            "False",
            "True"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "evidences": {
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
