# hate_speech18

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/hate_speech18)
- [Huggingface](https://huggingface.co/datasets/hate_speech18)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hate_speech18')
```

- **说明**：

```
These files contain text extracted from Stormfront, a white supremacist forum. A random set of
forums posts have been sampled from several subforums and split into sentences. Those sentences
have been manually labelled as containing hate speech or not, according to certain annotation guidelines.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 10944

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "user_id": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "subforum_id": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "num_contexts": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 4,
        "names": [
            "noHate",
            "hate",
            "idk/skip",
            "relation"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
