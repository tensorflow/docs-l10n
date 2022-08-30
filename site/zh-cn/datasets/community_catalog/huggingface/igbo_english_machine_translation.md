# igbo_english_machine_translation

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/igbo_english_machine_translation)
- [Huggingface](https://huggingface.co/datasets/igbo_english_machine_translation)

## ig-en

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:igbo_english_machine_translation/ig-en')
```

- **说明**：

```
Parallel Igbo-English Dataset
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 552
`'train'` | 10000
`'validation'` | 200

- **特征**：

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "translation": {
        "languages": [
            "ig",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    }
}
```
