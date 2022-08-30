# eitb_parcc

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/eitb_parcc)
- [Huggingface](https://huggingface.co/datasets/eitb_parcc)

## es-eu

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:eitb_parcc/es-eu')
```

- **说明**：

```
EiTB-ParCC: Parallel Corpus of Comparable News. A Basque-Spanish parallel corpus provided by Vicomtech (https://www.vicomtech.org), extracted from comparable news produced by the Basque public broadcasting group Euskal Irrati Telebista.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 637183

- **特征**：

```json
{
    "translation": {
        "languages": [
            "es",
            "eu"
        ],
        "id": null,
        "_type": "Translation"
    }
}
```
