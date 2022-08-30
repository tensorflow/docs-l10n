# capes

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/capes)
- [Huggingface](https://huggingface.co/datasets/capes)

## en-pt

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:capes/en-pt')
```

- **说明**：

```
A parallel corpus of theses and dissertations abstracts in English and Portuguese were collected from the CAPES website (Coordenação de Aperfeiçoamento de Pessoal de Nível Superior) - Brazil. The corpus is sentence aligned for all language pairs. Approximately 240,000 documents were collected and aligned using the Hunalign algorithm.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1157610

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "pt"
        ],
        "id": null,
        "_type": "Translation"
    }
}
```
