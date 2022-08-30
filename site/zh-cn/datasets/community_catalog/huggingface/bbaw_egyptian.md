# bbaw_egyptian

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/bbaw_egyptian)
- [Huggingface](https://huggingface.co/datasets/bbaw_egyptian)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bbaw_egyptian')
```

- **说明**：

```
The project `Strukturen und Transformationen des Wortschatzes der ägyptischen Sprache`
is compiling an extensively annotated digital corpus of Egyptian texts.
This publication comprises an excerpt of the internal database's contents.
```

- **许可**：Creative Commons-Lizenz - CC BY-SA - 4.0 International
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 100736

- **特征**：

```json
{
    "transcription": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "translation": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "hieroglyphs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
