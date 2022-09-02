# bswac

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/bswac)
- [Huggingface](https://huggingface.co/datasets/bswac)

## bswac

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bswac/bswac')
```

- **说明**：

```
The Bosnian web corpus bsWaC was built by crawling the .ba top-level domain in 2014. The corpus was near-deduplicated on paragraph level, normalised via diacritic restoration, morphosyntactically annotated and lemmatised. The corpus is shuffled by paragraphs. Each paragraph contains metadata on the URL, domain and language identification (Bosnian vs. Croatian vs. Serbian).

Version 1.0 of this corpus is described in http://www.aclweb.org/anthology/W14-0405. Version 1.1 contains newer and better linguistic annotations.
```

- **许可**：CC BY-SA 4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 354581267

- **特征**：

```json
{
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
