# emea

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/emea)
- [Huggingface](https://huggingface.co/datasets/emea)

## bg-el

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:emea/bg-el')
```

- **说明**：

```
This is a parallel corpus made out of PDF documents from the European Medicines Agency. All files are automatically converted from PDF to plain text using pdftotext with the command line arguments -layout -nopgbrk -eol unix. There are some known problems with tables and multi-column layouts - some of them are fixed in the current version.

source: http://www.emea.europa.eu/

22 languages, 231 bitexts
total number of files: 41,957
total number of tokens: 311.65M
total number of sentence fragments: 26.51M
```

- **许可**：无已知许可
- **版本**：3.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1044065

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
            "bg",
            "el"
        ],
        "id": null,
        "_type": "Translation"
    }
}
```

## cs-et

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:emea/cs-et')
```

- **说明**：

```
This is a parallel corpus made out of PDF documents from the European Medicines Agency. All files are automatically converted from PDF to plain text using pdftotext with the command line arguments -layout -nopgbrk -eol unix. There are some known problems with tables and multi-column layouts - some of them are fixed in the current version.

source: http://www.emea.europa.eu/

22 languages, 231 bitexts
total number of files: 41,957
total number of tokens: 311.65M
total number of sentence fragments: 26.51M
```

- **许可**：无已知许可
- **版本**：3.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1053164

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
            "cs",
            "et"
        ],
        "id": null,
        "_type": "Translation"
    }
}
```

## de-mt

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:emea/de-mt')
```

- **说明**：

```
This is a parallel corpus made out of PDF documents from the European Medicines Agency. All files are automatically converted from PDF to plain text using pdftotext with the command line arguments -layout -nopgbrk -eol unix. There are some known problems with tables and multi-column layouts - some of them are fixed in the current version.

source: http://www.emea.europa.eu/

22 languages, 231 bitexts
total number of files: 41,957
total number of tokens: 311.65M
total number of sentence fragments: 26.51M
```

- **许可**：无已知许可
- **版本**：3.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1000532

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
            "de",
            "mt"
        ],
        "id": null,
        "_type": "Translation"
    }
}
```

## fr-sk

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:emea/fr-sk')
```

- **说明**：

```
This is a parallel corpus made out of PDF documents from the European Medicines Agency. All files are automatically converted from PDF to plain text using pdftotext with the command line arguments -layout -nopgbrk -eol unix. There are some known problems with tables and multi-column layouts - some of them are fixed in the current version.

source: http://www.emea.europa.eu/

22 languages, 231 bitexts
total number of files: 41,957
total number of tokens: 311.65M
total number of sentence fragments: 26.51M
```

- **许可**：无已知许可
- **版本**：3.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1062753

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
            "fr",
            "sk"
        ],
        "id": null,
        "_type": "Translation"
    }
}
```

## es-lt

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:emea/es-lt')
```

- **说明**：

```
This is a parallel corpus made out of PDF documents from the European Medicines Agency. All files are automatically converted from PDF to plain text using pdftotext with the command line arguments -layout -nopgbrk -eol unix. There are some known problems with tables and multi-column layouts - some of them are fixed in the current version.

source: http://www.emea.europa.eu/

22 languages, 231 bitexts
total number of files: 41,957
total number of tokens: 311.65M
total number of sentence fragments: 26.51M
```

- **许可**：无已知许可
- **版本**：3.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1051370

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
            "es",
            "lt"
        ],
        "id": null,
        "_type": "Translation"
    }
}
```
