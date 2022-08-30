# ccaligned_multilingual

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/ccaligned_multilingual)
- [Huggingface](https://huggingface.co/datasets/ccaligned_multilingual)

## documents-zz_TR

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:ccaligned_multilingual/documents-zz_TR')
```

- **说明**：

```
CCAligned consists of parallel or comparable web-document pairs in 137 languages aligned with English. These web-document pairs were constructed by performing language identification on raw web-documents, and ensuring corresponding language codes were corresponding in the URLs of web documents. This pattern matching approach yielded more than 100 million aligned documents paired with English. Recognizing that each English document was often aligned to mulitple documents in different target language, we can join on English documents to obtain aligned documents that directly pair two non-English documents (e.g., Arabic-French).
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 41

- **特征**：

```json
{
    "Domain": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Source_URL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Target_URL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "translation": {
        "languages": [
            "en_XX",
            "zz_TR"
        ],
        "id": null,
        "_type": "Translation"
    }
}
```

## sentences-zz_TR

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:ccaligned_multilingual/sentences-zz_TR')
```

- **说明**：

```
CCAligned consists of parallel or comparable web-document pairs in 137 languages aligned with English. These web-document pairs were constructed by performing language identification on raw web-documents, and ensuring corresponding language codes were corresponding in the URLs of web documents. This pattern matching approach yielded more than 100 million aligned documents paired with English. Recognizing that each English document was often aligned to mulitple documents in different target language, we can join on English documents to obtain aligned documents that directly pair two non-English documents (e.g., Arabic-French).
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 34

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en_XX",
            "zz_TR"
        ],
        "id": null,
        "_type": "Translation"
    },
    "LASER_similarity": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    }
}
```

## documents-tz_MA

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:ccaligned_multilingual/documents-tz_MA')
```

- **说明**：

```
CCAligned consists of parallel or comparable web-document pairs in 137 languages aligned with English. These web-document pairs were constructed by performing language identification on raw web-documents, and ensuring corresponding language codes were corresponding in the URLs of web documents. This pattern matching approach yielded more than 100 million aligned documents paired with English. Recognizing that each English document was often aligned to mulitple documents in different target language, we can join on English documents to obtain aligned documents that directly pair two non-English documents (e.g., Arabic-French).
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 4

- **特征**：

```json
{
    "Domain": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Source_URL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Target_URL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "translation": {
        "languages": [
            "en_XX",
            "tz_MA"
        ],
        "id": null,
        "_type": "Translation"
    }
}
```

## sentences-tz_MA

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:ccaligned_multilingual/sentences-tz_MA')
```

- **说明**：

```
CCAligned consists of parallel or comparable web-document pairs in 137 languages aligned with English. These web-document pairs were constructed by performing language identification on raw web-documents, and ensuring corresponding language codes were corresponding in the URLs of web documents. This pattern matching approach yielded more than 100 million aligned documents paired with English. Recognizing that each English document was often aligned to mulitple documents in different target language, we can join on English documents to obtain aligned documents that directly pair two non-English documents (e.g., Arabic-French).
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 33

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en_XX",
            "tz_MA"
        ],
        "id": null,
        "_type": "Translation"
    },
    "LASER_similarity": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    }
}
```

## documents-ak_GH

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:ccaligned_multilingual/documents-ak_GH')
```

- **说明**：

```
CCAligned consists of parallel or comparable web-document pairs in 137 languages aligned with English. These web-document pairs were constructed by performing language identification on raw web-documents, and ensuring corresponding language codes were corresponding in the URLs of web documents. This pattern matching approach yielded more than 100 million aligned documents paired with English. Recognizing that each English document was often aligned to mulitple documents in different target language, we can join on English documents to obtain aligned documents that directly pair two non-English documents (e.g., Arabic-French).
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 249

- **特征**：

```json
{
    "Domain": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Source_URL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Target_URL": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "translation": {
        "languages": [
            "en_XX",
            "ak_GH"
        ],
        "id": null,
        "_type": "Translation"
    }
}
```

## sentences-ak_GH

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:ccaligned_multilingual/sentences-ak_GH')
```

- **说明**：

```
CCAligned consists of parallel or comparable web-document pairs in 137 languages aligned with English. These web-document pairs were constructed by performing language identification on raw web-documents, and ensuring corresponding language codes were corresponding in the URLs of web documents. This pattern matching approach yielded more than 100 million aligned documents paired with English. Recognizing that each English document was often aligned to mulitple documents in different target language, we can join on English documents to obtain aligned documents that directly pair two non-English documents (e.g., Arabic-French).
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 478

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en_XX",
            "ak_GH"
        ],
        "id": null,
        "_type": "Translation"
    },
    "LASER_similarity": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    }
}
```
