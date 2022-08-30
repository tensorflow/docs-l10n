# bianet

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/bianet)
- [Huggingface](https://huggingface.co/datasets/bianet)

## en_to_ku

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:bianet/en_to_ku')
```

- **Description**:

```
A parallel news corpus in Turkish, Kurdish and English.
Bianet collects 3,214 Turkish articles with their sentence-aligned Kurdish or English translations from the Bianet online newspaper.
3 languages, 3 bitexts
total number of files: 6
total number of tokens: 2.25M
total number of sentence fragments: 0.14M
```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 6402

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
            "en",
            "ku"
        ],
        "id": null,
        "_type": "Translation"
    }
}
```

## en_to_tr

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:bianet/en_to_tr')
```

- **Description**:

```
A parallel news corpus in Turkish, Kurdish and English.
Bianet collects 3,214 Turkish articles with their sentence-aligned Kurdish or English translations from the Bianet online newspaper.
3 languages, 3 bitexts
total number of files: 6
total number of tokens: 2.25M
total number of sentence fragments: 0.14M
```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 34770

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
            "en",
            "tr"
        ],
        "id": null,
        "_type": "Translation"
    }
}
```

## ku_to_tr

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:bianet/ku_to_tr')
```

- **Description**:

```
A parallel news corpus in Turkish, Kurdish and English.
Bianet collects 3,214 Turkish articles with their sentence-aligned Kurdish or English translations from the Bianet online newspaper.
3 languages, 3 bitexts
total number of files: 6
total number of tokens: 2.25M
total number of sentence fragments: 0.14M
```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 7325

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
            "ku",
            "tr"
        ],
        "id": null,
        "_type": "Translation"
    }
}
```
