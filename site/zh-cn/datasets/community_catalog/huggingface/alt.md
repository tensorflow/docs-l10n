# alt

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/alt)
- [Huggingface](https://huggingface.co/datasets/alt)

## alt-parallel

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:alt/alt-parallel')
```

- **说明**：

```
The ALT project aims to advance the state-of-the-art Asian natural language processing (NLP) techniques through the open collaboration for developing and using ALT. It was first conducted by NICT and UCSY as described in Ye Kyaw Thu, Win Pa Pa, Masao Utiyama, Andrew Finch and Eiichiro Sumita (2016). Then, it was developed under ASEAN IVO as described in this Web page. The process of building ALT began with sampling about 20,000 sentences from English Wikinews, and then these sentences were translated into the other languages. ALT now has 13 languages: Bengali, English, Filipino, Hindi, Bahasa Indonesia, Japanese, Khmer, Lao, Malay, Myanmar (Burmese), Thai, Vietnamese, Chinese (Simplified Chinese).
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1019
`'train'` | 18094
`'validation'` | 1004

- **特征**：

```json
{
    "SNT.URLID": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "SNT.URLID.SNTID": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "translation": {
        "languages": [
            "bg",
            "en",
            "en_tok",
            "fil",
            "hi",
            "id",
            "ja",
            "khm",
            "lo",
            "ms",
            "my",
            "th",
            "vi",
            "zh"
        ],
        "id": null,
        "_type": "Translation"
    }
}
```

## alt-en

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:alt/alt-en')
```

- **说明**：

```
The ALT project aims to advance the state-of-the-art Asian natural language processing (NLP) techniques through the open collaboration for developing and using ALT. It was first conducted by NICT and UCSY as described in Ye Kyaw Thu, Win Pa Pa, Masao Utiyama, Andrew Finch and Eiichiro Sumita (2016). Then, it was developed under ASEAN IVO as described in this Web page. The process of building ALT began with sampling about 20,000 sentences from English Wikinews, and then these sentences were translated into the other languages. ALT now has 13 languages: Bengali, English, Filipino, Hindi, Bahasa Indonesia, Japanese, Khmer, Lao, Malay, Myanmar (Burmese), Thai, Vietnamese, Chinese (Simplified Chinese).
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1017
`'train'` | 17889
`'validation'` | 988

- **特征**：

```json
{
    "SNT.URLID": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "SNT.URLID.SNTID": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "status": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "value": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## alt-jp

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:alt/alt-jp')
```

- **说明**：

```
The ALT project aims to advance the state-of-the-art Asian natural language processing (NLP) techniques through the open collaboration for developing and using ALT. It was first conducted by NICT and UCSY as described in Ye Kyaw Thu, Win Pa Pa, Masao Utiyama, Andrew Finch and Eiichiro Sumita (2016). Then, it was developed under ASEAN IVO as described in this Web page. The process of building ALT began with sampling about 20,000 sentences from English Wikinews, and then these sentences were translated into the other languages. ALT now has 13 languages: Bengali, English, Filipino, Hindi, Bahasa Indonesia, Japanese, Khmer, Lao, Malay, Myanmar (Burmese), Thai, Vietnamese, Chinese (Simplified Chinese).
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 931
`'train'` | 17202
`'validation'` | 953

- **特征**：

```json
{
    "SNT.URLID": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "SNT.URLID.SNTID": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "status": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "value": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "word_alignment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "jp_tokenized": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "en_tokenized": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## alt-my

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:alt/alt-my')
```

- **说明**：

```
The ALT project aims to advance the state-of-the-art Asian natural language processing (NLP) techniques through the open collaboration for developing and using ALT. It was first conducted by NICT and UCSY as described in Ye Kyaw Thu, Win Pa Pa, Masao Utiyama, Andrew Finch and Eiichiro Sumita (2016). Then, it was developed under ASEAN IVO as described in this Web page. The process of building ALT began with sampling about 20,000 sentences from English Wikinews, and then these sentences were translated into the other languages. ALT now has 13 languages: Bengali, English, Filipino, Hindi, Bahasa Indonesia, Japanese, Khmer, Lao, Malay, Myanmar (Burmese), Thai, Vietnamese, Chinese (Simplified Chinese).
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1018
`'train'` | 18088
`'validation'` | 1000

- **特征**：

```json
{
    "SNT.URLID": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "SNT.URLID.SNTID": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "value": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## alt-km

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:alt/alt-km')
```

- **说明**：

```
The ALT project aims to advance the state-of-the-art Asian natural language processing (NLP) techniques through the open collaboration for developing and using ALT. It was first conducted by NICT and UCSY as described in Ye Kyaw Thu, Win Pa Pa, Masao Utiyama, Andrew Finch and Eiichiro Sumita (2016). Then, it was developed under ASEAN IVO as described in this Web page. The process of building ALT began with sampling about 20,000 sentences from English Wikinews, and then these sentences were translated into the other languages. ALT now has 13 languages: Bengali, English, Filipino, Hindi, Bahasa Indonesia, Japanese, Khmer, Lao, Malay, Myanmar (Burmese), Thai, Vietnamese, Chinese (Simplified Chinese).
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1018
`'train'` | 18088
`'validation'` | 1000

- **特征**：

```json
{
    "SNT.URLID": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "SNT.URLID.SNTID": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "km_pos_tag": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "km_tokenized": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## alt-my-transliteration

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:alt/alt-my-transliteration')
```

- **说明**：

```
The ALT project aims to advance the state-of-the-art Asian natural language processing (NLP) techniques through the open collaboration for developing and using ALT. It was first conducted by NICT and UCSY as described in Ye Kyaw Thu, Win Pa Pa, Masao Utiyama, Andrew Finch and Eiichiro Sumita (2016). Then, it was developed under ASEAN IVO as described in this Web page. The process of building ALT began with sampling about 20,000 sentences from English Wikinews, and then these sentences were translated into the other languages. ALT now has 13 languages: Bengali, English, Filipino, Hindi, Bahasa Indonesia, Japanese, Khmer, Lao, Malay, Myanmar (Burmese), Thai, Vietnamese, Chinese (Simplified Chinese).
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 84022

- **特征**：

```json
{
    "en": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "my": {
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

## alt-my-west-transliteration

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:alt/alt-my-west-transliteration')
```

- **说明**：

```
The ALT project aims to advance the state-of-the-art Asian natural language processing (NLP) techniques through the open collaboration for developing and using ALT. It was first conducted by NICT and UCSY as described in Ye Kyaw Thu, Win Pa Pa, Masao Utiyama, Andrew Finch and Eiichiro Sumita (2016). Then, it was developed under ASEAN IVO as described in this Web page. The process of building ALT began with sampling about 20,000 sentences from English Wikinews, and then these sentences were translated into the other languages. ALT now has 13 languages: Bengali, English, Filipino, Hindi, Bahasa Indonesia, Japanese, Khmer, Lao, Malay, Myanmar (Burmese), Thai, Vietnamese, Chinese (Simplified Chinese).
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 107121

- **特征**：

```json
{
    "en": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "my": {
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
