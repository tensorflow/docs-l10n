# common_voice

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/common_voice)
- [Huggingface](https://huggingface.co/datasets/common_voice)

## ab

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/ab')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 8
`'other'` | 752
`'test'` | 9
`'train'` | 22
`'validated'` | 31
`'validation'` | 0

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## ar

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/ar')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 6333
`'other'` | 18283
`'test'` | 7622
`'train'` | 14227
`'validated'` | 43291
`'validation'` | 7517

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## as

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/as')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 31
`'other'` | 0
`'test'` | 110
`'train'` | 270
`'validated'` | 504
`'validation'` | 124

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## br

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/br')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 623
`'other'` | 10912
`'test'` | 2087
`'train'` | 2780
`'validated'` | 8560
`'validation'` | 1997

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## ca

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/ca')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 18846
`'other'` | 64446
`'test'` | 15724
`'train'` | 285584
`'validated'` | 416701
`'validation'` | 15724

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## cnh

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/cnh')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 433
`'other'` | 2934
`'test'` | 752
`'train'` | 807
`'validated'` | 2432
`'validation'` | 756

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## cs

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/cs')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 685
`'other'` | 7475
`'test'` | 4144
`'train'` | 5655
`'validated'` | 30431
`'validation'` | 4118

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## cv

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/cv')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 1282
`'other'` | 6927
`'test'` | 788
`'train'` | 931
`'validated'` | 3496
`'validation'` | 818

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## cy

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/cy')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 3648
`'other'` | 17919
`'test'` | 4820
`'train'` | 6839
`'validated'` | 72984
`'validation'` | 4776

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## de

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/de')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 32789
`'other'` | 10095
`'test'` | 15588
`'train'` | 246525
`'validated'` | 565186
`'validation'` | 15588

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## dv

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/dv')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 840
`'other'` | 0
`'test'` | 2202
`'train'` | 2680
`'validated'` | 11866
`'validation'` | 2077

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## el

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/el')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 185
`'other'` | 5659
`'test'` | 1522
`'train'` | 2316
`'validated'` | 5996
`'validation'` | 1401

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## en

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/en')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 189562
`'other'` | 169895
`'test'` | 16164
`'train'` | 564337
`'validated'` | 1224864
`'validation'` | 16164

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## eo

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/eo')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 4736
`'other'` | 2946
`'test'` | 8969
`'train'` | 19587
`'validated'` | 58094
`'validation'` | 8987

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## es

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/es')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 40640
`'other'` | 144791
`'test'` | 15089
`'train'` | 161813
`'validated'` | 236314
`'validation'` | 15089

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## et

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/et')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 3557
`'other'` | 569
`'test'` | 2509
`'train'` | 2966
`'validated'` | 10683
`'validation'` | 2507

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## eu

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/eu')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 5387
`'other'` | 23570
`'test'` | 5172
`'train'` | 7505
`'validated'` | 63009
`'validation'` | 5172

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## fa

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/fa')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 11698
`'other'` | 22510
`'test'` | 5213
`'train'` | 7593
`'validated'` | 251659
`'validation'` | 5213

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## fi

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/fi')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 59
`'other'` | 149
`'test'` | 428
`'train'` | 460
`'validated'` | 1305
`'validation'` | 415

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## fr

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/fr')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 40351
`'other'` | 3222
`'test'` | 15763
`'train'` | 298982
`'validated'` | 461004
`'validation'` | 15763

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## fy-NL

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/fy-NL')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 1031
`'other'` | 21569
`'test'` | 3020
`'train'` | 3927
`'validated'` | 10495
`'validation'` | 2790

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## ga-IE

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/ga-IE')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 409
`'other'` | 2130
`'test'` | 506
`'train'` | 541
`'validated'` | 3352
`'validation'` | 497

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## hi

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/hi')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 60
`'other'` | 139
`'test'` | 127
`'train'` | 157
`'validated'` | 419
`'validation'` | 135

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## hsb

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/hsb')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 227
`'other'` | 62
`'test'` | 387
`'train'` | 808
`'validated'` | 1367
`'validation'` | 172

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## hu

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/hu')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 169
`'other'` | 295
`'test'` | 1649
`'train'` | 3348
`'validated'` | 6457
`'validation'` | 1434

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## ia

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/ia')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 192
`'other'` | 1095
`'test'` | 899
`'train'` | 3477
`'validated'` | 5978
`'validation'` | 1601

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## id

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/id')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 470
`'other'` | 6782
`'test'` | 1844
`'train'` | 2130
`'validated'` | 8696
`'validation'` | 1835

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## it

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/it')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 12189
`'other'` | 14549
`'test'` | 12928
`'train'` | 58015
`'validated'` | 102579
`'validation'` | 12928

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## ja

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/ja')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 504
`'other'` | 885
`'test'` | 632
`'train'` | 722
`'validated'` | 3072
`'validation'` | 586

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## ka

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/ka')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 139
`'other'` | 44
`'test'` | 656
`'train'` | 1058
`'validated'` | 2275
`'validation'` | 527

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## kab

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/kab')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 18134
`'other'` | 88021
`'test'` | 14622
`'train'` | 120530
`'validated'` | 573718
`'validation'` | 14622

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## ky

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/ky')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 926
`'other'` | 7223
`'test'` | 1503
`'train'` | 1955
`'validated'` | 9236
`'validation'` | 1511

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## lg

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/lg')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 290
`'other'` | 3110
`'test'` | 584
`'train'` | 1250
`'validated'` | 2220
`'validation'` | 384

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## lt

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/lt')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 102
`'other'` | 1629
`'test'` | 466
`'train'` | 931
`'validated'` | 1644
`'validation'` | 244

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## lv

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/lv')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 143
`'other'` | 1560
`'test'` | 1882
`'train'` | 2552
`'validated'` | 6444
`'validation'` | 2002

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## mn

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/mn')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 667
`'other'` | 3272
`'test'` | 1862
`'train'` | 2183
`'validated'` | 7487
`'validation'` | 1837

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## mt

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/mt')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 314
`'other'` | 5714
`'test'` | 1617
`'train'` | 2036
`'validated'` | 5747
`'validation'` | 1516

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## nl

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/nl')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 3308
`'other'` | 27
`'test'` | 5708
`'train'` | 9460
`'validated'` | 52488
`'validation'` | 4938

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## or

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/or')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 62
`'other'` | 4302
`'test'` | 98
`'train'` | 388
`'validated'` | 615
`'validation'` | 129

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## pa-IN

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/pa-IN')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 43
`'other'` | 1411
`'test'` | 116
`'train'` | 211
`'validated'` | 371
`'validation'` | 44

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## pl

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/pl')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 4601
`'other'` | 12848
`'test'` | 5153
`'train'` | 7468
`'validated'` | 90791
`'validation'` | 5153

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## pt

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/pt')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 1740
`'other'` | 8390
`'test'` | 4641
`'train'` | 6514
`'validated'` | 41584
`'validation'` | 4592

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## rm-sursilv

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/rm-sursilv')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 639
`'other'` | 2102
`'test'` | 1194
`'train'` | 1384
`'validated'` | 3783
`'validation'` | 1205

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## rm-vallader

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/rm-vallader')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 374
`'other'` | 727
`'test'` | 378
`'train'` | 574
`'validated'` | 1316
`'validation'` | 357

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## ro

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/ro')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 485
`'other'` | 1945
`'test'` | 1778
`'train'` | 3399
`'validated'` | 6039
`'validation'` | 858

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## ru

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/ru')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 3056
`'other'` | 10247
`'test'` | 8007
`'train'` | 15481
`'validated'` | 74256
`'validation'` | 7963

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## rw

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/rw')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 206790
`'other'` | 22923
`'test'` | 15724
`'train'` | 515197
`'validated'` | 832929
`'validation'` | 15032

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## sah

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/sah')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 66
`'other'` | 1275
`'test'` | 757
`'train'` | 1442
`'validated'` | 2606
`'validation'` | 405

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## sl

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/sl')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 92
`'other'` | 2502
`'test'` | 881
`'train'` | 2038
`'validated'` | 4669
`'validation'` | 556

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## sv-SE

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/sv-SE')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 462
`'other'` | 3043
`'test'` | 2027
`'train'` | 2331
`'validated'` | 12552
`'validation'` | 2019

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## ta

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/ta')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 594
`'other'` | 7428
`'test'` | 1781
`'train'` | 2009
`'validated'` | 12652
`'validation'` | 1779

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## th

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/th')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 467
`'other'` | 2671
`'test'` | 2188
`'train'` | 2917
`'validated'` | 7028
`'validation'` | 1922

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## tr

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/tr')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 1726
`'other'` | 325
`'test'` | 1647
`'train'` | 1831
`'validated'` | 18685
`'validation'` | 1647

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## tt

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/tt')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 287
`'other'` | 1798
`'test'` | 4485
`'train'` | 11211
`'validated'` | 25781
`'validation'` | 2127

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## uk

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/uk')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 1255
`'other'` | 8161
`'test'` | 3235
`'train'` | 4035
`'validated'` | 22337
`'validation'` | 3236

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## vi

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/vi')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 78
`'other'` | 870
`'test'` | 198
`'train'` | 221
`'validated'` | 619
`'validation'` | 200

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## vot

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/vot')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 6
`'other'` | 411
`'test'` | 0
`'train'` | 3
`'validated'` | 3
`'validation'` | 0

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## zh-CN

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/zh-CN')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 5305
`'other'` | 8948
`'test'` | 8760
`'train'` | 18541
`'validated'` | 36405
`'validation'` | 8743

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## zh-HK

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/zh-HK')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 2999
`'other'` | 38830
`'test'` | 5172
`'train'` | 7506
`'validated'` | 41835
`'validation'` | 5172

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## zh-TW

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_voice/zh-TW')
```

- **说明**：

```
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
```

- **许可**：https://github.com/common-voice/common-voice/blob/main/LICENSE
- **版本**：6.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'invalidated'` | 3584
`'other'` | 22477
`'test'` | 2895
`'train'` | 3507
`'validated'` | 61232
`'validation'` | 2895

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "audio": {
        "sampling_rate": 48000,
        "mono": true,
        "decode": true,
        "id": null,
        "_type": "Audio"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "up_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "down_votes": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "accent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "locale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "segment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
