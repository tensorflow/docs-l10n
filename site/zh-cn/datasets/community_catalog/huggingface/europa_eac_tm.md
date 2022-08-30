# europa_eac_tm

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/europa_eac_tm)
- [Huggingface](https://huggingface.co/datasets/europa_eac_tm)

## en2bg

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2bg')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 4061

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "bg"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2cs

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2cs')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 3351

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "cs"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2da

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2da')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 3757

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "da"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2de

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2de')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 4473

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "de"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2el

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2el')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2818

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "el"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2es

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2es')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 4303

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "es"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2et

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2et')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2270

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "et"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2fi

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2fi')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1458

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "fi"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2fr

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2fr')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 4476

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "fr"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2hu

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2hu')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 3455

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "hu"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2is

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2is')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2206

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "is"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2it

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2it')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2170

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "it"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2lt

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2lt')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 3386

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "lt"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2lv

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2lv')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 3880

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "lv"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2mt

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2mt')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1722

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "mt"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2nb

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2nb')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 642

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "nb"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2nl

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2nl')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1805

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "nl"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2pl

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2pl')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 4027

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "pl"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2pt

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2pt')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 3501

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
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2ro

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2ro')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 3159

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "ro"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2sk

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2sk')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2972

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "sk"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2sl

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2sl')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 4644

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "sl"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2sv

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2sv')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2909

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "sv"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## en2tr

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:europa_eac_tm/en2tr')
```

- **说明**：

```
In October 2012, the European Union's (EU) Directorate General for Education and Culture ( DG EAC) released a translation memory (TM), i.e. a collection of sentences and their professionally produced translations, in twenty-six languages. This resource bears the name EAC Translation Memory, short EAC-TM.

EAC-TM covers up to 26 languages: 22 official languages of the EU (all except Irish) plus Icelandic, Croatian, Norwegian and Turkish. EAC-TM thus contains translations from English into the following 25 languages: Bulgarian, Czech, Danish, Dutch, Estonian, German, Greek, Finnish, French, Croatian, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish and Turkish.

All documents and sentences were originally written in English (source language is English) and then translated into the other languages. The texts were translated by staff of the National Agencies of the Lifelong Learning and Youth in Action programmes. They are typically professionals in the field of education/youth and EU programmes. They are thus not professional translators, but they are normally native speakers of the target language.
```

- **许可**：Creative Commons Attribution 4.0 International(CC BY 4.0) licence © European Union, 1995-2020
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 3198

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "tr"
        ],
        "id": null,
        "_type": "Translation"
    },
    "sentence_type": {
        "num_classes": 2,
        "names": [
            "form_data",
            "sentence_data"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
