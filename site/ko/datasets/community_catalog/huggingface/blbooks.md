# blbooks

참조:

- [Code](https://github.com/huggingface/datasets/blob/master/datasets/blbooks)
- [Huggingface](https://huggingface.co/datasets/blbooks)

## all

다음 명령어를 사용하여 TFDS에서 이 데이터세트를 로드합니다.

```python
ds = tfds.load('huggingface:blbooks/all')
```

- **설명**:

```
A dataset comprising of text created by OCR from the 49,455 digitised books, equating to 65,227 volumes (25+ million pages), published between c. 1510 - c. 1900.
The books cover a wide range of subject areas including philosophy, history, poetry and literature.
```

- **라이선스**: 알려진 라이선스 없음
- **버전**: 1.0.2
- **Splits**:

Split | 예제
:-- | --:
`'train'` | 14011953

- **특성**:

```json
{
    "record_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "raw_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "place": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "empty_pg": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "pg": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "mean_wc_ocr": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "std_wc_ocr": {
        "dtype": "float64",
        "id": null,
        "_type": "Value"
    },
    "name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all_names": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Publisher": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Country of publication 1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all Countries of publication": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Physical description": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_3": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_4": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "multi_language": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    }
}
```

## 1800s

다음 명령어를 사용하여 TFDS에서 이 데이터세트를 로드합니다.

```python
ds = tfds.load('huggingface:blbooks/1800s')
```

- **설명**:

```
A dataset comprising of text created by OCR from the 49,455 digitised books, equating to 65,227 volumes (25+ million pages), published between c. 1510 - c. 1900.
The books cover a wide range of subject areas including philosophy, history, poetry and literature.
```

- **라이선스**: 알려진 라이선스 없음
- **버전**: 1.0.2
- **Splits**:

Split | 예제
:-- | --:
`'train'` | 13781747

- **특성**:

```json
{
    "record_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "raw_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "place": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "empty_pg": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "pg": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "mean_wc_ocr": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "std_wc_ocr": {
        "dtype": "float64",
        "id": null,
        "_type": "Value"
    },
    "name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all_names": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Publisher": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Country of publication 1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all Countries of publication": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Physical description": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_3": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_4": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "multi_language": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    }
}
```

## 1700s

다음 명령어를 사용하여 TFDS에서 이 데이터세트를 로드합니다.

```python
ds = tfds.load('huggingface:blbooks/1700s')
```

- **설명**:

```
A dataset comprising of text created by OCR from the 49,455 digitised books, equating to 65,227 volumes (25+ million pages), published between c. 1510 - c. 1900.
The books cover a wide range of subject areas including philosophy, history, poetry and literature.
```

- **라이선스**: 알려진 라이선스 없음
- **버전**: 1.0.2
- **Splits**:

Split | 예제
:-- | --:
`'train'` | 178224

- **특성**:

```json
{
    "record_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "raw_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "place": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "empty_pg": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "pg": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "mean_wc_ocr": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "std_wc_ocr": {
        "dtype": "float64",
        "id": null,
        "_type": "Value"
    },
    "name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all_names": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Publisher": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Country of publication 1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all Countries of publication": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Physical description": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_3": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_4": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "multi_language": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    }
}
```

## 1510_1699

다음 명령어를 사용하여 TFDS에서 이 데이터세트를 로드합니다.

```python
ds = tfds.load('huggingface:blbooks/1510_1699')
```

- **설명**:

```
A dataset comprising of text created by OCR from the 49,455 digitised books, equating to 65,227 volumes (25+ million pages), published between c. 1510 - c. 1900.
The books cover a wide range of subject areas including philosophy, history, poetry and literature.
```

- **라이선스**: 알려진 라이선스 없음
- **버전**: 1.0.2
- **Splits**:

Split | 예제
:-- | --:
`'train'` | 51982

- **특성**:

```json
{
    "record_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "timestamp[s]",
        "id": null,
        "_type": "Value"
    },
    "raw_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "place": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "empty_pg": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "pg": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "mean_wc_ocr": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "std_wc_ocr": {
        "dtype": "float64",
        "id": null,
        "_type": "Value"
    },
    "name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all_names": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Publisher": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Country of publication 1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all Countries of publication": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Physical description": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_3": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_4": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "multi_language": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    }
}
```

## 1500_1899

다음 명령어를 사용하여 TFDS에서 이 데이터세트를 로드합니다.

```python
ds = tfds.load('huggingface:blbooks/1500_1899')
```

- **설명**:

```
A dataset comprising of text created by OCR from the 49,455 digitised books, equating to 65,227 volumes (25+ million pages), published between c. 1510 - c. 1900.
The books cover a wide range of subject areas including philosophy, history, poetry and literature.
```

- **라이선스**: 알려진 라이선스 없음
- **버전**: 1.0.2
- **Splits**:

Split | 예제
:-- | --:
`'train'` | 14011953

- **특성**:

```json
{
    "record_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "timestamp[s]",
        "id": null,
        "_type": "Value"
    },
    "raw_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "place": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "empty_pg": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "pg": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "mean_wc_ocr": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "std_wc_ocr": {
        "dtype": "float64",
        "id": null,
        "_type": "Value"
    },
    "name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all_names": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Publisher": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Country of publication 1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all Countries of publication": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Physical description": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_3": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_4": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "multi_language": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    }
}
```

## 1800_1899

다음 명령어를 사용하여 TFDS에서 이 데이터세트를 로드합니다.

```python
ds = tfds.load('huggingface:blbooks/1800_1899')
```

- **설명**:

```
A dataset comprising of text created by OCR from the 49,455 digitised books, equating to 65,227 volumes (25+ million pages), published between c. 1510 - c. 1900.
The books cover a wide range of subject areas including philosophy, history, poetry and literature.
```

- **라이선스**: 알려진 라이선스 없음
- **버전**: 1.0.2
- **Splits**:

Split | 예제
:-- | --:
`'train'` | 13781747

- **특성**:

```json
{
    "record_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "timestamp[s]",
        "id": null,
        "_type": "Value"
    },
    "raw_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "place": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "empty_pg": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "pg": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "mean_wc_ocr": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "std_wc_ocr": {
        "dtype": "float64",
        "id": null,
        "_type": "Value"
    },
    "name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all_names": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Publisher": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Country of publication 1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all Countries of publication": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Physical description": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_3": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_4": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "multi_language": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    }
}
```

## 1700_1799

다음 명령어를 사용하여 TFDS에서 이 데이터세트를 로드합니다.

```python
ds = tfds.load('huggingface:blbooks/1700_1799')
```

- **설명**:

```
A dataset comprising of text created by OCR from the 49,455 digitised books, equating to 65,227 volumes (25+ million pages), published between c. 1510 - c. 1900.
The books cover a wide range of subject areas including philosophy, history, poetry and literature.
```

- **라이선스**: 알려진 라이선스 없음
- **버전**: 1.0.2
- **Splits**:

Split | 예제
:-- | --:
`'train'` | 178224

- **특성**:

```json
{
    "record_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "timestamp[s]",
        "id": null,
        "_type": "Value"
    },
    "raw_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "place": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "empty_pg": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "pg": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "mean_wc_ocr": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "std_wc_ocr": {
        "dtype": "float64",
        "id": null,
        "_type": "Value"
    },
    "name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all_names": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Publisher": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Country of publication 1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all Countries of publication": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Physical description": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_3": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_4": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "multi_language": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    }
}
```
