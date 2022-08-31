# adversarial_qa

참조:

- [Code](https://github.com/huggingface/datasets/blob/master/datasets/adversarial_qa)
- [Huggingface](https://huggingface.co/datasets/adversarial_qa)

## adversarialQA

다음 명령어를 사용하여 TFDS에서 이 데이터세트를 로드합니다.

```python
ds = tfds.load('huggingface:adversarial_qa/adversarialQA')
```

- **설명**:

```
AdversarialQA is a Reading Comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles using an adversarial model-in-the-loop.
We use three different models; BiDAF (Seo et al., 2016), BERT-Large (Devlin et al., 2018), and RoBERTa-Large (Liu et al., 2019) in the annotation loop and construct three datasets; D(BiDAF), D(BERT), and D(RoBERTa), each with 10,000 training examples, 1,000 validation, and 1,000 test examples.
The adversarial human annotation paradigm ensures that these datasets consist of questions that current state-of-the-art models (at least the ones used as adversaries in the annotation loop) find challenging.
```

- **라이선스**: 알려진 라이선스 없음
- **버전**: 1.0.0
- **Splits**:

Split | 예제
:-- | --:
`'test'` | 3000
`'train'` | 30000
`'validation'` | 3000

- **특성**:

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "context": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "answers": {
        "feature": {
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "answer_start": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "metadata": {
        "split": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "model_in_the_loop": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## dbidaf

다음 명령어를 사용하여 TFDS에서 이 데이터세트를 로드합니다.

```python
ds = tfds.load('huggingface:adversarial_qa/dbidaf')
```

- **설명**:

```
AdversarialQA is a Reading Comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles using an adversarial model-in-the-loop.
We use three different models; BiDAF (Seo et al., 2016), BERT-Large (Devlin et al., 2018), and RoBERTa-Large (Liu et al., 2019) in the annotation loop and construct three datasets; D(BiDAF), D(BERT), and D(RoBERTa), each with 10,000 training examples, 1,000 validation, and 1,000 test examples.
The adversarial human annotation paradigm ensures that these datasets consist of questions that current state-of-the-art models (at least the ones used as adversaries in the annotation loop) find challenging.
```

- **라이선스**: 알려진 라이선스 없음
- **버전**: 1.0.0
- **Splits**:

Split | 예제
:-- | --:
`'test'` | 1000
`'train'` | 10000
`'validation'` | 1000

- **특성**:

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "context": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "answers": {
        "feature": {
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "answer_start": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "metadata": {
        "split": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "model_in_the_loop": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## dbert

다음 명령어를 사용하여 TFDS에서 이 데이터세트를 로드합니다.

```python
ds = tfds.load('huggingface:adversarial_qa/dbert')
```

- **설명**:

```
AdversarialQA is a Reading Comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles using an adversarial model-in-the-loop.
We use three different models; BiDAF (Seo et al., 2016), BERT-Large (Devlin et al., 2018), and RoBERTa-Large (Liu et al., 2019) in the annotation loop and construct three datasets; D(BiDAF), D(BERT), and D(RoBERTa), each with 10,000 training examples, 1,000 validation, and 1,000 test examples.
The adversarial human annotation paradigm ensures that these datasets consist of questions that current state-of-the-art models (at least the ones used as adversaries in the annotation loop) find challenging.
```

- **라이선스**: 알려진 라이선스 없음
- **버전**: 1.0.0
- **Splits**:

Split | 예제
:-- | --:
`'test'` | 1000
`'train'` | 10000
`'validation'` | 1000

- **특성**:

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "context": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "answers": {
        "feature": {
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "answer_start": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "metadata": {
        "split": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "model_in_the_loop": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```

## droberta

다음 명령어를 사용하여 TFDS에서 이 데이터세트를 로드합니다.

```python
ds = tfds.load('huggingface:adversarial_qa/droberta')
```

- **설명**:

```
AdversarialQA is a Reading Comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles using an adversarial model-in-the-loop.
We use three different models; BiDAF (Seo et al., 2016), BERT-Large (Devlin et al., 2018), and RoBERTa-Large (Liu et al., 2019) in the annotation loop and construct three datasets; D(BiDAF), D(BERT), and D(RoBERTa), each with 10,000 training examples, 1,000 validation, and 1,000 test examples.
The adversarial human annotation paradigm ensures that these datasets consist of questions that current state-of-the-art models (at least the ones used as adversaries in the annotation loop) find challenging.
```

- **라이선스**: 알려진 라이선스 없음
- **버전**: 1.0.0
- **Splits**:

Split | 예제
:-- | --:
`'test'` | 1000
`'train'` | 10000
`'validation'` | 1000

- **특성**:

```json
{
    "id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "context": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "answers": {
        "feature": {
            "text": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "answer_start": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "metadata": {
        "split": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "model_in_the_loop": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    }
}
```
