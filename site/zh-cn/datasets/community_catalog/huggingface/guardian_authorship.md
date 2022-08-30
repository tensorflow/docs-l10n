# guardian_authorship

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/guardian_authorship)
- [Huggingface](https://huggingface.co/datasets/guardian_authorship)

## cross_topic_1

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:guardian_authorship/cross_topic_1')
```

- **Description**:

```
A dataset cross-topic authorship attribution. The dataset is provided by Stamatatos 2013.
1- The cross-topic scenarios are based on Table-4 in Stamatatos 2017 (Ex. cross_topic_1 => row 1:P S U&W ).
2- The cross-genre scenarios are based on Table-5 in the same paper. (Ex. cross_genre_1 => row 1:B P S&U&W).

3- The same-topic/genre scenario is created by grouping all the datasts as follows.
For ex., to use same_topic and split the data 60-40 use:
train_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[:60%]+validation[:60%]+test[:60%]')
tests_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[-40%:]+validation[-40%:]+test[-40%:]')

IMPORTANT: train+validation+test[:60%] will generate the wrong splits becasue the data is imbalanced

* See https://huggingface.co/docs/datasets/splits.html for detailed/more examples
```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 207
`'train'` | 112
`'validation'` | 62

- **特征**：

```json
{
    "author": {
        "num_classes": 13,
        "names": [
            "catherinebennett",
            "georgemonbiot",
            "hugoyoung",
            "jonathanfreedland",
            "martinkettle",
            "maryriddell",
            "nickcohen",
            "peterpreston",
            "pollytoynbee",
            "royhattersley",
            "simonhoggart",
            "willhutton",
            "zoewilliams"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "topic": {
        "num_classes": 5,
        "names": [
            "Politics",
            "Society",
            "UK",
            "World",
            "Books"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## cross_genre_1

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:guardian_authorship/cross_genre_1')
```

- **Description**:

```
A dataset cross-topic authorship attribution. The dataset is provided by Stamatatos 2013.
1- The cross-topic scenarios are based on Table-4 in Stamatatos 2017 (Ex. cross_topic_1 => row 1:P S U&W ).
2- The cross-genre scenarios are based on Table-5 in the same paper. (Ex. cross_genre_1 => row 1:B P S&U&W).

3- The same-topic/genre scenario is created by grouping all the datasts as follows.
For ex., to use same_topic and split the data 60-40 use:
train_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[:60%]+validation[:60%]+test[:60%]')
tests_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[-40%:]+validation[-40%:]+test[-40%:]')

IMPORTANT: train+validation+test[:60%] will generate the wrong splits becasue the data is imbalanced

* See https://huggingface.co/docs/datasets/splits.html for detailed/more examples
```

- **许可**：无已知许可
- **版本**：13.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 269
`'train'` | 63
`'validation'` | 112

- **特征**：

```json
{
    "author": {
        "num_classes": 13,
        "names": [
            "catherinebennett",
            "georgemonbiot",
            "hugoyoung",
            "jonathanfreedland",
            "martinkettle",
            "maryriddell",
            "nickcohen",
            "peterpreston",
            "pollytoynbee",
            "royhattersley",
            "simonhoggart",
            "willhutton",
            "zoewilliams"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "topic": {
        "num_classes": 5,
        "names": [
            "Politics",
            "Society",
            "UK",
            "World",
            "Books"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## cross_topic_2

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:guardian_authorship/cross_topic_2')
```

- **Description**:

```
A dataset cross-topic authorship attribution. The dataset is provided by Stamatatos 2013.
1- The cross-topic scenarios are based on Table-4 in Stamatatos 2017 (Ex. cross_topic_1 => row 1:P S U&W ).
2- The cross-genre scenarios are based on Table-5 in the same paper. (Ex. cross_genre_1 => row 1:B P S&U&W).

3- The same-topic/genre scenario is created by grouping all the datasts as follows.
For ex., to use same_topic and split the data 60-40 use:
train_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[:60%]+validation[:60%]+test[:60%]')
tests_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[-40%:]+validation[-40%:]+test[-40%:]')

IMPORTANT: train+validation+test[:60%] will generate the wrong splits becasue the data is imbalanced

* See https://huggingface.co/docs/datasets/splits.html for detailed/more examples
```

- **许可**：无已知许可
- **版本**：2.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 179
`'train'` | 112
`'validation'` | 90

- **特征**：

```json
{
    "author": {
        "num_classes": 13,
        "names": [
            "catherinebennett",
            "georgemonbiot",
            "hugoyoung",
            "jonathanfreedland",
            "martinkettle",
            "maryriddell",
            "nickcohen",
            "peterpreston",
            "pollytoynbee",
            "royhattersley",
            "simonhoggart",
            "willhutton",
            "zoewilliams"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "topic": {
        "num_classes": 5,
        "names": [
            "Politics",
            "Society",
            "UK",
            "World",
            "Books"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## cross_topic_3

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:guardian_authorship/cross_topic_3')
```

- **Description**:

```
A dataset cross-topic authorship attribution. The dataset is provided by Stamatatos 2013.
1- The cross-topic scenarios are based on Table-4 in Stamatatos 2017 (Ex. cross_topic_1 => row 1:P S U&W ).
2- The cross-genre scenarios are based on Table-5 in the same paper. (Ex. cross_genre_1 => row 1:B P S&U&W).

3- The same-topic/genre scenario is created by grouping all the datasts as follows.
For ex., to use same_topic and split the data 60-40 use:
train_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[:60%]+validation[:60%]+test[:60%]')
tests_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[-40%:]+validation[-40%:]+test[-40%:]')

IMPORTANT: train+validation+test[:60%] will generate the wrong splits becasue the data is imbalanced

* See https://huggingface.co/docs/datasets/splits.html for detailed/more examples
```

- **许可**：无已知许可
- **版本**：3.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 152
`'train'` | 112
`'validation'` | 117

- **特征**：

```json
{
    "author": {
        "num_classes": 13,
        "names": [
            "catherinebennett",
            "georgemonbiot",
            "hugoyoung",
            "jonathanfreedland",
            "martinkettle",
            "maryriddell",
            "nickcohen",
            "peterpreston",
            "pollytoynbee",
            "royhattersley",
            "simonhoggart",
            "willhutton",
            "zoewilliams"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "topic": {
        "num_classes": 5,
        "names": [
            "Politics",
            "Society",
            "UK",
            "World",
            "Books"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## cross_topic_4

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:guardian_authorship/cross_topic_4')
```

- **Description**:

```
A dataset cross-topic authorship attribution. The dataset is provided by Stamatatos 2013.
1- The cross-topic scenarios are based on Table-4 in Stamatatos 2017 (Ex. cross_topic_1 => row 1:P S U&W ).
2- The cross-genre scenarios are based on Table-5 in the same paper. (Ex. cross_genre_1 => row 1:B P S&U&W).

3- The same-topic/genre scenario is created by grouping all the datasts as follows.
For ex., to use same_topic and split the data 60-40 use:
train_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[:60%]+validation[:60%]+test[:60%]')
tests_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[-40%:]+validation[-40%:]+test[-40%:]')

IMPORTANT: train+validation+test[:60%] will generate the wrong splits becasue the data is imbalanced

* See https://huggingface.co/docs/datasets/splits.html for detailed/more examples
```

- **许可**：无已知许可
- **版本**：4.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 207
`'train'` | 62
`'validation'` | 112

- **特征**：

```json
{
    "author": {
        "num_classes": 13,
        "names": [
            "catherinebennett",
            "georgemonbiot",
            "hugoyoung",
            "jonathanfreedland",
            "martinkettle",
            "maryriddell",
            "nickcohen",
            "peterpreston",
            "pollytoynbee",
            "royhattersley",
            "simonhoggart",
            "willhutton",
            "zoewilliams"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "topic": {
        "num_classes": 5,
        "names": [
            "Politics",
            "Society",
            "UK",
            "World",
            "Books"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## cross_topic_5

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:guardian_authorship/cross_topic_5')
```

- **Description**:

```
A dataset cross-topic authorship attribution. The dataset is provided by Stamatatos 2013.
1- The cross-topic scenarios are based on Table-4 in Stamatatos 2017 (Ex. cross_topic_1 => row 1:P S U&W ).
2- The cross-genre scenarios are based on Table-5 in the same paper. (Ex. cross_genre_1 => row 1:B P S&U&W).

3- The same-topic/genre scenario is created by grouping all the datasts as follows.
For ex., to use same_topic and split the data 60-40 use:
train_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[:60%]+validation[:60%]+test[:60%]')
tests_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[-40%:]+validation[-40%:]+test[-40%:]')

IMPORTANT: train+validation+test[:60%] will generate the wrong splits becasue the data is imbalanced

* See https://huggingface.co/docs/datasets/splits.html for detailed/more examples
```

- **许可**：无已知许可
- **版本**：5.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 229
`'train'` | 62
`'validation'` | 90

- **特征**：

```json
{
    "author": {
        "num_classes": 13,
        "names": [
            "catherinebennett",
            "georgemonbiot",
            "hugoyoung",
            "jonathanfreedland",
            "martinkettle",
            "maryriddell",
            "nickcohen",
            "peterpreston",
            "pollytoynbee",
            "royhattersley",
            "simonhoggart",
            "willhutton",
            "zoewilliams"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "topic": {
        "num_classes": 5,
        "names": [
            "Politics",
            "Society",
            "UK",
            "World",
            "Books"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## cross_topic_6

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:guardian_authorship/cross_topic_6')
```

- **Description**:

```
A dataset cross-topic authorship attribution. The dataset is provided by Stamatatos 2013.
1- The cross-topic scenarios are based on Table-4 in Stamatatos 2017 (Ex. cross_topic_1 => row 1:P S U&W ).
2- The cross-genre scenarios are based on Table-5 in the same paper. (Ex. cross_genre_1 => row 1:B P S&U&W).

3- The same-topic/genre scenario is created by grouping all the datasts as follows.
For ex., to use same_topic and split the data 60-40 use:
train_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[:60%]+validation[:60%]+test[:60%]')
tests_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[-40%:]+validation[-40%:]+test[-40%:]')

IMPORTANT: train+validation+test[:60%] will generate the wrong splits becasue the data is imbalanced

* See https://huggingface.co/docs/datasets/splits.html for detailed/more examples
```

- **许可**：无已知许可
- **版本**：6.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 202
`'train'` | 62
`'validation'` | 117

- **特征**：

```json
{
    "author": {
        "num_classes": 13,
        "names": [
            "catherinebennett",
            "georgemonbiot",
            "hugoyoung",
            "jonathanfreedland",
            "martinkettle",
            "maryriddell",
            "nickcohen",
            "peterpreston",
            "pollytoynbee",
            "royhattersley",
            "simonhoggart",
            "willhutton",
            "zoewilliams"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "topic": {
        "num_classes": 5,
        "names": [
            "Politics",
            "Society",
            "UK",
            "World",
            "Books"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## cross_topic_7

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:guardian_authorship/cross_topic_7')
```

- **Description**:

```
A dataset cross-topic authorship attribution. The dataset is provided by Stamatatos 2013.
1- The cross-topic scenarios are based on Table-4 in Stamatatos 2017 (Ex. cross_topic_1 => row 1:P S U&W ).
2- The cross-genre scenarios are based on Table-5 in the same paper. (Ex. cross_genre_1 => row 1:B P S&U&W).

3- The same-topic/genre scenario is created by grouping all the datasts as follows.
For ex., to use same_topic and split the data 60-40 use:
train_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[:60%]+validation[:60%]+test[:60%]')
tests_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[-40%:]+validation[-40%:]+test[-40%:]')

IMPORTANT: train+validation+test[:60%] will generate the wrong splits becasue the data is imbalanced

* See https://huggingface.co/docs/datasets/splits.html for detailed/more examples
```

- **许可**：无已知许可
- **版本**：7.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 179
`'train'` | 90
`'validation'` | 112

- **特征**：

```json
{
    "author": {
        "num_classes": 13,
        "names": [
            "catherinebennett",
            "georgemonbiot",
            "hugoyoung",
            "jonathanfreedland",
            "martinkettle",
            "maryriddell",
            "nickcohen",
            "peterpreston",
            "pollytoynbee",
            "royhattersley",
            "simonhoggart",
            "willhutton",
            "zoewilliams"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "topic": {
        "num_classes": 5,
        "names": [
            "Politics",
            "Society",
            "UK",
            "World",
            "Books"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## cross_topic_8

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:guardian_authorship/cross_topic_8')
```

- **Description**:

```
A dataset cross-topic authorship attribution. The dataset is provided by Stamatatos 2013.
1- The cross-topic scenarios are based on Table-4 in Stamatatos 2017 (Ex. cross_topic_1 => row 1:P S U&W ).
2- The cross-genre scenarios are based on Table-5 in the same paper. (Ex. cross_genre_1 => row 1:B P S&U&W).

3- The same-topic/genre scenario is created by grouping all the datasts as follows.
For ex., to use same_topic and split the data 60-40 use:
train_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[:60%]+validation[:60%]+test[:60%]')
tests_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[-40%:]+validation[-40%:]+test[-40%:]')

IMPORTANT: train+validation+test[:60%] will generate the wrong splits becasue the data is imbalanced

* See https://huggingface.co/docs/datasets/splits.html for detailed/more examples
```

- **许可**：无已知许可
- **版本**：8.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 229
`'train'` | 90
`'validation'` | 62

- **特征**：

```json
{
    "author": {
        "num_classes": 13,
        "names": [
            "catherinebennett",
            "georgemonbiot",
            "hugoyoung",
            "jonathanfreedland",
            "martinkettle",
            "maryriddell",
            "nickcohen",
            "peterpreston",
            "pollytoynbee",
            "royhattersley",
            "simonhoggart",
            "willhutton",
            "zoewilliams"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "topic": {
        "num_classes": 5,
        "names": [
            "Politics",
            "Society",
            "UK",
            "World",
            "Books"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## cross_topic_9

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:guardian_authorship/cross_topic_9')
```

- **Description**:

```
A dataset cross-topic authorship attribution. The dataset is provided by Stamatatos 2013.
1- The cross-topic scenarios are based on Table-4 in Stamatatos 2017 (Ex. cross_topic_1 => row 1:P S U&W ).
2- The cross-genre scenarios are based on Table-5 in the same paper. (Ex. cross_genre_1 => row 1:B P S&U&W).

3- The same-topic/genre scenario is created by grouping all the datasts as follows.
For ex., to use same_topic and split the data 60-40 use:
train_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[:60%]+validation[:60%]+test[:60%]')
tests_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[-40%:]+validation[-40%:]+test[-40%:]')

IMPORTANT: train+validation+test[:60%] will generate the wrong splits becasue the data is imbalanced

* See https://huggingface.co/docs/datasets/splits.html for detailed/more examples
```

- **许可**：无已知许可
- **版本**：9.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 174
`'train'` | 90
`'validation'` | 117

- **特征**：

```json
{
    "author": {
        "num_classes": 13,
        "names": [
            "catherinebennett",
            "georgemonbiot",
            "hugoyoung",
            "jonathanfreedland",
            "martinkettle",
            "maryriddell",
            "nickcohen",
            "peterpreston",
            "pollytoynbee",
            "royhattersley",
            "simonhoggart",
            "willhutton",
            "zoewilliams"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "topic": {
        "num_classes": 5,
        "names": [
            "Politics",
            "Society",
            "UK",
            "World",
            "Books"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## cross_topic_10

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:guardian_authorship/cross_topic_10')
```

- **Description**:

```
A dataset cross-topic authorship attribution. The dataset is provided by Stamatatos 2013.
1- The cross-topic scenarios are based on Table-4 in Stamatatos 2017 (Ex. cross_topic_1 => row 1:P S U&W ).
2- The cross-genre scenarios are based on Table-5 in the same paper. (Ex. cross_genre_1 => row 1:B P S&U&W).

3- The same-topic/genre scenario is created by grouping all the datasts as follows.
For ex., to use same_topic and split the data 60-40 use:
train_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[:60%]+validation[:60%]+test[:60%]')
tests_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[-40%:]+validation[-40%:]+test[-40%:]')

IMPORTANT: train+validation+test[:60%] will generate the wrong splits becasue the data is imbalanced

* See https://huggingface.co/docs/datasets/splits.html for detailed/more examples
```

- **许可**：无已知许可
- **版本**：10.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 152
`'train'` | 117
`'validation'` | 112

- **特征**：

```json
{
    "author": {
        "num_classes": 13,
        "names": [
            "catherinebennett",
            "georgemonbiot",
            "hugoyoung",
            "jonathanfreedland",
            "martinkettle",
            "maryriddell",
            "nickcohen",
            "peterpreston",
            "pollytoynbee",
            "royhattersley",
            "simonhoggart",
            "willhutton",
            "zoewilliams"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "topic": {
        "num_classes": 5,
        "names": [
            "Politics",
            "Society",
            "UK",
            "World",
            "Books"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## cross_topic_11

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:guardian_authorship/cross_topic_11')
```

- **Description**:

```
A dataset cross-topic authorship attribution. The dataset is provided by Stamatatos 2013.
1- The cross-topic scenarios are based on Table-4 in Stamatatos 2017 (Ex. cross_topic_1 => row 1:P S U&W ).
2- The cross-genre scenarios are based on Table-5 in the same paper. (Ex. cross_genre_1 => row 1:B P S&U&W).

3- The same-topic/genre scenario is created by grouping all the datasts as follows.
For ex., to use same_topic and split the data 60-40 use:
train_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[:60%]+validation[:60%]+test[:60%]')
tests_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[-40%:]+validation[-40%:]+test[-40%:]')

IMPORTANT: train+validation+test[:60%] will generate the wrong splits becasue the data is imbalanced

* See https://huggingface.co/docs/datasets/splits.html for detailed/more examples
```

- **许可**：无已知许可
- **版本**：11.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 202
`'train'` | 117
`'validation'` | 62

- **特征**：

```json
{
    "author": {
        "num_classes": 13,
        "names": [
            "catherinebennett",
            "georgemonbiot",
            "hugoyoung",
            "jonathanfreedland",
            "martinkettle",
            "maryriddell",
            "nickcohen",
            "peterpreston",
            "pollytoynbee",
            "royhattersley",
            "simonhoggart",
            "willhutton",
            "zoewilliams"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "topic": {
        "num_classes": 5,
        "names": [
            "Politics",
            "Society",
            "UK",
            "World",
            "Books"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## cross_topic_12

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:guardian_authorship/cross_topic_12')
```

- **Description**:

```
A dataset cross-topic authorship attribution. The dataset is provided by Stamatatos 2013.
1- The cross-topic scenarios are based on Table-4 in Stamatatos 2017 (Ex. cross_topic_1 => row 1:P S U&W ).
2- The cross-genre scenarios are based on Table-5 in the same paper. (Ex. cross_genre_1 => row 1:B P S&U&W).

3- The same-topic/genre scenario is created by grouping all the datasts as follows.
For ex., to use same_topic and split the data 60-40 use:
train_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[:60%]+validation[:60%]+test[:60%]')
tests_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[-40%:]+validation[-40%:]+test[-40%:]')

IMPORTANT: train+validation+test[:60%] will generate the wrong splits becasue the data is imbalanced

* See https://huggingface.co/docs/datasets/splits.html for detailed/more examples
```

- **许可**：无已知许可
- **版本**：12.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 174
`'train'` | 117
`'validation'` | 90

- **特征**：

```json
{
    "author": {
        "num_classes": 13,
        "names": [
            "catherinebennett",
            "georgemonbiot",
            "hugoyoung",
            "jonathanfreedland",
            "martinkettle",
            "maryriddell",
            "nickcohen",
            "peterpreston",
            "pollytoynbee",
            "royhattersley",
            "simonhoggart",
            "willhutton",
            "zoewilliams"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "topic": {
        "num_classes": 5,
        "names": [
            "Politics",
            "Society",
            "UK",
            "World",
            "Books"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## cross_genre_2

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:guardian_authorship/cross_genre_2')
```

- **Description**:

```
A dataset cross-topic authorship attribution. The dataset is provided by Stamatatos 2013.
1- The cross-topic scenarios are based on Table-4 in Stamatatos 2017 (Ex. cross_topic_1 => row 1:P S U&W ).
2- The cross-genre scenarios are based on Table-5 in the same paper. (Ex. cross_genre_1 => row 1:B P S&U&W).

3- The same-topic/genre scenario is created by grouping all the datasts as follows.
For ex., to use same_topic and split the data 60-40 use:
train_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[:60%]+validation[:60%]+test[:60%]')
tests_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[-40%:]+validation[-40%:]+test[-40%:]')

IMPORTANT: train+validation+test[:60%] will generate the wrong splits becasue the data is imbalanced

* See https://huggingface.co/docs/datasets/splits.html for detailed/more examples
```

- **许可**：无已知许可
- **版本**：14.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 319
`'train'` | 63
`'validation'` | 62

- **特征**：

```json
{
    "author": {
        "num_classes": 13,
        "names": [
            "catherinebennett",
            "georgemonbiot",
            "hugoyoung",
            "jonathanfreedland",
            "martinkettle",
            "maryriddell",
            "nickcohen",
            "peterpreston",
            "pollytoynbee",
            "royhattersley",
            "simonhoggart",
            "willhutton",
            "zoewilliams"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "topic": {
        "num_classes": 5,
        "names": [
            "Politics",
            "Society",
            "UK",
            "World",
            "Books"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## cross_genre_3

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:guardian_authorship/cross_genre_3')
```

- **Description**:

```
A dataset cross-topic authorship attribution. The dataset is provided by Stamatatos 2013.
1- The cross-topic scenarios are based on Table-4 in Stamatatos 2017 (Ex. cross_topic_1 => row 1:P S U&W ).
2- The cross-genre scenarios are based on Table-5 in the same paper. (Ex. cross_genre_1 => row 1:B P S&U&W).

3- The same-topic/genre scenario is created by grouping all the datasts as follows.
For ex., to use same_topic and split the data 60-40 use:
train_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[:60%]+validation[:60%]+test[:60%]')
tests_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[-40%:]+validation[-40%:]+test[-40%:]')

IMPORTANT: train+validation+test[:60%] will generate the wrong splits becasue the data is imbalanced

* See https://huggingface.co/docs/datasets/splits.html for detailed/more examples
```

- **许可**：无已知许可
- **版本**：15.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 291
`'train'` | 63
`'validation'` | 90

- **特征**：

```json
{
    "author": {
        "num_classes": 13,
        "names": [
            "catherinebennett",
            "georgemonbiot",
            "hugoyoung",
            "jonathanfreedland",
            "martinkettle",
            "maryriddell",
            "nickcohen",
            "peterpreston",
            "pollytoynbee",
            "royhattersley",
            "simonhoggart",
            "willhutton",
            "zoewilliams"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "topic": {
        "num_classes": 5,
        "names": [
            "Politics",
            "Society",
            "UK",
            "World",
            "Books"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## cross_genre_4

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:guardian_authorship/cross_genre_4')
```

- **Description**:

```
A dataset cross-topic authorship attribution. The dataset is provided by Stamatatos 2013.
1- The cross-topic scenarios are based on Table-4 in Stamatatos 2017 (Ex. cross_topic_1 => row 1:P S U&W ).
2- The cross-genre scenarios are based on Table-5 in the same paper. (Ex. cross_genre_1 => row 1:B P S&U&W).

3- The same-topic/genre scenario is created by grouping all the datasts as follows.
For ex., to use same_topic and split the data 60-40 use:
train_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[:60%]+validation[:60%]+test[:60%]')
tests_ds = load_dataset('guardian_authorship', name="cross_topic_<<#>>",
                        split='train[-40%:]+validation[-40%:]+test[-40%:]')

IMPORTANT: train+validation+test[:60%] will generate the wrong splits becasue the data is imbalanced

* See https://huggingface.co/docs/datasets/splits.html for detailed/more examples
```

- **许可**：无已知许可
- **版本**：16.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 264
`'train'` | 63
`'validation'` | 117

- **特征**：

```json
{
    "author": {
        "num_classes": 13,
        "names": [
            "catherinebennett",
            "georgemonbiot",
            "hugoyoung",
            "jonathanfreedland",
            "martinkettle",
            "maryriddell",
            "nickcohen",
            "peterpreston",
            "pollytoynbee",
            "royhattersley",
            "simonhoggart",
            "willhutton",
            "zoewilliams"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "topic": {
        "num_classes": 5,
        "names": [
            "Politics",
            "Society",
            "UK",
            "World",
            "Books"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "article": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
