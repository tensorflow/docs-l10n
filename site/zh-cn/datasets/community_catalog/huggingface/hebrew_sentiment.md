# hebrew_sentiment

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/hebrew_sentiment)
- [Huggingface](https://huggingface.co/datasets/hebrew_sentiment)

## token

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hebrew_sentiment/token')
```

- **说明**：

```
HebrewSentiment is a data set consists of 12,804 user comments to posts on the official Facebook page of Israel’s
president, Mr. Reuven Rivlin. In October 2015, we used the open software application Netvizz (Rieder,
2013) to scrape all the comments to all of the president’s posts in the period of June – August 2014,
the first three months of Rivlin’s presidency.2 While the president’s posts aimed at reconciling tensions
and called for tolerance and empathy, the sentiment expressed in the comments to the president’s posts
was polarized between citizens who warmly thanked the president, and citizens that fiercely critiqued his
policy. Of the 12,804 comments, 370 are neutral; 8,512 are positive, 3,922 negative.

Data Annotation: A trained researcher examined each comment and determined its sentiment value,
where comments with an overall positive sentiment were assigned the value 1, comments with an overall
negative sentiment were assigned the value -1, and comments that are off-topic to the post’s content
were assigned the value 0. We validated the coding scheme by asking a second trained researcher to
code the same data. There was substantial agreement between raters (N of agreements: 10623, N of
disagreements: 2105, Coehn’s Kappa = 0.697, p = 0).
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2560
`'train'` | 10244

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 3,
        "names": [
            "pos",
            "neg",
            "off-topic"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## morph

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hebrew_sentiment/morph')
```

- **说明**：

```
HebrewSentiment is a data set consists of 12,804 user comments to posts on the official Facebook page of Israel’s
president, Mr. Reuven Rivlin. In October 2015, we used the open software application Netvizz (Rieder,
2013) to scrape all the comments to all of the president’s posts in the period of June – August 2014,
the first three months of Rivlin’s presidency.2 While the president’s posts aimed at reconciling tensions
and called for tolerance and empathy, the sentiment expressed in the comments to the president’s posts
was polarized between citizens who warmly thanked the president, and citizens that fiercely critiqued his
policy. Of the 12,804 comments, 370 are neutral; 8,512 are positive, 3,922 negative.

Data Annotation: A trained researcher examined each comment and determined its sentiment value,
where comments with an overall positive sentiment were assigned the value 1, comments with an overall
negative sentiment were assigned the value -1, and comments that are off-topic to the post’s content
were assigned the value 0. We validated the coding scheme by asking a second trained researcher to
code the same data. There was substantial agreement between raters (N of agreements: 10623, N of
disagreements: 2105, Coehn’s Kappa = 0.697, p = 0).
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2555
`'train'` | 10221

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 3,
        "names": [
            "pos",
            "neg",
            "off-topic"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
