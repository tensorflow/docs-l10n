# covid_tweets_japanese

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/covid_tweets_japanese)
- [Huggingface](https://huggingface.co/datasets/covid_tweets_japanese)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:covid_tweets_japanese')
```

- **说明**：

```
53,640 Japanese tweets with annotation if a tweet is related to COVID-19 or not. The annotation is by majority decision by 5 - 10 crowd workers. Target tweets include "COVID" or "コロナ". The period of the tweets is from around January 2020 to around June 2020. The original tweets are not contained. Please use Twitter API to get them, for example.
```

- **许可**：CC-BY-ND 4.0
- **版本**：1.1.1
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 53639

- **特征**：

```json
{
    "tweet_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "assessment_option_id": {
        "num_classes": 6,
        "names": [
            "63",
            "64",
            "65",
            "66",
            "67",
            "68"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
