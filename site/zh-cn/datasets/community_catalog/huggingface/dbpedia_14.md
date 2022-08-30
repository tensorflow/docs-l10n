# dbpedia_14

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/dbpedia_14)
- [Huggingface](https://huggingface.co/datasets/dbpedia_14)

## dbpedia_14

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:dbpedia_14/dbpedia_14')
```

- **Description**:

```
The DBpedia ontology classification dataset is constructed by picking 14 non-overlapping classes
from DBpedia 2014. They are listed in classes.txt. From each of thse 14 ontology classes, we
randomly choose 40,000 training samples and 5,000 testing samples. Therefore, the total size
of the training dataset is 560,000 and testing dataset 70,000.
There are 3 columns in the dataset (same for train and test splits), corresponding to class index
(1 to 14), title and content. The title and content are escaped using double quotes ("), and any
internal double quote is escaped by 2 double quotes (""). There are no new lines in title or content.
```

- **许可**：Creative Commons Attribution-ShareAlike 3.0 和 The GNU Free Documentation License
- **版本**：2.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 70000
`'train'` | 560000

- **特征**：

```json
{
    "label": {
        "num_classes": 14,
        "names": [
            "Company",
            "EducationalInstitution",
            "Artist",
            "Athlete",
            "OfficeHolder",
            "MeanOfTransportation",
            "Building",
            "NaturalPlace",
            "Village",
            "Animal",
            "Plant",
            "Album",
            "Film",
            "WrittenWork"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "content": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
