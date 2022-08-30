# hrenwac_para

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/hrenwac_para)
- [Huggingface](https://huggingface.co/datasets/hrenwac_para)

## hrenWaC

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hrenwac_para/hrenWaC')
```

- **说明**：

```
The hrenWaC corpus version 2.0 consists of parallel Croatian-English texts crawled from the .hr top-level domain for Croatia. The corpus was built with Spidextor (https://github.com/abumatran/spidextor), a tool that glues together the output of SpiderLing used for crawling and Bitextor used for bitext extraction. The accuracy of the extracted bitext on the segment level is around 80% and on the word level around 84%.
```

- **许可**：CC BY-SA 3.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 99001

- **特征**：

```json
{
    "translation": {
        "languages": [
            "en",
            "hr"
        ],
        "id": null,
        "_type": "Translation"
    }
}
```
