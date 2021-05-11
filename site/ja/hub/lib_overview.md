<!--* freshness: { owner: 'kempy' } *-->

# TensorFlow Hub ライブラリの概要

[`tensorflow_hub`](https://github.com/tensorflow/hub) ライブラリでは、最小限のコードでトレーニング済みのモデルをダウンロードし、TensorFlow プログラムで再利用することができます。トレーニング済みモデルの読み込みには、主に `hub.KerasLayer` API が使用されます。

```python
import tensorflow_hub as hub

embed = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2")
embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
print(embeddings.shape, embeddings.dtype)
```

## ダウンロードのキャッシュロケーションを設定する

By default, `tensorflow_hub` uses a system-wide, temporary directory to cache downloaded and uncompressed models. See [Caching](caching.md) for options to use other, possibly more persistent locations.

## API stability

Although we hope to prevent breaking changes, this project is still under active development and is not yet guaranteed to have a stable API or model format.

## Fairness

As in all of machine learning, [fairness](http://ml-fairness.com) is an [important](https://research.googleblog.com/2016/10/equality-of-opportunity-in-machine.html) consideration. Many pre-trained models are trained on large datasets. When reusing any model, it’s important to be mindful of what data the model was trained on (and whether there are any existing biases there), and how these might impact your use of it.

## Security

Since they contain arbitrary TensorFlow graphs, models can be thought of as programs. [Using TensorFlow Securely](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md) describes the security implications of referencing a model from an untrusted source.

## Next Steps

- [ライブラリを使用する](https://gitlocalize.com/repo/4592/ja/site/en-snapshot/hub/tf2_saved_model.md)
- [Reusable SavedModel](https://gitlocalize.com/repo/4592/ja/site/en-snapshot/hub/reusable_saved_models.md)
