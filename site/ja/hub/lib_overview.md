<!--* freshness: { owner: 'akhorlin' } *-->

# TensorFlow Hub ライブラリの概要

[`tensorflow_hub`](https://github.com/tensorflow/hub) ライブラリでは、最小限のコードでトレーニング済みのモデルをダウンロードし、TensorFlow プログラムで再利用することができます。トレーニング済みモデルの読み込みには、主に `hub.KerasLayer` API が使用されます。

```python
import tensorflow_hub as hub

embed = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2")
embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
print(embeddings.shape, embeddings.dtype)
```

## ダウンロードのキャッシュロケーションを設定する

デフォルトでは、`tensorflow_hub` はシステム全体の一時ディレクトリを使用して、ダウンロードや圧縮されたモデルをキャッシュします。その他のより永続的なロケーションのオプションについては、[キャッシング](caching.md)をご覧ください。

## API の安定性

変更による破損を回避することに努めてはいますが、このプロジェクトは現在開発中であるため、安定した API やモデル形式を保証していません。

## 公平性

すべての機械学習と同様に、[公平性](http://ml-fairness.com)は[重要な](https://research.googleblog.com/2016/10/equality-of-opportunity-in-machine.html)考慮事項です。多くのトレーニング済みモデルは、大規模なデータセットでトレーニングされています。モデルを再利用する際は、どのデータを使ってモデルがトレーニングされたのか（およびバイアスが既存していたかどうか）、およびこのことがモデルの使用にどのような影響を与えるのかに配慮することが重要です。

## セキュリティ

任意の TensorFlow グラフが含まれるため、モデルはプログラムとして捉えられます。「[TensorFlow を安全に使用する](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md)」では、信頼されないソースのモデルを参照する際のセキュリティ問題が説明されています。

## 次のステップ

- ライブラリを使用する
- Reusable SavedModel
