**更新日: 2020 年 8 月 7 日**

## 量子化

- ダイナミックレンジカーネル用のポストトレーニング量子化 -- [公開済み](https://blog.tensorflow.org/2018/09/introducing-model-optimization-toolkit.html)
- （8b）固定小数点カーネル用のポストトレーニング量子化 -- [公開済み](https://blog.tensorflow.org/2019/06/tensorflow-integer-quantization.html)
- （8b）固定小数点カーネル用量子化対応トレーニングおよび <8b 用実験 -- [公開済み](https://blog.tensorflow.org/2020/04/quantization-aware-training-with-tensorflow-model-optimization-toolkit.html)
- ［進行中］（8b）固定小数点 RNN 用ポストトレーニング量子化
- （8b）固定小数点 RNN 用量子化対応トレーニング
- ［進行中］ポストトレーニングのダイナミックレンジ量子化の品質とパフォーマンスの改善

## プルーニング / スパース化

- トレーニング中のマグニチュードベースの重みプルーニング -- [公開済み](https://blog.tensorflow.org/2019/05/tf-model-optimization-toolkit-pruning-API.html)
- TensorFlow Lite におけるスパースモデルの実行サポート -- [進行中](https://github.com/tensorflow/model-optimization/issues/173)

## 重みクラスタリング

- トレーニング中の重みクラスタリング -- [公開済み](https://blog.tensorflow.org/2020/08/tensorflow-model-optimization-toolkit-weight-clustering-api.html)

## カスケード圧縮テクニック

- ［進行中］さまざまな圧縮テクニックを組み合わせるための追加サポート。現在、1 つのトレーニング中のテクニックとポストトレーニング量子化のみを組み合わせることができます。提案内容は近日公開されます。

## 圧縮

- ［進行中］テンソル圧縮 API
