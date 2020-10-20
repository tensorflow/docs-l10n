# TensorFlow Lite Model Maker

## 概要

TensorFlow Lite Model Maker ライブラリは、カスタムデータセットを使用した TensorFlow Lite のモデルのトレーニングプロセスを簡素化します。転移学習を使用するので必要なトレーニングデータ量が軽減され、トレーニングに費やす時間が短縮されます。

## サポートするタスク

現時点で Model Maker がサポートする機械学習タスクは以下のとおりです。モデルのトレーニング方法については、以下のリンクをクリックしてガイドをご覧ください。

サポートするタスク | タスクのユーティリティ
--- | ---
画像分類 [ガイド](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification) | 画像をあらかじめ定義したカテゴリに分類します。
テキスト分類 [ガイド](https://www.tensorflow.org/lite/tutorials/model_maker_text_classification) | テキストをあらかじめ定義したカテゴリに分類します。
質問の回答 [ガイド](https://www.tensorflow.org/lite/tutorials/model_maker_question_answer) | 与えられた質問に対する回答を特定の文脈の中で探します。

## エンドツーエンドの例

Model Maker は、カスタムデータセットを使用して TensorFlow Lite のモデルをわずか数行のコードでトレーニングすることができます。例えば、画像分類モデルのトレーニング手順は以下の通りです。

```python
# Load input data specific to an on-device ML app.
data = ImageClassifierDataLoader.from_folder('flower_photos/')
train_data, test_data = data.split(0.9)

# Customize the TensorFlow model.
model = image_classifier.create(data)

# Evaluate the model.
loss, accuracy = model.evaluate(test_data)

# Export to Tensorflow Lite model and label file in `export_dir`.
model.export(export_dir='/tmp/')
```

詳細については、[画像分類ガイド](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification)をご覧ください。

## インストール

Model Maker のインストールには 2 通りの方法があります。

- 構築済みの pip パッケージをインストールする。

```shell
pip install tflite-model-maker
```

ナイトリー版をインストールする場合は、コマンドに従ってください。

```shell
pip install tflite-model-maker-nightly
```

- GitHub からソースコードをクローンし、インストールする。

```shell
git clone https://github.com/tensorflow/examples
cd examples/tensorflow_examples/lite/model_maker/pip_package
pip install -e .
```
