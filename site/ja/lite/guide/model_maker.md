# TensorFlow Lite Model Maker

## 概要

TensorFlow Lite Model Maker ライブラリは、カスタムデータセットを使用した TensorFlow Lite のモデルのトレーニングプロセスを簡素化します。転移学習を使用するので必要なトレーニングデータ量が軽減され、トレーニングに費やす時間が短縮されます。

## サポートするタスク

現時点で Model Maker がサポートする機械学習タスクは以下のとおりです。モデルのトレーニング方法については、以下のリンクをクリックしてガイドをご覧ください。

サポートするタスク | タスクの使用目的
--- | ---
画像分類：[チュートリアル](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification)、[api](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/image_classifier) | 画像をあらかじめ定義したカテゴリに分類します。
物体検出：[チュートリアル](https://www.tensorflow.org/lite/tutorials/model_maker_object_detection)、[api](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/object_detector) | リアルタイムで物体を検出します。
テキスト分類：[チュートリアル](https://www.tensorflow.org/lite/tutorials/model_maker_text_classification)、[api](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/text_classifier) | テキストをあらかじめ定義したカテゴリに分類します。
BERT 質問応答：[チュートリアル](https://www.tensorflow.org/lite/tutorials/model_maker_question_answer)、[api](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/question_answer) | BERT を使って与えられた質問に対する回答を特定の文脈の中で探します。
音声分類：[チュートリアル](https://www.tensorflow.org/lite/tutorials/model_maker_audio_classification)、[api](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/audio_classifier) | 音声を事前定義されたカテゴリに分類します。
推薦：[デモ](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/demo/recommendation_demo.py)、[api](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/recommendation) | デバイス上のシナリオのコンテキスト情報に基づいてアイテムを推奨します。

タスクがサポートされていない場合は、最初に [TensorFlow](https://www.tensorflow.org/guide) を使用して、転移学習を使用して TensorFlow モデルを再トレーニングしてください（[画像](https://www.tensorflow.org/tutorials/images/transfer_learning)、[テキスト](https://www.tensorflow.org/official_models/fine_tuning_bert)、[音声](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio)などのガイドに従ってください) 。または、最初からトレーニングしてから、TensorFlowLite モデルに[変換](https://www.tensorflow.org/lite/convert)します。

## エンドツーエンドの例

Model Maker は、カスタムデータセットを使用して TensorFlow Lite のモデルをわずか数行のコードでトレーニングすることができます。例えば、画像分類モデルのトレーニング手順は以下の通りです。

```python
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

# Load input data specific to an on-device ML app.
data = DataLoader.from_folder('flower_photos/')
train_data, test_data = data.split(0.9)

# Customize the TensorFlow model.
model = image_classifier.create(train_data)

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

ナイトリー版をインストールする場合は、以下のコマンドに従ってください。

```shell
pip install tflite-model-maker-nightly
```

- GitHub からソースコードをクローンし、インストールする。

```shell
git clone https://github.com/tensorflow/examples
cd examples/tensorflow_examples/lite/model_maker/pip_package
pip install -e .
```

TensorFlow Lite Model Maker は、TensorFlow [pip パッケージ](https://www.tensorflow.org/install/pip)に依存しています。GPU ドライバについては、TensorFlow の [GPU ガイド](https://www.tensorflow.org/install/gpu)または[インストールガイド](https://www.tensorflow.org/install)を参照してください。

## Python API リファレンス

Model Maker のパブリック API については [API リファレンス](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker)を参照してください。
