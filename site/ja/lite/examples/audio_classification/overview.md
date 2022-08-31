# 音声分類

<img src="../images/audio.png" class="attempt-right">

音声が表している内容を特定するタスクは、*音声分類*と呼ばれます。音声分類モデルは、さまざまな音声イベントを認識するようにトレーニングされています。たとえば、モデルをトレーニングして、拍手、指を鳴らす音、タイピングという 3 つの異なるイベントを表すイベントを認識できます。TensorFlow Lite は、モバイルアプリケーションでデプロイできる、最適化された事前トレーニング済みモデルを提供します。TensorFlow を使用した音声分類の詳細については、[こちら](https://www.tensorflow.org/tutorials/audio/simple_audio)を参照してください。

次の図は、Android での音声分類モデルの出力を示します。

<img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ja/lite/examples/audio_classification/images/android_audio_classification.png?raw=true" alt="Screenshot of Android example" class="">

注意: (1) 既存のモデルを統合するには、[TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier) を試してください。(2) モデルをカスタマイズするには、[TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/audio_classification) を試してください。

## はじめに

TensorFlow Lite を初めて使用する場合、Android を使用する場合は、以下のサンプルアプリをご覧ください。

[TensorFlow Lite Task Library](../../inference_with_metadata/task_library/audio_classifier) のそのまま簡単に使用できる API を利用して、わずか数行のコードで音声分類モデルを統合できます。また、[TensorFlow Lite Support Library](../../inference_with_metadata/lite_support) を使用して、独自のカスタム推論パイプラインを構築することもできます。

次の Android の例では、[TFLite Task Library](https://github.com/tensorflow/examples/tree/master/lite/examples/sound_classification/android) を使用した実装を示します。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/sound_classification/android">Android の例を見る</a>

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/sound_classification/ios">iOS の例を見る</a>

Android/iOS 以外のプラットフォームを使用する場合、または、すでに [TensorFlow Lite API](https://www.tensorflow.org/api_docs/python/tf/lite) に精通している場合は、スターターモデルと追加ファイル (該当する場合) をダウンロードしてください。

<a class="button button-primary" href="https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite">TensorFlow Hub からスターターモデルをダウンロード</a>

## モデルの説明

YAMNet は音声イベント分類器であり、音声波形を入力として受け取り、[AudioSet](https://g.co/audioset) オントロジーから 521 の音声イベントのそれぞれに対して独立した予測を行います。このモデルは、MobileNet v1 アーキテクチャを使用し、AudioSet コーパスを使用してトレーニングされました。このモデルは、当初、モデルソースコード、元のモデルチェックポイント、詳細ドキュメントが提供されている TensorFlow Model Garden でリリースされました。

### 使い方

TFLite に変換された YAMNet モデルには次の 2 つのバージョンがあります。

- [YAMNet](https://tfhub.dev/google/yamnet/1): 元の音声分類モデル。動的入力サイズで、転移学習、Web、モバイルデプロイに適しています。さらに複雑な出力も可能です。

- [YAMNet/classification](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1): よりシンプルな固定長フレーム入力の量子化バージョン (15600 サンプル)。521 の音声イベントクラスに対してスコアの単一のベクトルを返します。

### 入力

モデルには、範囲 `[-1.0, +1.0]` のモノラル 16kHz サンプルとして表された 0.975 秒の波形を含む、長さ 15600 の1-D `float32` テンソルまたは NumPy 配列を入力できます。

### 出力

モデルは、YAMNet でサポートされる AudioSet オントロジーの 521 クラスのそれぞれに対する予測スコアを含む形状 (1, 521) の 2-D `float32` テンソルを返します。スコアテンソルの列インデックス (0-520) は、YAMNet クラスマップを使用して、対応する AudioSet クラス名にマッピングされます。YAMNet クラスマップは、モデルファイルにパッケージ化された関連付けられたファイル `yamnet_label_list.txt` として提供されています。使用方法については、以下を参照してください。

### 適している使用方法

YAMNet は次の方法で使用できます。

- スタンドアロンの音声イベント分類器: さまざまな音声イベントにわたって合理的なベースラインを提供します。
- ハイレベルの特徴抽出器: 特定のタスクで少量のデータに対してトレーニングできる別のモデルの入力特徴として、YAMNet の 1024-D 埋め込み出力を使用できます。これにより、大量のラベル付けされたデータがなくても、大きいモデルをエンドツーエンドでトレーニングしなくても、専用の音声分類器をすばやく作成できます。
- ウォームスタート: YAMNet モデルパラメータを使用して、大きいモデルの一部を初期化し、微調整とモデル探索が高速化します。

### 制限事項

- YAMNet の分類器出力はクラス全体でキャリブレーションされていないため、直接出力を確立として処理することはできません。ほとんどの場合、特定のタスクで、タスク固有のデータを使用して、キャリブレーションを実行し、適切なクラス単位のしきい値とスケーリングを割り当てる必要があります。
- YAMNet は数百万もの YouTube 動画に対してトレーニングされました。これらの動画は非常に多様ですが、平均の YouTube 動画と特定のタスクに対して予想された音声入力との間には、まだドメインの不一致がある可能性があります。構築するシステムで YAMNet を使用できるようにするには、ある程度の微調整とキャリブレーションを実行する必要があります。

## モデルのカスタマイズ

用意されているトレーニング済みのモデルは、521 の異なる音声クラスを検出するようにトレーニングされています。クラスの一覧については、<a href="https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv">モデルリポジトリ</a>のラベルファイルを参照してください。

元のセットにないクラスを認識するようにモデルを再トレーニングするには転移学習と呼ばれる手法を使用します。たとえば、複数の鳥の鳴き声を検出するように、モデルを再トレーニングできます。これを行うには、トレーニングする新しいラベルごとに一連のトレーニング音声が必要です。推奨される方法は、[TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/audio_classification) ライブラリを使用することです。このライブラリは、カスタムデータセットと数行のコードを使用して、TensorFlow Lite モデルのトレーニングプロセスを簡素化します。また、転移学習が使用されるため、必要なトレーニングデータと時間が少なくなります。転移学習の例については、[音声認識の転移学習](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio)を参照してください。

## その他の資料とリソース

音声分類に関連する概念の詳細については、次のリソースを使用してください。

- [TensorFlow を使用した音声分類](https://www.tensorflow.org/tutorials/audio/simple_audio)
- [音声認識の転移学習](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio)
- [音声データ拡張](https://www.tensorflow.org/io/tutorials/audio)
