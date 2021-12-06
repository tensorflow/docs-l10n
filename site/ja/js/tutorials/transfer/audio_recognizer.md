# 音声識別器の転移学習

このチュートリアルでは、TensorFlow.js を使ってブラウザでトレーニングするカスタム音声分類器を構築する方法を学習します。音を鳴らし、その分類器を使用してブラウザ内のスライダーを制御します。

転移学習を使用して、比較的少ないトレーニングデータで短い音を分類するモデルを作成します。ここでは、[音声コマンド認識](https://github.com/tensorflow/tfjs-models/tree/master/speech-commands)を行うトレーニング済みのモデルを使用します。このモデルを土台に、独自のカスタム音声クラスを認識する新しいモデルをトレーニングします。

このチュートリアルは Colab として提供されています。[こちらのリンクから Codelab にアクセスしてください](https://github.com/tensorflow/tfjs-models/tree/master/speech-commands)。
