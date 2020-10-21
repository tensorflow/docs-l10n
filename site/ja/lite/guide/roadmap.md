# TensorFlow Lite のロードマップ

**更新日: 2020 年 4 月 18 日**

以下は、TensorFlow Lite の大まかな 2020 年度計画について説明しています。ここに示すロードマップはいつでも変更される可能性があり、以下に示される順序は優先順位を示すものではないことに注意してください。原則として、課題の優先順位は通常、影響を受けているユーザー数に基づいています。

このロードマップは、ユーザビリティ、パフォーマンス、最適化、および移植性という 4 つの主要区分に分けられています。ぜひロードマップにコメントを残し、[TF Lite ディスカッショングループ](https://groups.google.com/a/tensorflow.org/g/tflite)にフィードバックをご提供いただきますようお願いいたします。

## ユーザビリティ

- **演算子の対応範囲の拡大**
    - ユーザーフィードバックに基づく優先度の高い演算子の追加
- **TensorFlow Lite での TensorFlow 演算子の使用の改善**
    - Bintray（Android）と Cocoapods（iOS）経由でのビルド済みライブラリの提供
    - 演算子ストリップで一部の TF 演算子を使用する際のバイナリサイズの縮小
- **LSTM / RNN サポート**
    - LSTM と RNN 変換の完全サポート（Keras でのサポートを含む）
- **処理前および処理後のサポートライブラリと codegen ツール**
    - 共通する ML タスクですぐに利用できる API ビルディングブロック
    - より多くのモデル（NLP など）やプラットフォーム（iOS など）のサポート
- **Android Studio の統合**
    - Drag & drop TFLite models into Android Studio to generate model binding classes
- **オンデバイスでの制御フローとトレーニング**
    - パーソナライズ化と転移学習に焦点を当てたオンデバイストレーニングのサポート
- **TensorBoard による視覚化ツール**
    - TensorBoard に高度なツール機能を提供
- **Model Maker**
    - オブジェクト検出や BERT ベースの NLP タスクを含む、より多くのタスクのサポート
- **モデルと例の追加**
    - More examples to demonstrate model usage as well as new features and APIs, covering different platforms.
- **タスクライブラリ**
    - 事前構築済みのバイナリを提供したり、ユーザーフレンドリーなワークフローを作成したソースコードから構築できるようにするといった、C++ タスクライブラリのユーザービリティの改善
    - タスクライブラリの使用例を集めたリファレンスの公開
    - より多くのタスクの種類の提供
    - クロスプラットフォームサポートの改善と iOS 向けタスクの追加提供

## パフォーマンス

- **ツールの改善**
    - リリースごとのパフォーマンスゲインを追跡する公開ダッシュボード
- **CPU パフォーマンスの改善**
    - 高度に最適化された、畳み込みモデル用の新しい浮動小数点カーネルライブラリ
    - ファーストクラスの x86 サポート
- **NN API サポートの更新**
    - 新しい Android R NN API 機能、演算子、および型のフルサポート
- **GPU バックエンドの最適化**
    - Android での Vulkan サポート
    - 整数量子化モデルのサポート
- **Hexagon DSP バックエンド**
    - Per-channel quantization support for all models created through post-training quantization
    - 入力バッチサイズの動的サポート
    - LSTM を含む演算子のカバレッジ強化
- **Core ML バックエンド**
    - 起動時間の最適化
    - 動的量子化モデルのサポート
    - Float16 量子化モデルのサポート
    - 演算子の対応範囲の拡大

## 最適化

- **量子化**

    - （8b）固定小数点 RNN のトレーニング後量子化
    - （8b）固定小数点 RNN のトレーニング中量子化
    - トレーニング後のダイナミックレンジ量子化の品質とパフォーマンスの改善

- **プルーニング / スパース化**

    - TensorFlow Lite におけるスパースモデルの実行サポート - [進行中](https://github.com/tensorflow/model-optimization/issues/173)
    - 重みクラスタリング API

## 移植性

- **マイクロコントローラのサポート**
    - Add support for a range of 32-bit MCU architecture use cases for speech and image classification
    - ビジョンデータと音声データ用のサンプルコードとモデル
    - マイクロコントローラにおける TF Lite 演算子のフルサポート
    - CircuitPython サポートを含む、プラットフォームサポートの追加
