# TensorFlow Lite Task ライブラリ

TensorFlow Lite Task ライブラリには、強力で使いやすいタスク固有の一連のライブラリが含まれているので、アプリ開発者はTFLite を使用して機械学習を利用できます。画像分類や質疑応答など、一般的な機械学習タスク用に最適化された、すぐに使用できるモデルインターフェイスを提供します。モデルインターフェイスは、最高のパフォーマンスと使いやすさを実現するために、タスクごとに設計されています。Task ライブラリはクロスプラットフォームで動作し、Java、C++、および Swift でサポートされています。

## Task ライブラリに期待すること

- **機械学習の専門家ではない人向けに分かりやすく明確に定義された API ** <br>推論はわずか5行のコードで実行できます。Task ライブラリの強力で使いやすい API をビルディングブロックとして使用して、モバイルデバイスで TFLite を使用して機械学習を簡単に開発できるようにします。

- **複雑かつ一般的なデータ処理** <br> モデルに必要とされるデータ形式にデータを変換するための一般的なビジョンと自然言語処理ロジックをサポートします。 トレーニングと推論のために同じ共有可能な処理ロジックを提供します。

- **高性能ゲイン** <br> データ処理にかかる時間は数ミリ秒以内なので、TensorFlow Lite を使用した高速推論が可能になります。

- **拡張性とカスタマイズ** <br> Task ライブラリインフラストラクチャが提供するすべての利点を活用して、独自の Android/iOS 推論 API を簡単に構築できます。

## サポートされているタスク

以下は、サポートされているタスクタイプのリストです。今後、ますます多くのユースケースが利用可能になり、リストに追加される予定です。

- **Vision API**

    - [ImageClassifier](image_classifier.md)
    - [ObjectDetector](object_detector.md)
    - [ImageSegmenter](image_segmenter.md)

- **Natural Language (NL) API**

    - [NLClassifier](nl_classifier.md)
    - [BertNLCLassifier](bert_nl_classifier.md)
    - [BertQuestionAnswerer](bert_question_answerer.md)

- **カスタム API**

    - Task API インフラストラクチャを拡張し、[カスタマイズされた API](customized_task_api.md) を構築します。
