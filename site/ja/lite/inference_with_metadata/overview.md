# メタデータを使用する TensorFlow Lite 推論

[メタデータを使用するモデル](../convert/metadata.md)の推論は、コードを数行記述するだけで簡単に実行できます。TensorFlow Lite メタデータには、モデルの機能とモデルの使用方法に関する豊富な説明が含まれています。[TensorFlow Lite Android コードジェネレータ](codegen.md#mlbinding)や [Android Studio ML Binding 機能](codegen.md#codegen)を使用するなど、コードジェネレータが自動的に推論コードを生成できるようになります。また、カスタム推論パイプラインを構成するために使用することもできます。

## ツールとライブラリ

TensorFlow Lite は、次のようにデプロイメント要件のさまざまな層に対応するためのさまざまなツールとライブラリを提供します。

### Android コードジェネレータを使用してモデルインターフェイスを生成する

メタデータを使用して TensorFlow Lite モデルに必要とされる Android ラッパーコードを自動的に生成する方法は 2 つあります。

1. [Android Studio ML Model Binding](codegen.md#mlbinding) は、Android Studio 内で利用可能なツールで、グラフィカルインターフェイスを介して TensorFlow Lite モデルをインポートします。Android  Studio は、プロジェクトの設定を自動的に構成し、モデルのメタデータに基づいてラッパークラスを生成します。

2. [TensorFlow Lite Code Generator](codegen.md) は、メタデータに基づいてモデルインターフェイスを自動的に生成する実行可能ファイルです。現在、Android と Java をサポートしています。ラッパーコードにより、`ByteBuffer`と直接対話する必要がなくなります。代わりに、開発者は`Bitmap`や`Rect`などの型付きオブジェクトを使用して TensorFlow Lite モデルと対話できます。また、Android Studio ユーザーは、[Android Studio ML Binding ](codegen.md#generate-code-with-android-studio-ml-model-binding)を介して codegen 機能にアクセスすることもできます。

### TensorFlow Lite Task Library で既成の API を活用する

[TensorFlow Lite Task Library](task_library/overview.md) は、画像分類、質問応答など、一般的な機械学習タスク用に最適化された、既成のモデルインターフェイスを提供します。モデルインターフェイスは、最高のパフォーマンスと使いやすさを実現するために、タスクごとに特別に設計されています。Task Library はクロスプラットフォームで動作し、Java、C++、および Swift でサポートされています。

### TensorFlow Lite Support Library を使用してカスタム推論パイプラインを構築する

[TensorFlow Lite Support Library](lite_support.md) は、モデルインターフェイスのカスタマイズと推論パイプラインの構築に役立つクロスプラットフォームライブラリです。これには、前処理および後処理とデータ変換を実行するためのさまざまな util メソッドとデータ構造が含まれています。また、TF.Image や TF.Text などの TensorFlow モジュールの動作に一致するように設計されており、トレーニングから推論までの一貫性を保証します。

## メタデータを含む事前トレーニング済みモデルを探索する

ビジョンタスクとテキストタスク向けのメタデータを含む事前トレーニング済みモデルをダウンロードするには [TensorFlow Lite ホステッドモデル](https://www.tensorflow.org/lite/guide/hosted_models)と [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite) を参照してください。また、[メタデータの視覚化](../convert/metadata.md#visualize-the-metadata)のさまざまなオプションも参照してください。

## TensorFlow Lite サポート GitHub リポジトリ

その他の例とソースコードについては、[TensorFlow Lite サポート GitHub リポジトリ](https://github.com/tensorflow/tflite-support)を参照してください。[新しい GitHub issue](https://github.com/tensorflow/tflite-support/issues/new) を作成して、フィードバックを共有してください。
