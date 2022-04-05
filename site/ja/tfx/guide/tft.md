# TFX を使用しないユーザー向けの Transform ライブラリ

Transform はスタンドアロンライブラリとして提供されています。

- [TensorFlow Transform の基礎](/tfx/transform/get_started)
- TensorFlow Transform API リファレンス

`tft` モジュールドキュメントは TFX ユーザーに関連する唯一のモジュールです。`tft_beam` モジュールは、Transform をスタンドアロンライブラリとして使用する場合にのみ関連性があります。通常、TFX ユーザーは `preprocessing_fn` を構築し、残りの Transform ライブラリ呼び出しは、Transform コンポーネントによって行われます。
