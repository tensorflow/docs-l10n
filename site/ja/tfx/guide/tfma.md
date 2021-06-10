# TensorFlow Model Analysis によるモデル品質の改善

## はじめに

モデルを開発しながら調整する場合、その変更によってモデルが改善するかどうかを確認する必要があります。精度を確認するだけでは不十分な場合があります。たとえば、問題の分類子があり、95% のインスタンスが陽性である場合、常に陽性を予測するだけで精度を改善できるかもしれませんが、それでは非常に強力な分類子は得られません。

## 概要

TensorFlow Model Analysis は、TFX でモデルを評価するための仕組みを提供することを目標としています。TensorFlow Model Analysis では、TFX パイプラインでモデル評価を実行し、結果として出力されるメトリクスを表示して、Jupyter ノートブックにプロットすることができます。具体的には、次を提供できます。

- training と holdout データセット全体で計算された[メトリクス](../model_analysis/metrics) と Next-day 検証
- 長期間にわたるメトリクスの追跡
- 異なる特徴量スライスにおけるモデルの品質パフォーマンス
- モデルのパフォーマンスが一貫して維持されるようにするための[モデル検証](../model_analysis/model_validations)

## 次のステップ

[TFMA チュートリアル](../tutorials/model_analysis/tfma_basic)をお試しください。

[GitHub](https://github.com/tensorflow/model-analysis) ページで、サポートされている[メトリクスとグラフ](../model_analysis/metrics)、および関連するノートブックの[視覚化](../model_analysis/visualizations)に関する詳細をご覧ください。

[インストール](../model_analysis/install)ガイドと[基礎](../model_analysis/get_started)ガイドでは、スタンドアロンパイプラインでの[セットアップ](../model_analysis/setup)方法に関する情報と例が掲載されています。TFMA は TFX の [Evaluator](evaluator.md) コンポーネントでも使用されているため、これらのリソースは TFX の使用開始にも役立ちます。
