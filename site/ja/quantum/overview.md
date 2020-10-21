# TensorFlow Quantum

TensorFlow Quantum (TFQ) は [量子機械学習](concepts.md)用の Python フレームワークです。アプリケーションフレームワークとして、TFQ を使用すると、量子アルゴリズムや機械学習アプリケーションの研究者は、すべて TensorFlow 内から Google の量子コンピューティングフレームワークを活用できます。

TensorFlow Quantum は、*量子データ*と*ハイブリッド量子古典モデル*の構築に重点を置き、<a href="https://github.com/quantumlib/Cirq" class="external">Circq</a> で設計された量子アルゴリズムとロジックを TensorFlow でインターリーブするツールを提供します。TensorFlow Quantum を効果的に使用するには、量子コンピューティングの基本的な理解が必要です。

TensorFlow Quantum を使い始めるには、[インストールガイド](install.md)、および実行可能な[ノートブックチュートリアル](./tutorials/hello_many_worlds.ipynb)をご覧ください。

## 設計

TensorFlow Quantum は、TensorFlow を量子コンピューティングハードウェアと統合するために必要なコンポーネントを実装するために、2 つのデータ型のプリミティブを導入します。

- *Quantum circuit*: これは、TensorFlow 内の Circq で定義された量子回路を表します。さまざまな実数値データポイントのバッチと同様に、さまざまなサイズの回路のバッチを作成します。
- *Pauli sum*: Cirq で定義されたパウリ演算子のテンソル積の線形結合を表します。回路と同様に、さまざまなサイズの演算子のバッチを作成します。

これらのプリミティブを使用して量子回路を表すと、TensorFlow Quantum は次の演算を提供します。

- 回路のバッチの出力分布からのサンプル。
- 回路のバッチに対する Pauli sum のバッチの期待値を計算します。 TFQ は、バックプロパゲーションと互換性のある勾配計算を実装します。
- 回路と状態のバッチをシミュレートします。量子回路全体ですべての量子状態の振幅を直接検査することは、大規模な実世界のシナリオでは非効率的ですが、研究者は状態シミュレーションを介して量子回路が状態をほぼ正確なレベルの精度にマッピングすることを理解することができます。

TensorFlow Quantum の実装の詳細については、[設計ガイド](design.md)をご覧ください。

## 問題の報告

バグの報告や機能リクエストには、<a href="https://github.com/tensorflow/quantum/issues" class="external">TensorFlow Quantum issue トラッカー</a>を使用してください。
