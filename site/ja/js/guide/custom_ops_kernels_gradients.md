# TensorFlow.js でのカスタム演算、カーネル、勾配の記述

## Overview

このガイドでは、TensorFlow.js でカスタム演算（ops）、カーネル、勾配を定義するためのメカニズムの概要を説明します。これは、主要な概念の概要と、実際の概念を示すコードへのポインターを提供することを目的としています。

### このガイドの対象者

これは、TensorFlow.js の内部に触れるかなり高度なガイドであり、次のグループの人々に特に役立つ場合があります。

- さまざまな数学演算の動作のカスタマイズに関心のある TensorFlow.js 上級ユーザー（たとえば、既存の勾配の実装をオーバーライドする研究者や、ライブラリにない機能にパッチを適用する必要があるユーザー）
- TensorFlow.js を拡張するライブラリを構築するユーザー（たとえば、TensorFlow.js プリミティブまたは新しい TensorFlow.js バックエンドの上に構築された一般的な線形代数ライブラリ）。
- これらのメカニズムがどのように機能するかについての一般的な概要を知りたい、tensorflow.js に新しい操作を提供することに関心のあるユーザー。

内部実装メカニズムを説明するため、一般的な使用法のガイド**ではありません**。 TensorFlow.js を使用するために、これらのメカニズムを理解する必要はありません。

このガイドを最大限に活用するには、TensorFlow.js ソースコードを読むことに慣れている（または試してみる）必要があります。

## 用語

このガイドでは、いくつかの重要な用語をあらかじめ説明するのに役立ちます。

**演算（Ops）** — 1 つ以上のテンソルを出力として生成する 1 つ以上のテンソルに対する数学演算。 Ops は「高レベル」なコードであり、他の Ops を使用してロジックを定義できます。

**カーネル** — 特定のハードウェア/プラットフォーム機能に関連付けられた op 固有の実装。カーネルは「低レベル」でバックエンド固有です。演算からカーネルへの 1 対 1 のマッピングがある演算もあれば、複数のカーネルを使用する演算もあります。

**勾配** **/ GradFunc**  — ある入力に関してその関数の導関数を計算する **op/kernel** の「バックワードモード」定義。勾配は「高レベル」コード（バックエンド固有ではない）であり、他の ops またはカーネルを呼び出すことができます。

**カーネルレジストリ**- **（カーネル名、バックエンド名）**タプルからカーネル実装へのマップ。

**勾配レジストリ** — **カーネル名から勾配実装へ**のマップ。

## コード編成

[演算](https://github.com/tensorflow/tfjs/tree/master/tfjs-core/src/ops)と[勾配](https://github.com/tensorflow/tfjs/tree/master/tfjs-core/src/gradients)は [tfjs-core](https://github.com/tensorflow/tfjs/tree/master/tfjs-core) で定義されます。

カーネルはバックエンド固有であり、それぞれのバックエンドフォルダーで定義されます（例: [tfjs-backend-cpu](https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-cpu/src/kernels) ）。

カスタム演算、カーネル、勾配は、これらのパッケージ内で定義する必要はありません。ただし、実装では同様の記号を使用することがよくあります。

## カスタム演算の実装

カスタム演算を考える 1 つの方法は、テンソル出力を返す JavaScript 関数 と同じです。多くの場合、入力としてテンソルが使用されます。

- 一部の演算は、既存の演算に関して完全に定義でき、これらの関数を直接インポートして呼び出す必要があります。 [こちらが例](https://github.com/tensorflow/tfjs/blob/1bec37b9364df6164a4a0ad64c29e0859382f0b4/tfjs-core/src/ops/moving_average.ts)です。
- 演算の実装は、特定のカーネルをバックエンドにディスパッチすることもできます。これは `Engine.runKernel` を介して実行され、「カスタムカーネルの実装」セクションで詳しく説明します。 [こちらが例](https://github.com/tensorflow/tfjs/blob/1bec37b9364df6164a4a0ad64c29e0859382f0b4/tfjs-core/src/ops/sqrt.ts)です。

## カスタムカーネルの実装

バックエンド固有のカーネル実装により、特定の演算のロジックの最適化された実装が可能になります。カーネルは、[`tf.engine().runKernel()`](https://cs.opensource.google/tensorflow/tfjs/+/master:tfjs-core/src/engine.ts?q=runKernel&ss=tensorflow%2Ftfjs:tfjs-core%2F) を呼び出す演算によって呼び出されます。カーネルの実装は、次の 4 つで定義されます。

- カーネル名。
- カーネルのバックエンドはに実装されています。
- 入力: カーネル関数へのテンソル引数。
- 属性: カーネル関数への非テンソル引数。

[カーネル実装](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-cpu/src/kernels/Square.ts)の例を次に示します。実装に使用される規則はバックエンド固有であり、各特定のバックエンドの実装とドキュメントを見ると最もよく理解できます。

通常、カーネルはテンソルよりも低いレベルで動作し、代わりにメモリに対して直接読み取りと書き込みを行い、最終的に tfjs-core によってテンソルにラップされます。

カーネルが実装されると、tfjs-core の [`registerKernel` 関数](https://cs.opensource.google/tensorflow/tfjs/+/master:tfjs-core/src/kernel_registry.ts?q=registerKernel&ss=tensorflow%2Ftfjs:tfjs-core%2F)を使用して TensorFlow.js に登録できます。カーネルを動作させたいすべてのバックエンドにカーネルを登録できます。登録すると、カーネルは `tf.engine().runKernel(...)` で呼び出すことができ、TensorFlow.js は現在のアクティブなバックエンドで実装にディスパッチするようにします。

## カスタム勾配の実装

勾配は通常、特定のカーネルに対して定義されます（`tf.engine().runKernel(...)` の呼び出しで使用されるのと同じカーネル名で識別されます）。これにより、tfjs-core はレジストリを使用して、実行時に任意のカーネルの勾配定義を検索できます。

カスタム勾配の実装は、次の場合に役立ちます。

- ライブラリに存在しない可能性のある勾配定義を追加する
- 既存の勾配定義をオーバーライドして、特定のカーネルの勾配計算をカスタマイズする。

[ここでは勾配実装](https://github.com/tensorflow/tfjs/tree/master/tfjs-core/src/gradients)例を見ることができます。

特定の呼び出しに勾配を実装したら、tfjs-core の [`registerGradient` 関数](https://cs.opensource.google/tensorflow/tfjs/+/master:tfjs-core/src/kernel_registry.ts?q=registerGradient&ss=tensorflow%2Ftfjs:tfjs-core%2F)を使用して、TensorFlow.js に登録できます。

レジストリをバイパスする（したがって、任意の方法で任意の関数の勾配を計算できるようにする）カスタム勾配を実装するためのもう1つのアプローチは、[tf.customGrad](https://js.tensorflow.org/api/latest/#customGrad) を使用することです。

customGrad を使用する[ライブラリ内の操作の例](https://github.com/tensorflow/tfjs/blob/f111dc03a87ab7664688011812beba4691bae455/tfjs-core/src/ops/losses/softmax_cross_entropy.ts#L64)はこちらをご覧ください。
