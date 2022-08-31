# XLA アーキテクチャ

<div style="width:50%; margin:auto; margin-bottom:10px; margin-top:20px;"> <img style="width:50%" src="./images/xlalogo.png">
</div>

## XLAを開発した理由とは

XLA を TensorFlow と連携させるに当たって、以下のようないくつかの目標がありました。

- *Improve execution speed.* Compile subgraphs to reduce the execution time of short-lived Ops to eliminate overhead from the TensorFlow runtime, fuse pipelined operations to reduce memory overhead, and specialize to known tensor shapes to allow for more aggressive constant propagation.

- *Improve memory usage.* Analyze and schedule memory usage, in principle eliminating many intermediate storage buffers.

- *Reduce reliance on custom Ops.* Remove the need for many custom Ops by improving the performance of automatically fused low-level Ops to match the performance of custom Ops that were fused by hand.

- *Reduce mobile footprint.* Eliminate the TensorFlow runtime by ahead-of-time compiling the subgraph and emitting an object/header file pair that can be linked directly into another application. The results can reduce the footprint for mobile inference by several orders of magnitude.

- *Improve portability.* Make it relatively easy to write a new backend for novel hardware, at which point a large fraction of TensorFlow programs will run unmodified on that hardware. This is in contrast with the approach of specializing individual monolithic Ops for new hardware, which requires TensorFlow programs to be rewritten to make use of those Ops.

## XLA の仕組み

XLA への入力言語は「HLO IR」または単に HLO（High Level Operations: 高レベル演算）と呼ばれています。HLO のセマンティクスは[演算セマンティクス](./operation_semantics.md) ページで説明されています。HLO を[コンパイラ IR](https://www.tensorflow.org/?hl=en)として考えるのが最もわかりやすいでしょう。

XLA は HLO で定義されたグラフ（「コンピュテーション」）を取り、それらを様々なアーキテクチャで使用できる機械指示にコンパイルします。別のバックエンドに移して[新たな HW アーキテクチャで動作させる](https://www.tensorflow.org/xla/jit)ことが容易であるという点で、XLAはモジュール化されていると言えます。x64 や AMR64 用の CPU バックエンドや NVIDIA GPU バックエンドは、TensorFlow のソースツリーで参照できます。

次の図では、XLAの内部で行われているコンパイル処理を示しています。

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;"><img src="./images/how-does-xla-work.png"></div>

XLAには、[CSE](https://en.wikipedia.org/wiki/Common_subexpression_elimination)、ターゲットに依存しない演算の融合、コンピュテーションにランタイムメモリを割り当てるためのバッファ分析など、ターゲットに依存しないいくつかの最適化と分析パスがあります。

ターゲットに依存しないステップの後、XLA は HLO コンピュテーションをバックエンドに送信します。バックエンドはさらに HLO レベルの最適化を行いますが、ここではターゲットに特化した情報とニーズを念頭において実行されます。たとえば、XLA GPU バックエンドは、GPU プログラミングモデルに特に有用な演算融合を実行し、計算をストリームに分割する方法を決定することがあります。この段階では、バックエンドは特定の演算またはそれによって生じた組み合わせを最適化されたライブラリ呼び出しに対してにパターンマッチングすることもあります。

次のステップは、ターゲット固有のコードの生成です。XLA に含まれる CPU と GPU のバックエンドは [LLVM](https://www.tensorflow.org/xla/developing_new_backend) を使用して、低レベル IR、最適化、およびコード生成を行います。これらのバックエンドは XLA HLO コンピュテーションを効率的に表現するために必要な LLVM IR を発行し、その上で、LLVM を呼び出して、この LLVM IR からネイティブコードを発行します。

GPU バックエンドは現在、LLVM NVPTX バックエンド経由で NVIDIA をサポートしています。CPU バックエンドは複数の CPU ISA をサポートしています。
