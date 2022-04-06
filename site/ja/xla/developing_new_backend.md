# 新しい XLA 用バックエンドの開発

この予備ガイドは、効率的な方法で TensorFlow を簡単にハードウェアに対応させたいと考えている早期採用者を対象としています。このガイドはステップごとに手順を説明するものではなく、[LLVM](http://llvm.org)、[Bazel](https://www.tensorflow.org/?hl=en)、および TensorFlow に関する知識を前提としています。

XLA は、新しいアーキテクチャやアクセラレータが、TensorFlow グラフを実行するバックエンドを作成するために実装できる抽象的なインターフェースを提供します。XLA への対応は、新しいハードウェア向けに既存のあらゆる TensorFlow Op を実装するのに比べ、はるかに単純でスケーラブルです。

ほとんどの実装は、次のいずれかのシナリオに該当します。

1. [LLVM](http://llvm.org) バックエンドの有無に関係なく、公式に XLA でサポートされていない既存の CPU アーキテクチャ。
2. LLVM バックエンドが存在する、CPU ではないハードウェア。
3. LLVM バックエンドが存在しない、CPU ではないハードウェア。

> 注意: LLVM バックエンドとは、正式リリースの LLVM バックエンドか自社開発のカスタム LLVM バックエンドのいずれかを指します。

## シナリオ 1: 公式に XLA でサポートされていない既存の CPU アーキテクチャ

このシナリオの場合、既存の [XLA CPU バックエンド](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/cpu/)をよく見ることから始めます。XLA の CPU バックエンド間の主な違いは、LLVM が生成するコードであることから、XLA では LLVM を使ってさまざまな CPU に TensorFlow を簡単に対応させることができます。Google は、x64 と ARM64 アーキテクチャを対象に XLA をテストしています。

ハードウェアベンダーがそのハードウェア向けの LLVM バックエンドを用意している場合、そのバックエンドを XLA でビルドされLLVM にリンクすることは簡単です。JIT モードでは、XLA CPU バックエンドはホスト CPU のコードを発行します。事前コンパイルでは、[`xla::AotCompilationOptions`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/compiler.h) で、ターゲットアーキテクチャを構成する LLVM Triple が提供されます。

既存の LLVM バックエンドがなくても別のコードジェネレータが存在する場合は、既存の CPU バックエンドのほとんどを再利用できる可能性があります。

## シナリオ 2: LLVM バックエンドが存在する、CPU ではないハードウェア

LLVM IR を発行する既存の [`xla::CPUCompiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/cpu/cpu_compiler.cc) と [`xla::GPUCompiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/gpu/nvptx_compiler.cc) クラスを基に、新しい [`xla::Compiler`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/compiler.h) 実装をモデルすることが可能です。ハードウェアの性質によって異なりますが、LLVM IR 生成の多くは変更しなければならないものの、多数のコードを既存のバックエンドに共有することが可能です。

XLA の [GPU バックエンド](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/gpu/)が良い参考になります。GPU バックエンドは CPU ではない ISA に対応するため、コード生成の一部の側面は GPU ドメインに特有です。Hexagon（アップストリーム LLVM バックエンド）のような DSP といったその他のハードウェアは、LLVM IR 発行ロジックの一部を再利用できますが、他の部分は一意となります。

## シナリオ 3: LLVM バックエンドが存在しない、CPUではないハードウェア

LLVM を使用できない場合の最善のオプションは、目的のハードウェア向けに XLA 用の新しい バックエンドを実装することです。このオプションには多大な労力が伴います。以下は、実装が必要なクラスです。

- [`StreamExecutor`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/stream_executor/stream_executor.h): 多くのデバイスでは、`StreamExecutor` のすべてのメソッドが必要になることはありません。詳細は既存の `StreamExecutor` の実装をご覧ください。
- [`xla::Compiler`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/compiler.h): このクラスは、HLO コンピュテーションのコンパイルを `xla::Executable` にカプセル化します。
- [`xla::Executable`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/executable.h): このクラスは、コンパイルされたコンピュテーションをプラットフォーム上で起動するために使用します。
- [`xla::TransferManager`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/transfer_manager.h): このクラスは、特定のバイスメモリのハンドルから XLA のリテラルデータを構築するための、プラットフォーム特有の仕組みをバックエンドが提供できるようにします。言い換えれば、ホストからデバイスまたはその逆方向のデータ転送処理をカプセル化します。
