# XLA でのエイリアシング

このドキュメントでは、XLA のエイリアシング API について説明します。XLA プログラムをビルドするときに、入力バッファと出力バッファの間で必要なエイリアシングを指定できます。

## コンパイル時のエイリアシングの定義

例として、入力に `1` を追加するだけの簡単な HLO モジュールについて考察します。

```
HloModule increment

ENTRY entry {
  %p = f32[] parameter(0)
  %c = f32[] constant(1)
  ROOT %out = f32[] add(%p, %c)
}
```

このモジュールは、2 つの 4 バイトバッファを割り当てます。1 つは入力 `%p` に、もう 1 つは出力 `%out` に割り当てます。

ただし、インプレースアップグレードを実行することが望ましい場合がよくあります（たとえば、式を生成するフロントエンドで計算後に入力変数が有効でなくなった場合など（例：増分 `p++` ））。

このような更新を効率的に実行するために、入力エイリアシングを指定できます。

```
HloModule increment, input_output_alias={ {}: 0 }

ENTRY entry {
  %p = f32[] parameter(0)
  %c = f32[] constant(1)
  ROOT %out = f32[] add(%p, %c)
}
```

この形式は、出力全体（`{}` でマークされている）が入力パラメータ `0` にエイリアスされることを指定します。

プログラムでエイリアシングを指定するには、 [`XlaBuilder::SetUpAlias`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h) API を参照してください。

## ランタイムのエイリアシングの定義

前のステップで定義されたエイリアシングは、*コンパイル*中に指定されます。実行中に、[`LocalClient::RunAsync`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/local_client.h) API を使用して実際にバッファを提供するかどうかを選択できます。

プログラムへの入力バッファは [`ExecutionInput`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/executable.h) でラップされ、`MaybeOwningDeviceMemory` のツリーが含まれます。メモリが *owning* として指定されている場合（バッファの所有権は XLA ランタイムに渡される場合）、バッファは実際に提供され、コンパイル時のエイリアシング API の要求に応じて、インプレースアップグレードが実行されます。

ただし、コンパイル時にエイリアスされるバッファがランタイムに*提供されない場合*、*copy-protection* が開始されます。追加の出力バッファ `O` が割り当てられ、エイリアス化されることを意図した入力バッファ `P` の内容が `O` にコピーされます。（したがって、プログラムは、ランタイムにバッファ `O` が提供されたかのように効果的に実行できます）。

## フロントエンド相互運用

### TF/XLA

XLA でコンパイルされた TensorFlow プログラムのクラスタでは、すべてのリソース変数の更新はコンパイル時にエイリアスされます（ランタイムのエイリアスは、他にリソース変数テンソルへの参照を保持しているかどうかによって異なります）。
