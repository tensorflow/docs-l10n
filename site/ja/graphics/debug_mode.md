# TensorFlow Graphics のデバッグモード

Tensorflow Graphics は、L2 正規化テンソルだけでなく、入力が特定の範囲にあることを期待する三角関数にも大きく依存しています。最適化中の更新によって、変数が、こういった関数が `Inf` または `NaN` 値を返してしまう値を取るようになってしまう場合があります。このような問題のデバッグを簡単に行えるようにするため、TensorFlow Graphics には、グラフにアサーションをインジェクトして適切な範囲と返される値の妥当性を確認するデバッグフラグが用意されています。このため計算が減速する可能性があるため、デフォルトではデバッグフラグは `False` に設定されています。

ユーザーは、`-tfg_debug` フラグを設定することで、コードをデバッグモードで実行することができます。このフラグの設定は、次の 2 つのモジュールをインポートすることによっても行えます。

```python
from absl import flags
from tensorflow_graphics.util import tfg_flags
```

この後に、次の行をコードに追加します。

```python
flags.FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value = True
```
