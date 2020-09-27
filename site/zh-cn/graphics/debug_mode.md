# TensorFlow Graphics 的调试模式

Tensorflow Graphics 很大程度上依赖于 L2 归一化张量以及期望其输入在一定范围内的三角函数。在优化过程中，更新可使这些变量采用使这些函数返回 `Inf` 或 `NaN` 值的值。为了简化此类问题的调试，TensorFlow Graphics 提供了一个调试标记，可将断言注入计算图中来检查正确的范围和返回值的有效性。由于这会减慢计算速度，因此默认情况下将调试标记设置为 `False`。

用户可以设置 `-tfg_debug` 标记以在调试模式下运行其代码。此外，首先导入以下两个模块，还能以编程方式设置该标记：

```python
from absl import flags
from tensorflow_graphics.util import tfg_flags
```

随后将以下行添加到代码中。

```python
flags.FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value = True
```
