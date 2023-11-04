# Modo de depuração do TensorFlow Graphics

O Tensorflow Graphics depende bastante dos tensores normalizados L2, funções trigonométricas que esperam que suas entradas estejam em um determinado intervalo. Durante a otimização, uma atualização pode fazer essas variáveis aceitarem valores que acabem fazendo as funções retornarem valores `Inf` ou `NaN`. Para facilitar a depuração desse tipo de problema, o TensorFlow Graphics conta com um sinalizador de depuração que injeta asserções no grafo para checar os intervalos corretos e a validade dos valores retornados. Como isso pode causar lentidão nas computações, o sinalizador de depuração é definido como `False` por padrão.

Os usuários podem definir o sinalizador `-tfg_debug` para executar o código no modo de depuração. O sinalizador também pode ser definido no código pela importação destes dois módulos:

```python
from absl import flags
from tensorflow_graphics.util import tfg_flags
```

e pelo acréscimo da seguinte linha de código:

```python
flags.FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value = True
```
