# Modo de depuración para TensorFlow Graphics

Tensorflow Graphics depende en gran medida de tensores normalizados L2, así como de funciones trigonométricas que esperan que sus entradas estén en un intervalo determinado. Durante la optimización, una actualización puede hacer que estas variables tomen valores que hagan que estas funciones devuelvan valores `Inf` o `NaN`. Para simplificar la depuración de estos problemas, TensorFlow Graphics ofrece un indicador de depuración que inyecta aserciones al gráfico para comprobar los intervalos correctos y la validez de los valores devueltos. Como esto puede ralentizar los cálculos, la marca de depuración está configurada en `False` de forma predeterminada.

Los usuarios pueden configurar la marca `-tfg_debug` para ejecutar su código en modo de depuración. La marca también se puede configurar mediante programación si primero se importan estos dos módulos:

```python
from absl import flags
from tensorflow_graphics.util import tfg_flags
```

y luego se agrega la siguiente línea al código.

```python
flags.FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value = True
```
