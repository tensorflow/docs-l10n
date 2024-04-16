# Empezar con la optimización del modelo de TensorFlow

## 1. Elija el mejor modelo para la tarea

Según la tarea, deberá buscar el equilibrio entre la complejidad y el tamaño del modelo. Si su tarea requiere una alta precisión, es posible que necesite un modelo grande y complejo. Para tareas que requieren menos precisión, es mejor usar un modelo más pequeño porque no solo usan menos espacio en disco y memoria, sino que también son generalmente más rápidos y más eficientes energéticamente.

## 2. Modelos preoptimizados

Vea si algún [modelo preoptimizado de TensorFlow Lite](https://www.tensorflow.org/lite/models) existente proporciona la eficiencia que requiere su aplicación.

## 3. Herramientas posentrenamiento

Si no puede usar un modelo preentrenado para su aplicación, intente usar las [herramientas de cuantificación posentrenamiento de TensorFlow Lite](./quantization/post_training) durante la [conversión de TensorFlow Lite](https://www.tensorflow.org/lite/convert), que pueden optimizar su modelo de TensorFlow ya entrenado.

Consulte el [tutorial de cuantificación posentrenamiento](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/post_training_quant.ipynb) para obtener más información.

## Próximos pasos: herramientas durante el entrenamiento

Si estas soluciones simples no satisfacen sus necesidades, es posible que deba usar técnicas de optimización durante el entrenamiento. [Optimice aún más](optimize_further.md) con nuestras herramientas durante el entrenamiento y vaya más a fondo.
