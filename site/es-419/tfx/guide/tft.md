# Biblioteca Transform para usuarios que no usan TFX

Transform está disponible como biblioteca independiente.

- [Introducción a TensorFlow Transform](https://www.tensorflow.org/tfx/transform/get_started)
- [Referencia de la API TensorFlow Transform](https://www.tensorflow.org/tfx/transform/api_docs/python/tft)

La documentación del módulo `tft` es el único módulo relevante para los usuarios de TFX. El módulo `tft_beam` es relevante solo cuando se usa Transform como biblioteca independiente. Normalmente, un usuario de TFX construye una función `preprocessing_fn` y al resto de las llamadas a la biblioteca Transform las hace el componente Transform.
