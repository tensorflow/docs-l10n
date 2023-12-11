# Biblioteca Transform para usuários não-TFX

A biblioteca Transform está disponível como uma biblioteca independente.

- [Introdução ao TensorFlow Transform](https://www.tensorflow.org/tfx/transform/get_started)
- [Referência da API Transform do TensorFlow](https://www.tensorflow.org/tfx/transform/api_docs/python/tft)

A documentação do módulo `tft` é o único módulo relevante para usuários do TFX. O módulo `tft_beam` é relevante apenas ao usar Transform como uma biblioteca independente. Normalmente, um usuário TFX cria uma `preprocessing_fn` e o restante das chamadas da biblioteca Transform são feitas pelo componente Transform.
