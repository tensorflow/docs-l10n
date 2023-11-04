# TensorFlow Hub

O TensorFlow Hub é um repositório e biblioteca abertos para aprendizado de máquina reutilizável. O repositório [tfhub.dev](https://tfhub.dev) fornece diversos modelos pré-treinados: embeddings de texto, modelos de classificação de imagens, modelos de TF.js/TFLite e muito mais. O repositório é aberto para [contribuidores da comunidade](https://tfhub.dev/s?subtype=publisher).

A biblioteca [`tensorflow_hub`](https://github.com/tensorflow/hub) permite baixá-los e reutilizá-los em seu programa do TensorFlow com uma quantidade mínima de código.

```python
import tensorflow_hub as hub

model = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2")
embeddings = model(["The rain in Spain.", "falls",
                    "mainly", "In the plain!"])

print(embeddings.shape)  #(4,128)
```

## Próximos passos

- [Encontre modelos em tfhub.dev](https://tfhub.dev)
- [Publique modelos em tfhub.dev](publish.md)
- Biblioteca do TensorFlow Hub
    - [Instalação do TensorFlow Hub](installation.md)
    - [Visão geral da biblioteca](lib_overview.md)
- [Confira tutoriais](tutorials)
