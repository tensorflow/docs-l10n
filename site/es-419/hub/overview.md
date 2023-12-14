# TensorFlow Hub

TensorFlow Hub es un repositorio y una biblioteca de acceso abiertos para el aprendizaje automático reutilizable. El repositorio [tfhub.dev](https://tfhub.dev) proporciona muchos modelos preentrenados: incrustaciones de texto, modelos de clasificación de imágenes, modelos TF.js/TFLite y mucho más. El repositorio es de acceso abierto para los [contribuyentes de la comunidad](https://tfhub.dev/s?subtype=publisher).

La biblioteca [`tensorflow_hub`](https://github.com/tensorflow/hub) le permite descargarlos y reutilizarlos en su programa de TensorFlow con un código mínimo.

```python
import tensorflow_hub as hub

model = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2")
embeddings = model(["The rain in Spain.", "falls",
                    "mainly", "In the plain!"])

print(embeddings.shape)  #(4,128)
```

## Próximos pasos

- [Encontrar modelos en tfhub.dev](https://tfhub.dev)
- [Publicar modelos en tfhub.dev](publish.md)
- Biblioteca TensorFlow Hub
    - [Instalar TensorFlow Hub](installation.md)
    - [Descripción general de la biblioteca](lib_overview.md)
- [Seguir tutoriales](tutorials)
