# Instalación

## Instalar tensorflow_hub

La biblioteca `tensorflow_hub` se puede instalar junto con TensorFlow 1 y TensorFlow 2. Le recomendamos a los usuarios nuevos que comiencen con TensorFlow 2 de inmediato y que los usuarios actuales se actualicen.

### Uso con TensorFlow 2

Use [pip](https://pip.pypa.io/) para [instalar TensorFlow 2](https://www.tensorflow.org/install) como de costumbre. (También puede consultarlo para obtener instrucciones adicionales sobre la compatibilidad con GPU). Luego instale una versión actual de [`tensorflow-hub`](https://pypi.org/project/tensorflow-hub/) junto a ella (debe ser 0.5.0 o posterior).

```bash
$ pip install "tensorflow>=2.0.0"
$ pip install --upgrade tensorflow-hub
```

La API estilo TF1 de TensorFlow Hub funciona con el modo de compatibilidad v1 de TensorFlow 2.

### Uso heredado con TensorFlow 1

TensorFlow 1.15 es la única versión de TensorFlow 1.x que aún es compatible con la biblioteca `tensorflow_hub` (a partir de la versión 0.11.0). De forma predeterminada, TensorFlow 1.15 tiene un comportamiento compatible con TF1, pero contiene muchas características de TF2 internas para permitir cierto uso de las API de estilo TF2 de TensorFlow Hub.

```bash
$ pip install "tensorflow>=1.15,<2.0"
$ pip install --upgrade tensorflow-hub
```

### Uso de versiones preliminares

Los paquetes pip `tf-nightly` y `tf-hub-nightly` se crean automáticamente a partir del código fuente en github, sin pruebas de lanzamiento. Esto permite a los desarrolladores probar el código más reciente sin [compilarlo desde el código fuente](build_from_source.md).

```bash
$ pip install tf-nightly
$ pip install --upgrade tf-hub-nightly
```

## Próximos pasos

- [Descripción general de la biblioteca](lib_overview.md)
- Tutoriales:
    - [Clasificación de texto](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/tf2_text_classification.ipynb)
    - [Clasificación de imágenes](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/tf2_image_retraining.ipynb)
    - Ejemplos adicionales [en GitHub](https://github.com/tensorflow/hub/blob/master/examples/README.md)
- Encuentrar modelos en [tfhub.dev](https://tfhub.dev).
