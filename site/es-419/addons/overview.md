<div align="center">
<img src="https://tensorflow.org/images/SIGAddons.png" width="60%"><br><br>
</div>

---

# Complementos de TensorFlow

**Los complementos de TensorFlow** son un repositorio de contribuciones que se ajustan a patrones de API bien establecidos, pero que implementan nuevas funciones que no están disponibles en el núcleo de TensorFlow. TensorFlow admite de forma nativa una gran cantidad de operadores, capas, métricas, pérdidas y optimizadores. Sin embargo, en un campo de rápido movimiento como ML, hay muchos desarrollos nuevos interesantes que no se pueden integrar en el núcleo de TensorFlow (porque su amplia aplicabilidad aún no está definida, o es utilizado principalmente por un pequeño grupo de la comunidad).

## Instalación

#### Compilación estable

Para instalar la última versión, ejecute el siguiente comando:

```
pip install tensorflow-addons
```

Para usar complementos:

```python
import tensorflow as tf
import tensorflow_addons as tfa
```

#### Compilación Nightly

También hay compilaciones nocturanas de los complementos de TensorFlow en el paquete pip `tfa-nightly` , que se basa en la última versión estable de TensorFlow. Las compilaciones nocturnas incluyen características más nuevas, pero pueden ser menos estables que las versiones oficiales.

```
pip install tfa-nightly
```

#### Instalación desde el  código fuente

También puede instalar desde el código fuente. Esto requiere el sistema de compilación [Bazel.](https://bazel.build/)

```
git clone https://github.com/tensorflow/addons.git
cd addons

# Para GPU Ops (Requiere CUDA 10.0 y CuDNN 7)
export TF_NEED_CUDA=1
export CUDA_TOOLKIT_PATH="/path/to/cuda10" (default: /usr/local/cuda)
export CUDNN_INSTALL_PATH="/path/to/cudnn" (default: /usr/lib/x86_64-linux-gnu)

# Este script  enlaza las dependencia de TensorFlow
python3 ./configure.py

bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tensorflow_addons-*.whl
```

## Conceptos básicos

#### API estandarizada dentro de subpaquetes

La experiencia del usuario y la capacidad de mantenimiento del proyecto son conceptos básicos en TF-Addons. Para lograr esto, necesitamos que nuestros complemetentos se ajusten a los patrones de API establecidos  en el núcleo de TensorFlow.

#### Operadores personalizados GPU / CPU

Una de las principales ventajas de los complementos de TensorFlow es que existen operadores precompilados. Si no se encuentra una instalación de CUDA 10, el operador recurrirá automáticamente a una implementación de CPU.

#### Mantenimiento de proxy

Los complementos se han diseñado para compartimentar subpaquetes y submódulos para que puedan recibir mantenimiento de usuarios con experiencia y un interés personal en ellos.

El derecho de mantenimiento de subpaquete solo se otorgará después de que se haya realizado una contribución sustancial con la finalidad de  limitar el número de usuarios con permiso de escritura. Las contribuciones pueden venir en forma de resolución de problemas, corrección de errores, documentación, código nuevo u optimización del código existente. El derecho a mantenimiento de submódulo se puede otorgar por una contribución menor, ya que esto no incluirá permisos de escritura en el repositorio.

Para obtener más información, consulte [el RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190308-addons-proxy-maintainership.md) sobre este tema.

#### Evaluación periódica de subpaquetes

Dada la naturaleza de este repositorio, los subpaquetes y submódulos pueden volverse cada vez menos útiles para la comunidad a medida que pasa el tiempo. Para mantener el repositorio sostenible, realizaremos revisiones semestrales de nuestro código para asegurarnos de que todo sigue perteneciendo al repositorio. Los factores que contribuyen a esta revisión serán:

1. Número de personas con derecho activas
2. Cantidad de uso del OSS
3. Cantidad de problemas o errores atribuidos al código
4. Si una solución mejor está disponible

La funcionalidad dentro de los complementos de TensorFlow se puede clasificar en tres grupos:

- **Sugerido** : API con mantenimiento; se recomienda su uso.
- **Desestimado** : hay una alternativa mejor disponible; la API se mantiene por razones históricas; o la API requiere mantenimiento y espera quedar obsoleto.
- **Obsoleto** : utilícelo bajo su propio riesgo; sujeto a ser eliminado.

El cambio de estado entre estos tres grupos es: sugerido &lt;-&gt; desestimado -&gt; obsoleto.

El período entre una API que se marca como obsoleta y se elimina será de 90 días. El fundamento es:

1. En el caso de que los complementos de TensorFlow se publiquen mensualmente, habrá 2-3 lanzamientos antes de que se elimine una API. Las notas de la versión podrían dar suficiente advertencia al usuario.

2. Plazo de 90 días  a soporte para corregir su código.

## Contribución

TF-Addons es un proyecto de código abierto dirigido por la comunidad. Como tal, el proyecto depende de contribuciones públicas, corrección de errores y documentación. Consulte las [pautas de contribución](https://github.com/tensorflow/addons/blob/master/CONTRIBUTING.md) para obtener una guía sobre cómo contribuir. Este proyecto se adhiere al [código de conducta de TensorFlow](https://github.com/tensorflow/addons/blob/master/CODE_OF_CONDUCT.md) . Al participar, se espera que se apegue a este código.

## Comunidad

- [Lista de distribución pública](https://groups.google.com/a/tensorflow.org/forum/#!forum/addons)
- [Notas de la reunión mensual de SIG](https://docs.google.com/document/d/1kxg5xIHWLY7EMdOJCdSGgaPu27a9YKpupUz2VTXqTJg)
    - Únase a nuestra lista de correo y reciba invitaciones del calendario a la reunión

## Licencia

[Licencia Apache 2.0](LICENSE)
