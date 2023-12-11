# Complemento de malla

## Descripción general

Las mallas y las nubes de puntos son tipos de datos importantes y potentes que permiten representar formas tridimensionales y que han sido ampliamente estudiados en el campo de la visión artificial y la computación gráfica. Los datos 3D son cada vez más comunes y los investigadores se enfrentan a nuevos problemas, como la reconstrucción de geometrías 3D a partir de datos 2D, la segmentación semántica de nubes de puntos 3D, la alineación o el modelado de objetos 3D, etc. Por lo tanto, la visualización de los resultados, especialmente durante la fase de entrenamiento, es fundamental para comprender mejor el rendimiento del modelo.

![Mesh Plugin in TensorBoard](https://storage.googleapis.com/tensorflow-graphics/git/readme/tensorboard_plugin.jpg){width="100%"}

Este complemento pretende mostrar nubes de puntos 3D o mallas (nubes de puntos trianguladas) en TensorBoard. Además, permite al usuario interactuar con los objetos renderizados.

## API de resumen

Tanto una malla como una nube de puntos se pueden representar mediante un conjunto de tensores. Por ejemplo, se puede ver una nube de puntos como un conjunto de coordenadas 3D de los puntos y algunos colores asociados con cada punto.

```python
from tensorboard.plugins.mesh import summary as mesh_summary
...

point_cloud = tf.constant([[[0.19, 0.78, 0.02], ...]], shape=[1, 1064, 3])
point_colors = tf.constant([[[128, 104, 227], ...]], shape=[1, 1064, 3])

summary = mesh_summary.op('point_cloud', vertices=point_cloud, colors=point_colors)
```

NOTA: El tensor `colors` en este caso es opcional, pero puede ser útil para mostrar las diferentes semánticas de los puntos.

Por el momento, el complemento solo admite mallas triangulares, que se diferencian de las nubes de puntos anteriores solo por la presencia de caras (conjunto de vértices que representan el triángulo en la malla).

```python
mesh = tf.constant([[[0.19, 0.78, 0.02], ...]], shape=[1, 1064, 3])
colors = tf.constant([[[128, 104, 227], ...]], shape=[1, 1064, 3])
faces = tf.constant([[[13, 78, 54], ...]], shape=[1, 752, 3])

summary = mesh_summary.op('mesh', vertices=mesh, colors=colors, faces=faces)
```

Solo el tensor `colors` es opcional para los resúmenes de malla.

## Configuración de escenas

La forma en que se mostrarán los objetos también depende de la configuración de la escena, es decir, la intensidad y el color de las fuentes de luz, el material de los objetos, los modelos de cámara, etc. Todo esto se puede configurar mediante un parámetro adicional `config_dict`. Este diccionario puede contener tres claves de alto nivel: `camera`, `lights` y `material`. Cada clave también debe ser un diccionario con la clave obligatoria `cls`, que representa un nombre de clase válido de [THREE.js](https://threejs.org).

```python
camera_config = {'cls': 'PerspectiveCamera'}
summary = mesh_summary.op(
    "mesh",
    vertices=mesh,
    colors=colors,
    faces=faces,
    config_dict={"camera": camera_config},
)
```

`camera_config` del fragmento anterior se puede ampliar de acuerdo con la [documentación de THREE.js](https://threejs.org/docs/index.html#manual/en/introduction/Creating-a-scene). Todas las claves de `camera_config` se pasarán a una clase con el nombre `camera_config.cls`. Por ejemplo (según la [documentación](https://threejs.org/docs/index.html#api/en/cameras/PerspectiveCamera) de `PerspectiveCamera`):

```python
camera_config = {
  'cls': 'PerspectiveCamera',
  'fov': 75,
  'aspect': 0.9,
}
...
```

Tenga en cuenta que la configuración de la escena no es una variable entrenable (es decir, estática) y solo debe indicarse durante la creación de los resúmenes.

## Cómo instalar

Actualmente, el complemento es parte de la compilación nocturna de TensorBoard, por lo tanto, debe instalarlo antes de usarlo.

### Colab

```
!pip install -q -U tb-nightly
```

Luego, cargue la extensión Tensorboard y ejecútela, de manera similar a como lo haría en la Terminal:

```
%load_ext tensorboard
%tensorboard --logdir=/path/to/logs
```

Consulte el [bloc de notas de ejemplo de Colab](https://colab.research.google.com/github/tensorflow/tensorboard/blob/master/tensorboard/plugins/mesh/Mesh_Plugin_Tensorboard.ipynb) para obtener más detalles.

### Terminal

Si desea ejecutar la compilación nocturna de TensorBoard a nivel local, primero debe instalarla:

```shell
pip install tf-nightly
```

Luego, simplemente ejecútela:

```shell
tensorboard --logdir path/to/logs
```
