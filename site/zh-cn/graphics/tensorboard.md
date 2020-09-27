# 网格插件

## 概述

网格和点云是表示 3D 形状的重要而强大的数据类型，在计算机视觉和计算机图形学领域得到了广泛的研究。3D 数据变得越来越普遍，研究人员向新问题提出了挑战，例如使用 2D 数据重建 3D 几何形状、3D 点云语义分割、3D 物体的对齐或变形等。因此，可视化结果（尤其是在训练阶段）对于更好地了解模型表现至关重要。

![Mesh Plugin in TensorBoard](https://storage.googleapis.com/tensorflow-graphics/git/readme/tensorboard_plugin.jpg){width="100%"}

此插件的作用是在 TensorBoard 中显示 3D 点云或网格（三角点云）。另外，它还允许用户与渲染的物体进行交互。

## Summary API

网格或点云都可由一组张量表示。例如，人们可将点云视为点的一组 3D 坐标以及与每个点关联的一些颜色。

```python
from tensorboard.plugins.mesh import summary as mesh_summary
...

point_cloud = tf.constant([[[0.19, 0.78, 0.02], ...]], shape=[1, 1064, 3])
point_colors = tf.constant([[[128, 104, 227], ...]], shape=[1, 1064, 3])

summary = mesh_summary.op('point_cloud', vertices=point_cloud, colors=point_colors)
```

请注意，`colors` 张量在这种情况下是可选的，但对于显示点的不同语义可能很有用。

此插件目前仅支持三角网格，三角网格仅因面的存在而与上面的点云不同，面是表示网格上三角形的一组顶点。

```python
mesh = tf.constant([[[0.19, 0.78, 0.02], ...]], shape=[1, 1064, 3])
colors = tf.constant([[[128, 104, 227], ...]], shape=[1, 1064, 3])
faces = tf.constant([[[13, 78, 54], ...]], shape=[1, 752, 3])

summary = mesh_summary.op('mesh', vertices=mesh, colors=colors, faces=faces)
```

对于网格摘要，仅 `colors` 张量是可选的。

## 场景配置

如何显示物体的方式还取决于场景配置，即光源的强度和颜色、物体的材质、相机模型等。所有这些均可通过附加参数 `config_dict` 进行配置。此字典可能包含三个高级键：`camera`、`lights` 和 `material`。每个键也必须是一个字典，其中包含必需的键 `cls`，表示有效的 [THREE.js](https://threejs.org) 类名称。

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

可根据 [THREE.js 文档](https://threejs.org/docs/index.html#manual/en/introduction/Creating-a-scene)扩展上述代码段中的 `camera_config`。`camera_config` 中的所有键都将传递给名称为 `camera_config.cls` 的类。例如（基于 [`PerspectiveCamera` 文档](https://threejs.org/docs/index.html#api/en/cameras/PerspectiveCamera)）：

```python
camera_config = {
  'cls': 'PerspectiveCamera',
  'fov': 75,
  'aspect': 0.9,
}
...
```

请注意，场景配置不是可训练的变量（即静态），应仅在创建摘要时提供。

## 如何安装

目前，此插件是 TensorBoard Nightly 版本的一部分，因此您必须先安装该插件，然后才能使用。

### Colab

```
!pip install -q -U tb-nightly
```

随后加载 Tensorboard 扩展程序并运行它，类似于在终端中执行的操作：

```
%load_ext tensorboard
%tensorboard --logdir=/path/to/logs
```

有关更多详细信息，请参阅[示例 Colab 笔记本](https://colab.research.google.com/github/tensorflow/tensorboard/blob/master/tensorboard/plugins/mesh/Mesh_Plugin_Tensorboard.ipynb)。

### 终端

如果您想在本地运行 TensorBoard Nightly 版本，首先需要安装它：

```shell
pip install tf-nightly
```

随后即可运行：

```shell
tensorboard --logdir path/to/logs
```
