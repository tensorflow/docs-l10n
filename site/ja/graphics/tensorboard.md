# メッシュプラグイン

## 概要

メッシュとポイントクラウドは、3D 形状を表現する、重要で強力な種類のデータであり、コンピュータビジョンとコンピュータグラフィックスの分野で広く研究されています。3D データのユビキタス化はさらに進化しており、研究者は 2D データから 3D 幾何の再構築、3D ポイントクラウドのセマンティックセグメント化、3D オブジェクトのアライメントと変形といった新しい問題に取り組んでいます。したがって、特にトレーニング段階における結果の視覚化は、モデルがどのような性能を発揮するかをより理解することが重要です。

![Mesh Plugin in TensorBoard](https://storage.googleapis.com/tensorflow-graphics/git/readme/tensorboard_plugin.jpg){width="100%"}

このプラグインは、TensorBoard に 3D ポイントクラウドまたはメッシュ（三角形のポイントクラウド）を表示するのが目的です。さらに、ユーザーはレンダリングされたオブジェクトを操作することができます。

## Summary API

メッシュまたはポイントクラウドは、一連のテンソルで表現することができます。たとえば、ポイントクラウドをポイントの一連の 3D 座標と各ポイントに関連付けられた色として見ることができます。

```python
from tensorboard.plugins.mesh import summary as mesh_summary
...

point_cloud = tf.constant([[[0.19, 0.78, 0.02], ...]], shape=[1, 1064, 3])
point_colors = tf.constant([[[128, 104, 227], ...]], shape=[1, 1064, 3])

summary = mesh_summary.op('point_cloud', vertices=point_cloud, colors=point_colors)
```

注意: `colors` テンソルはこの場合オプションですが、ポイントの異なるセマンティクスを表示する上で役立つ場合があります。

プラグインは、現在三角メッシュのみをサポートしています。三角メッシュは、メッシュの三角形を表す角のセットである面が存在するということが、上記のポイントクラウドと異なる点です。

```python
mesh = tf.constant([[[0.19, 0.78, 0.02], ...]], shape=[1, 1064, 3])
colors = tf.constant([[[128, 104, 227], ...]], shape=[1, 1064, 3])
faces = tf.constant([[[13, 78, 54], ...]], shape=[1, 752, 3])

summary = mesh_summary.op('mesh', vertices=mesh, colors=colors, faces=faces)
```

メッシュの要約では、`colors` テンソルのみがオプションです。

## シーン構成

オブジェクトの表示方法は、光源の明暗と色、オブジェクトの材質、カメラモデルなどのシーン構成によっても異なります。こう言ったすべての要素は、`config_dict` パラメータを追加して構成することができます。このディクショナリには、`camera`、`lights`、および `material` という 3 つの高齢ベルキーが含まれます。各キーは、有効な [THREE.js](https://threejs.org) クラス名を表す、必須キー `cls` を伴うディクショナリである必要もあります。

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

上記のスニペットにある `camera_config` は、[THREE.js ドキュメント](https://threejs.org/docs/index.html#manual/en/introduction/Creating-a-scene)に従って拡張することができます。`camera_config` のすべてのキーは `camera_config.cls` という名前でクラスに渡されます。次は、[`PerspectiveCamera`](https://threejs.org/docs/index.html#api/en/cameras/PerspectiveCamera) ドキュメントに基づく例です。

```python
camera_config = {
  'cls': 'PerspectiveCamera',
  'fov': 75,
  'aspect': 0.9,
}
...
```

シーン構成は、トレーニング可能な変数ではなく（静的）、要約の作成中にのみ指定されます。

## インストール方法

現在のところ、このプラグインは、TensorBoard ナイトリービルドの一部であるため、プラグインを使用する前に、ナイトリービルドをインストールする必要があります。

### Colab

```
!pip install -q -U tb-nightly
```

そして、ターミナルで実行する方法と同じように、Tensorboard 拡張機能を読み込んで実行します。

```
%load_ext tensorboard
%tensorboard --logdir=/path/to/logs
```

詳細は、[サンプル Colab ノートブック](https://colab.research.google.com/github/tensorflow/tensorboard/blob/master/tensorboard/plugins/mesh/Mesh_Plugin_Tensorboard.ipynb) をご覧ください。

### ターミナル

TensorBoard ナイトリービルドをローカルで実行する場合は、先にそれをインストールする必要があります。

```shell
pip install tf-nightly
```

そして、そのビルドを実行します。

```shell
tensorboard --logdir path/to/logs
```
