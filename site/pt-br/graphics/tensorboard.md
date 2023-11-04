# Plug-in de malha

## Visão geral

Malhas e nuvens de pontos são tipos de dados importantes e poderosos para representar formas tridimensionais, além de serem amplamente estudadas no campo da visão computacional e computação gráfica. Dados tridimensionais estão se tornando onipresentes, e os pesquisadores enfrentam novos problemas, como reconstrução geométrica tridimensional usando dados bidimensionais, segmentação de semântica de nuvem de pontos tridimensional, alinhamento ou transformação de objetos tridimensionais e assim por diante. Portanto, visualizar resultados, principalmente durante a fase de treinamento, é essencial para avaliar o desempenho do modelo.

![Mesh Plugin in TensorBoard](https://storage.googleapis.com/tensorflow-graphics/git/readme/tensorboard_plugin.jpg){width="100%"}

O objetivo deste plug-in é exibir malhas ou nuvens de pontos tridimensionais (nuvens de pontos trianguladas) no TensorBoard. Além disso, ele permite que o usuário interaja com os objetos renderizados.

## Summary API

Uma malha ou nuvem de pontos pode ser representada por um conjunto de tensores. Por exemplo, é possível ver uma nuvem de pontos como um conjunto de coordenadas tridimensionais dos pontos e algumas corres associadas a cada ponto.

```python
from tensorboard.plugins.mesh import summary as mesh_summary
...

point_cloud = tf.constant([[[0.19, 0.78, 0.02], ...]], shape=[1, 1064, 3])
point_colors = tf.constant([[[128, 104, 227], ...]], shape=[1, 1064, 3])

summary = mesh_summary.op('point_cloud', vertices=point_cloud, colors=point_colors)
```

Observação: o tensor `colors` (cores) é opcional neste caso, mas pode ser útil para mostrar semânticas diferentes dos pontos.

Atualmente, o plug-in tem suporte apenas a malhas triangulares, que são diferentes das nuvens de pontos acima somente pela presença de faces: conjuntos de vértices representando o triângulo da malha.

```python
mesh = tf.constant([[[0.19, 0.78, 0.02], ...]], shape=[1, 1064, 3])
colors = tf.constant([[[128, 104, 227], ...]], shape=[1, 1064, 3])
faces = tf.constant([[[13, 78, 54], ...]], shape=[1, 752, 3])

summary = mesh_summary.op('mesh', vertices=mesh, colors=colors, faces=faces)
```

Somente o tensor `colors` é opcional para resumos de malhas.

## Configuração da cena

A forma como os objetos serão exibidos também depende da configuração da cena, como a intensidade e cor das fontes de luz, o material dos objetos, os modelos de câmera e assim por diante. Tudo isso pode ser configurado usando-se um parâmetro adicional `config_dict`. Esse dicionário pode conter três chaves de alto nível: `camera` (câmera), `lights` (luzes) e `material`. Cada chave também precisa ser um dicionário com a chave obrigatória `cls`, que representa o nome válido da classe [THREE.js](https://threejs.org).

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

No trecho de código acima, `camera_config` pode ser expandido de acordo com a [documentação do THREE.js](https://threejs.org/docs/index.html#manual/en/introduction/Creating-a-scene). Todas as chaves de `camera_config` serão passadas para uma classe de nome `camera_config.cls`. Por exemplo (com base na [documentação de `PerspectiveCamera`](https://threejs.org/docs/index.html#api/en/cameras/PerspectiveCamera)):

```python
camera_config = {
  'cls': 'PerspectiveCamera',
  'fov': 75,
  'aspect': 0.9,
}
...
```

Observe que a configuração da cena não é uma variável treinável (é estática) e deve ser fornecida somente durante a criação de resumos.

## Como instalar

No momento, o plug-in faz parte do build noturno do TensorBoard e, portanto, é necessário instalá-lo antes de usar o plug-in.

### Colab

```
!pip install -q -U tb-nightly
```

Em seguida, carregue e execute a extensão do TensorBoard, similar a como você faria no terminal:

```
%load_ext tensorboard
%tensorboard --logdir=/path/to/logs
```

Confira mais detalhes no [notebook de exemplo do Colab](https://colab.research.google.com/github/tensorflow/tensorboard/blob/master/tensorboard/plugins/mesh/Mesh_Plugin_Tensorboard.ipynb).

### Terminal

Se você quiser executar o build noturno do TensorBoard localmente, primeiro precisa instalá-lo:

```shell
pip install tf-nightly
```

Em seguida, basta executá-lo:

```shell
tensorboard --logdir path/to/logs
```
