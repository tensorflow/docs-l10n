# 메시 플러그인

## 개요

메시와 포인트 클라우드는 3D 형태를 나타내는 중요하고 강력한 데이터 유형이며 컴퓨터 비전 및 컴퓨터 그래픽 분야에서 널리 연구되고 있습니다. 3D 데이터는 더욱 보편화되고 있으며 연구원들은 2D 데이터로부터 3D 기하학 재구성, 3D 포인트 클라우드 의미론적 분할, 3D 객체 정렬 또는 모핑 등과 같은 새로운 문제에 도전하고 있습니다. 따라서 특히 훈련 단계에서 결과를 시각화하는 것은 모델의 성능을 더 잘 이해하는 데 중요합니다.

![TensorBoard의 메시 플러그인](https://storage.googleapis.com/tensorflow-graphics/git/readme/tensorboard_plugin.jpg) {width = "100 %"}

이 플러그인은 TensorBoard에서 3D 포인트 클라우드 또는 메시(삼각 측량 포인트 클라우드)를 표시하려고 합니다. 또한, 사용자가 렌더링된 개체와 상호 작용할 수 있습니다.

## 요약 API

메시 또는 포인트 클라우드는 텐서 세트로 표현할 수 있습니다. 예를 들어, 포인트 클라우드를 포인트의 3D 좌표 세트와 각 포인트와 관련된 일부 색상으로 볼 수 있습니다.

```python
from tensorboard.plugins.mesh import summary as mesh_summary
...

point_cloud = tf.constant([[[0.19, 0.78, 0.02], ...]], shape=[1, 1064, 3])
point_colors = tf.constant([[[128, 104, 227], ...]], shape=[1, 1064, 3])

summary = mesh_summary.op('point_cloud', vertices=point_cloud, colors=point_colors)
```

참고: `colors` 텐서는 이 경우 선택 사항이지만, 포인트의 다른 의미를 표시하는 데 유용할 수 있습니다.

플러그인은 현재 면(메시에서 삼각형을 나타내는 정점 세트)의 존재만 위의 포인트 클라우드와 다른 삼각형 메시만 지원합니다.

```python
mesh = tf.constant([[[0.19, 0.78, 0.02], ...]], shape=[1, 1064, 3])
colors = tf.constant([[[128, 104, 227], ...]], shape=[1, 1064, 3])
faces = tf.constant([[[13, 78, 54], ...]], shape=[1, 752, 3])

summary = mesh_summary.op('mesh', vertices=mesh, colors=colors, faces=faces)
```

`colors` 텐서만 메시 요약에서 선택 사항입니다.

## 장면 구성

객체가 표시되는 방법은 장면 구성, 즉 광원의 강도와 색상, 객체의 재질, 카메라 모델 등에 따라 달라집니다. 이 모든 것은 추가 매개변수 `config_dict`를 통해 구성할 수 있습니다. 이 사전에는 `camera`, `lights` 및 `material`의 3가지 고급 키가 포함될 수 있습니다. 각 키는 유효한 <a>THREE.js</a> 클래스 이름을 나타내는 필수 키 <code>cls</code>를 가진 사전이어야 합니다.

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

위의 `camera_config`는 <a>THREE.js 설명서</a>에 따라 확장할 수 있습니다. `camera_config`의 모든 키는 이름이 `camera_config.cls`인 클래스로 전달됩니다. 예를 들어, 다음과 같습니다([`PerspectiveCamera` 설명서](https://threejs.org/docs/index.html#api/en/cameras/PerspectiveCamera) 기반).

```python
camera_config = {
  'cls': 'PerspectiveCamera',
  'fov': 75,
  'aspect': 0.9,
}
...
```

장면 구성은 훈련 가능한 변수(예: 정적)가 아니며 요약 생성 중에만 제공되어야 합니다.

## 설치하는 방법

현재 플러그인은 TensorBoard nightly 빌드의 일부이므로 플러그인을 사용하기 전에 설치해야 합니다.

### Colab

```
!pip install -q -U tb-nightly
```

그런 다음 Tensorboard 확장을 로드하고 실행합니다. 터미널에서 수행하는 것과 유사합니다.

```
%load_ext tensorboard
%tensorboard --logdir=/path/to/logs
```

자세한 내용은 [예제 Colab 노트북](https://colab.research.google.com/github/tensorflow/tensorboard/blob/master/tensorboard/plugins/mesh/Mesh_Plugin_Tensorboard.ipynb)을 참조하세요.

### 터미널

TensorBoard nightly build를 로컬에서 실행하려면 먼저 설치해야 합니다.

```shell
pip install tf-nightly
```

그런 다음 실행합니다.

```shell
tensorboard --logdir path/to/logs
```
