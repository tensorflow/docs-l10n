# 在使用 TensorFlow Hub 的情况下从 TF1 迁移到 TF2

本页介绍了在将 TensorFlow 代码从 TensorFlow 1 迁移到 TensorFlow 2 时如何继续使用 TensorFlow Hub，旨在补充 TensorFlow 的常规[迁移指南](https://www.tensorflow.org/guide/migrate)。

对于 TF2，TF Hub 已经从旧版 `hub.Module` API 转为用于构建 `tf.compat.v1.Graph`，与 `tf.contrib.v1.layers` 类似。现在提供 `hub.KerasLayer` 与其他 Keras 层用于构建 `tf.keras.Model`（通常在 TF2 的新 [Eager Eexecution 环境](https://www.tensorflow.org/guide/eager_)中），其底层 `hub.load()` 方法用于低级 TensorFlow 代码。

`tensorflow_hub` 库中仍包含 `hub.Module` API，可在 TF1 以及 TF2 的 TF1 兼容模式下使用。该 API 只能加载 [TF1 Hub 格式](tf1_hub_module.md)的模型。

`hub.load()` 和 `hub.KerasLayer` 的新 API 适用于 TensorFlow 1.15（在 Eager 和计算图模式下）以及 TensorFlow 2。这一新版 API 可以加载新的 [TF2 SavedModel](tf2_saved_model.md) 资源，在[模型兼容性指南](model_compatibility.md)所述限制下，也可以加载 TF1 Hub 格式的模型。

通常情况下，建议尽可能使用新版 API。

## 新版 API 摘要

`hub.load()` 是新的低级函数，用于从 TensorFlow Hub （或兼容的服务）加载 SavedModel。它可以包装 TF2 的 `tf.saved_model.load()`；TensorFlow 的 [SavedModel 指南](https://www.tensorflow.org/guide/saved_model)介绍了您可以对结果执行的操作。

```python
m = hub.load(handle)
outputs = m(inputs)
```

`hub.KerasLayer` 类可调用 `hub.load()` 并调整结果以与其他 Keras 层共同用于 Keras 中。（它甚至可以方便地包装以其他方式使用的加载 SavedModel。）

```python
model = tf.keras.Sequential([
    hub.KerasLayer(handle),
    ...])
```

许多教程都展示了这些 API 的实际运行。具体请参阅：

- [文本分类示例笔记本](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb)
- [图像分类示例笔记本](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb)

### 在 Estimator 训练中使用新版 API

如果您在 Estimator 中通过参数服务器训练 TF2 SavedModel（或者在将变量置于远程设备上的 TF1 会话中），则需要在 tf.Session 的 ConfigProto 中设置 `experimental.share_cluster_devices_in_session`，否则您将收到错误消息，例如“Assigned device '/job:ps/replica:0/task:0/device:CPU:0' does not match any device.”

可按以下方式设置所需选项：

```python
session_config = tf.compat.v1.ConfigProto()
session_config.experimental.share_cluster_devices_in_session = True
run_config = tf.estimator.RunConfig(..., session_config=session_config)
estimator = tf.estimator.Estimator(..., config=run_config)
```

自 TF2.2 起，此选项已不再为实验性选项，可以删除 `.experimental` 部分。

## 加载 TF1 Hub 格式的旧版模型

有可能会出现新的 TF2 SavedModel 尚不支持您的用例的情况，此时您需要加载 TF1 Hub 格式的旧版模型。自 `tensorflow_hub` 0.7 版起，您可以将 TF1 Hub 格式的旧版模型与 `hub.KerasLayer` API 配合使用，如下所示：

```python
m = hub.KerasLayer(handle)
tensor_out = m(tensor_in)
```

此外，`KerasLayer` 还提供了指定 `tags`、`signature`、`output_key` 和 `signature_outputs_as_dict` 的功能，从而可以使用 TF1 Hub 格式的旧版模型和旧版 SavedModel 实现更具体的用途。

有关 TF1 Hub 格式兼容性的更多信息，请参阅[模型兼容性指南](model_compatibility.md)。

## 使用低级 API

旧版 TF1 Hub 格式模型可以通过 `tf.saved_model.load` 加载。不建议使用：

```python
# DEPRECATED: TensorFlow 1
m = hub.Module(handle, tags={"foo", "bar"})
tensors_out_dict = m(dict(x1=..., x2=...), signature="sig", as_dict=True)
```

而是建议使用：

```python
# TensorFlow 2
m = hub.load(path, tags={"foo", "bar"})
tensors_out_dict = m.signatures["sig"](x1=..., x2=...)
```

在以上示例中，`m.signatures` 是一个由签名名称键控的 TensorFlow [具体函数](https://www.tensorflow.org/tutorials/customization/performance#tracing)字典。调用此类函数会计算其所有输出，即使某些并未使用。（这与 TF1 的计算图模式的惰性评估不同。）
