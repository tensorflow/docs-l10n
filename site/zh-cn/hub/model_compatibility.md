<!--* freshness: { owner: 'maringeo' reviewed: '2021-04-12' review_interval: '6 months' } *-->

# TF1/TF2 的模型兼容性

## TF Hub 模型格式

TF Hub 提供了可重用的模型部分，这些部分可以在 TensorFlow 程序中重新加载、以之为基础构建模型以及重新训练。其中包括两种格式：

- 自定义 [TF1 Hub 格式](https://www.tensorflow.org/hub/tf1_hub_module)。该格式主要通过其 [hub.Module API](https://www.tensorflow.org/hub/api_docs/python/hub/Module) 用于 TF1（或 TF2 中的 TF1 兼容性模式）。[下文](#compatibility_of_hubmodule)介绍了完整的兼容性详细信息。
- 原生 [TF2 SavedModel](https://www.tensorflow.org/hub/tf2_saved_model) 格式。该格式主要通过 [hub.load](https://www.tensorflow.org/hub/api_docs/python/hub/load) 和 [hub.KerasLayer](https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer) API 用于 TF2。[下文](#compatibility_of_tf2_savedmodel)介绍了完整的兼容性详细信息。

[tfhub.dev](https://tfhub.dev) 上的模型页面中提供了模型格式信息。基于模型格式，TF1/2 中可能不支持模型**加载/推断**、**微调**或**创建**。

## TF1 Hub 格式兼容性 {:#compatibility_of_hubmodule}

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 20%">
    <col style="width: 40%">
    <col style="width: 40%">
    <td style="text-align: center; background-color: #D0D0D0">操作</td>
    <td style="text-align: center; background-color: #D0D0D0">TF2 中的 TF1/TF1 兼容性模式 <a href="#compatfootnote">[1]</a>
</td>
    <td style="text-align: center; background-color: #D0D0D0">TF2</td>
  </tr>
  <tr>
    <td>加载/推断</td>
    <td>完全支持（<a href="https://www.tensorflow.org/hub/tf1_hub_module#using_a_module">完整的 TF1 Hub 格式加载指南</a>）      <pre style="font-size: 12px;" lang="python">m = hub.Module(handle) outputs = m(inputs)</pre> </td>
    <td>建议使用 hub.load     <pre style="font-size: 12px;" lang="python">m = hub.load(handle)
outputs = m.signatures["sig"](inputs)</pre>       或 hub.KerasLayer       <pre style="font-size: 12px;" lang="python">m = hub.KerasLayer(handle, signature="sig")
outputs = m(inputs)</pre>
</td>
  </tr>
  <tr>
    <td>微调</td>
    <td>完全支持（<a href="https://www.tensorflow.org/hub/tf1_hub_module#for_consumers">完整的 TF1 Hub 格式微调指南</a>）<pre style="font-size: 12px;" lang="python">m = hub.Module(handle, trainable=True, tags=["train"]*is_training) outputs = m(inputs)</pre> <div style="font-style: italic; font-size: 14px"> 注：不需要单独的训练计算图的模块没有训练标签。</div>
</td>
    <td style="text-align: center">       不支持</td>
  </tr>
  <tr>
    <td>创建</td>
    <td>完全支持（请参阅<a href="https://www.tensorflow.org/hub/tf1_hub_module#general_approach">完整的 TF1 Hub 格式创建指南</a>）<br> <div style="font-style: italic; font-size: 14px"> 注：TF1 Hub 格式适用于 TF1，而在 TF2 中仅部分受支持。请考虑创建一个 TF2 SavedModel。      </div> </td>
    <td style="text-align: center">不支持</td>
  </tr>
</table>

## TF2 SavedModel 兼容性 {:#compatibility_of_tf2_savedmodel}

TF1.15 之前版本不支持。

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 20%">
    <col style="width: 40%">
    <col style="width: 40%">
    <td style="text-align: center; background-color: #D0D0D0">操作</td>
    <td style="text-align: center; background-color: #D0D0D0">TF2 中的 TF1.15/ TF1 兼容性模式 <a href="#compatfootnote">[1]</a>
</td>
    <td style="text-align: center; background-color: #D0D0D0">TF2</td>
  </tr>
  <tr>
    <td>加载/推断</td>
    <td>       使用 hub.load     <pre style="font-size: 12px;" lang="python">m = hub.load(handle) outputs = m(inputs)</pre>       或 hub.KerasLayer       <pre style="font-size: 12px;" lang="python">m = hub.KerasLayer(handle) outputs = m(inputs)</pre> </td>
    <td> 完全支持（<a href="https://www.tensorflow.org/hub/tf2_saved_model#using_savedmodels_from_tf_hub">完整的 TF2 SavedModel 加载指南）。使用 hub.load     <pre style="font-size: 12px;" lang="python">m = hub.load(handle)
outputs = m(inputs)</pre>       或 hub.KerasLayer       <pre style="font-size: 12px;" lang="python">m = hub.KerasLayer(handle)
outputs = m(inputs)</pre></a>
</td>
  </tr>
  <tr>
    <td>微调</td>
    <td>当通过 Model.fit() 训练或在 Estimator（model_fn 根据<a href="https://www.tensorflow.org/guide/migrate#using_a_custom_model_fn">自定义 model_fn 指南</a>封装模型）中进行训练时，在 tf.keras.Model 中使用的 hub.KerasLayer 支持。       <br><div style="font-style: italic; font-size: 14px;"> 注：hub.KerasLayer <span style="font-weight: bold;">不会</span>像旧 tf.compat.v1.layers 或 hub.Module API 一样填充计算图集合。</div>
</td>
    <td>       完全支持（<a href="https://www.tensorflow.org/hub/tf2_saved_model#for_savedmodel_consumers">完整的 TF2 SavedModel 微调指南</a>）。      使用 hub.load：      <pre style="font-size: 12px;" lang="python">m = hub.load(handle)
outputs = m(inputs, training=is_training)</pre>       或 hub.KerasLayer：      <pre style="font-size: 12px;" lang="python">m =  hub.KerasLayer(handle, trainable=True)
outputs = m(inputs)</pre>
</td>
  </tr>
  <tr>
    <td>创建</td>
    <td>可以在兼容性模式下调用 TF2 API <a href="https://www.tensorflow.org/api_docs/python/tf/saved_model/save">tf.saved_model.save()</a>。</td>
   <td>完全支持（请参阅 <a href="https://www.tensorflow.org/hub/tf2_saved_model#creating_savedmodels_for_tf_hub">complete TF2 SavedModel 创建指南</a>）</td>
  </tr>
</table>

<p id="compatfootnote">[1]“TF2 中的 TF1 兼容性模式”指的是使用 <code style="font-size: 12px;" lang="python">import tensorflow.compat.v1 as tf</code> 导入 TF2  并运行   <code style="font-size: 12px;" lang="python">tf.disable_v2_behavior()</code> 的组合效果，如 <a href="https://www.tensorflow.org/guide/migrate">TensorFlow 迁移指南</a>所述。</p>
