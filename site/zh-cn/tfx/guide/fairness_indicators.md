# Fairness Indicators

Fairness Indicators 旨在与更广泛的 TensorFlow 工具包进行合作，支持团队评估和改善模型的公平性问题。目前，我们的许多产品都在内部积极使用该工具，现在该工具以推出 BETA 版本，您可以在自己的用例中试用。

![Fairness Indicator 信息中心](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/guide/images/fairnessIndicators.png?raw=true)

## 什么是 Fairness Indicators？

Fairness Indicators 是用于轻松计算二元和多类分类器常用公平性指标的库。许多现有的评估公平性问题的工具在大型数据集和模型上效果不佳。对于 Google 来说，拥有可以在数十亿用户的系统上运行的工具很重要。Fairness Indicators 将使您能够评估任何大小的用例。

特别是，Fairness Indicators 包括以下功能：

- 评估数据集的分布
- 评估模型性能，切分为定义的用户组
    - 通过置信区间和多个阈值的评估，对结果充满信心
- 深入研究各个切片，探索根本原因和改进机会

本[案例研究](https://developers.google.com/machine-learning/practica/fairness-indicators)附有[视频](https://www.youtube.com/watch?v=pHT-ImFXPQo)和编程练习，演示了如何在您自己的产品中使用 Fairness Indicators 持续评估公平性问题。

[](http://www.youtube.com/watch?v=pHT-ImFXPQo)

Pip 软件包下载包括以下内容：

- **[TensorFlow Data Validation (TFDV)](https://www.tensorflow.org/tfx/data_validation/get_started)**
- **[TensorFlow Model Analysis (TFMA)](https://www.tensorflow.org/tfx/model_analysis/get_started)**
    - **Fairness Indicators**
- **[What-If 工具 (WIT)](https://www.tensorflow.org/tensorboard/what_if_tool)**

## 在 TensorFlow 模型中使用 Fairness Indicators

### 数据

要使用 TFMA 运行 Fairness Indicators，请确保已为要切分的特征标记了评估数据集。如果没有针对公平性问题的确切切片特征，可以尝试查找具有此特征的评估集，或者在特征集中考虑可能突出显示结果差异的代理特征。有关其他指导，请参阅[此处](https://tensorflow.org/responsible_ai/fairness_indicators/guide/guidance)。

### 模型

您可以使用 TensorFlow Estimator 类来构建模型。TFMA 即将支持 Keras 模型。如果要在 Keras 模型上运行 TFMA，请参阅下文的“与模型无关的 TFMA”部分。

训练 Estimator 后，您需要导出已保存的模型以进行评估。要了解详情，请参阅 [TFMA 指南](/tfx/model_analysis/get_started)。

### 配置切片

接下来，定义要评估的切片：

```python
slice_spec = [
  tfma.slicer.SingleSliceSpec(columns=[‘fur color’])
]
```

如果想要评估交叉切片（例如，毛皮颜色和高度），可以进行以下设置：

```python
slice_spec = [
  tfma.slicer.SingleSliceSpec(columns=[‘fur_color’, ‘height’])
]`
```

### 计算公平性指标

将 Fairness Indicators 回调添加到 `metrics_callback` 列表。您可以在回调中定义一个阈值列表，在其中评估模型。

```python
from tensorflow_model_analysis.addons.fairness.post_export_metrics import fairness_indicators

# Build the fairness metrics. Besides the thresholds, you also can config the example_weight_key, labels_key here. For more details, please check the api.
metrics_callbacks = \
    [tfma.post_export_metrics.fairness_indicators(thresholds=[0.1, 0.3,
     0.5, 0.7, 0.9])]

eval_shared_model = tfma.default_eval_shared_model(
    eval_saved_model_path=tfma_export_dir,
    add_metrics_callbacks=metrics_callbacks)
```

在运行配置之前，请确定是否要启用置信区间的计算。置信区间使用泊松自助法计算，需要基于 20 个样本重新计算。

```python
compute_confidence_intervals = True
```

运行 TFMA 评估流水线：

```python
validate_dataset = tf.data.TFRecordDataset(filenames=[validate_tf_file])

# Run the fairness evaluation.
with beam.Pipeline() as pipeline:
  _ = (
      pipeline
      | beam.Create([v.numpy() for v in validate_dataset])
      | 'ExtractEvaluateAndWriteResults' >>
       tfma.ExtractEvaluateAndWriteResults(
                 eval_shared_model=eval_shared_model,
                 slice_spec=slice_spec,
                 compute_confidence_intervals=compute_confidence_intervals,
                 output_path=tfma_eval_result_path)
  )
eval_result = tfma.load_eval_result(output_path=tfma_eval_result_path)
```

### 呈现 Fairness Indicators

```python
from tensorflow_model_analysis.addons.fairness.view import widget_view

widget_view.render_fairness_indicator(eval_result=eval_result)
```

![Fairness Indicators](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/guide/images/fairnessIndicators.png?raw=true)

使用 Fairness Indicators 的提示：

- 通过选中左侧的复选框来**选择要显示的指标**。每个指标的单个图表将按顺序显示在微件中。
- 使用下拉选择器**更改基准切片**（图表上的第一个条形图）。将使用此基准值计算增量。
- 使用下拉选择器**选择阈值**。您可以在同一个图表上查看多个阈值。所选阈值将以粗体显示，您可以单击粗体阈值取消选择。
- **将鼠标悬停在条形图上**查看该切片的指标。
- 使用“Diff w. baseline”列**确定与基准的差异**，该列标识当前切片与基准之间的百分比差异。
- 使用 [What-If Tool](https://pair-code.github.io/what-if-tool/) **深入探索切片的数据点**。请参阅[此处](https://github.com/tensorflow/fairness-indicators/)的示例。

#### 为多个模型呈现 Fairness Indicators

Fairness Indicators 也可用于比较模型。不传入单个 eval_result，而是传入 multi_eval_results 对象，该对象是将两个模型名称映射到 eval_result 对象的字典。

```python
from tensorflow_model_analysis.addons.fairness.view import widget_view

eval_result1 = tfma.load_eval_result(...)
eval_result2 = tfma.load_eval_result(...)
multi_eval_results = {"MyFirstModel": eval_result1, "MySecondModel": eval_result2}

widget_view.render_fairness_indicator(multi_eval_results=multi_eval_results)
```

![Fairness Indicators - 模型比较](https://gitlocalize.com/repo/4592/zh-cn/site/en-snapshot/tfx/guide/images/fairnessIndicatorsCompare.png)

模型比较可以与阈值比较一起使用。例如，您可以在两组阈值下比较两个模型，以找到公平性指标的最佳组合。

## 在非 TensorFlow 模型中使用 Fairness Indicators

为了更好地支持具有不同模型和工作流的客户端，我们开发了一个评估库，该库与所评估的模型无关。

任何想要评估其机器学习系统的人都可以使用这个库，尤其是当您具有基于非 TensorFlow 的模型时。您可以使用 Apache Beam Python SDK 创建独立的 TFMA 评估二进制文件，然后运行它来分析模型。

### 数据

此步骤是为了提供用于运行评估的数据集。该数据集应为 [tf.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord) proto 格式，并具有标签、预测和其他您可能想要切分的特征。

```python
tf.Example {
    features {
        feature {
          key: "fur_color" value { bytes_list { value: "gray" } }
        }
        feature {
          key: "height" value { bytes_list { value: "tall" } }
        }
        feature {
          key: "prediction" value { float_list { value: 0.9 } }
        }
        feature {
          key: "label" value { float_list { value: 1.0 } }
        }
    }
}
```

### 模型

无需指定模型，而是创建模型无关的评估配置和提取程序，以进行解析并提供 TFMA 计算指标所需的数据。[ModelAgnosticConfig](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/model_agnostic_eval/model_agnostic_predict.py) 规范定义了要从输入样本中使用的特征、预测和标签。

为此，使用表示所有特征的键（包括标签键和预测键）和表示特征的数据类型的值来创建特征映射。

```python
feature_map[label_key] = tf.FixedLenFeature([], tf.float32, default_value=[0])
```

使用标签键、预测键和特征映射创建与模型无关的配置。

```python
model_agnostic_config = model_agnostic_predict.ModelAgnosticConfig(
    label_keys=list(ground_truth_labels),
    prediction_keys=list(predition_labels),
    feature_spec=feature_map)
```

### 设置与模型无关的提取程序

[提取程序](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/model_agnostic_eval/model_agnostic_extractor.py)用于使用与模型无关的配置从输入中提取特征、标签和预测。如果要对数据进行切分，还需要定义[切片键规范](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/slicer)，其中包含有关要切分的列的信息。

```python
model_agnostic_extractors = [
    model_agnostic_extractor.ModelAgnosticExtractor(
        model_agnostic_config=model_agnostic_config, desired_batch_size=3),
    slice_key_extractor.SliceKeyExtractor([
        slicer.SingleSliceSpec(),
        slicer.SingleSliceSpec(columns=[‘height’]),
    ])
]
```

### 计算公平性指标

作为 [EvalSharedModel](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/types/EvalSharedModel) 的一部分，您可以提供希望对模型进行评估的所有指标。指标以指标回调的形式提供，例如在 [post_export_metrics](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/post_export_metrics/post_export_metrics.py) 或 [fairness_indicators](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/addons/fairness/post_export_metrics/fairness_indicators.py) 中定义的指标回调。

```python
metrics_callbacks.append(
    post_export_metrics.fairness_indicators(
        thresholds=[0.5, 0.9],
        target_prediction_keys=[prediction_key],
        labels_key=label_key))
```

它还接受 `construct_fn`，用于创建 TensorFlow 计算图来执行评估。

```python
eval_shared_model = types.EvalSharedModel(
    add_metrics_callbacks=metrics_callbacks,
    construct_fn=model_agnostic_evaluate_graph.make_construct_fn(
        add_metrics_callbacks=metrics_callbacks,
        fpl_feed_config=model_agnostic_extractor
        .ModelAgnosticGetFPLFeedConfig(model_agnostic_config)))
```

完成所有设置后，请使用 [model_eval_lib](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/api/model_eval_lib.py) 提供的 `ExtractEvaluate` 或 `ExtractEvaluateAndWriteResults` 函数来评估模型。

```python
_ = (
    examples |
    'ExtractEvaluateAndWriteResults' >>
        model_eval_lib.ExtractEvaluateAndWriteResults(
        eval_shared_model=eval_shared_model,
        output_path=output_path,
        extractors=model_agnostic_extractors))

eval_result = tensorflow_model_analysis.load_eval_result(output_path=tfma_eval_result_path)
```

最后，按照上面“呈现 Fairness Indicators”部分中的说明来呈现 Fairness Indicators。

## 更多示例

[Fairness Indicators 示例目录](https://github.com/tensorflow/fairness-indicators/tree/master/fairness_indicators/examples)包含了几个示例：

- [Fairness_Indicators_Example_Colab.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/fairness_indicators/documentation/examples/Fairness_Indicators_Example_Colab.ipynb) 概述了 [TensorFlow Model Analysis](https://www.tensorflow.org/tfx/guide/tfma) 中的 Fairness Indicators，以及如何将其用于实际的数据集。此笔记本还介绍了 [TensorFlow Data Validation](https://www.tensorflow.org/tfx/data_validation/get_started) 和 [What-If 工具](https://pair-code.github.io/what-if-tool/)，这两个工具用于分析使用 Fairness Indicators 打包的 TensorFlow 模型。
- [Fairness_Indicators_on_TF_Hub.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/fairness_indicators/documentation/examples/Fairness_Indicators_on_TF_Hub_Text_Embeddings.ipynb) 演示了如何使用 Fairness Indicators 比较在不同[文本嵌入向量](https://en.wikipedia.org/wiki/Word_embedding)上训练的模型。此笔记本使用来自 [TensorFlow Hub](https://www.tensorflow.org/hub)（TensorFlow 的库）的文本嵌入向量来发布、发现和重用模型组件。
- [Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/fairness_indicators/documentation/examples/Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb) 演示了如何在 TensorBoard 中直观呈现 Fairness Indicators。
