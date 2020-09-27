# Tuner TFX 流水线组件

Tuner 组件用于调节模型的超参数。

## Tuner 组件和 KerasTuner 库

Tuner 组件广泛使用 Python [KerasTuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) API 来调节超参数。

注：无论建模 API 如何，KerasTuner 库都可用于超参数调节，而不仅限于 Keras 模型。

## 组件

Tuner 需要：

- 用于训练和评估的 tf.Examples。
- 由用户提供、用于定义调节逻辑的模块文件（或模块 fn），包括模型定义、超参数搜索空间、目标等。
- 训练参数和评估参数的 [Protobuf](https://developers.google.com/protocol-buffers) 定义。
- （可选）调节参数的 [Protobuf](https://developers.google.com/protocol-buffers) 定义。
- （可选）由上游 Transform 组件生成的转换计算图。
- （可选）由 SchemaGen 流水线组件创建并且可由开发者选择性更改的数据架构。

利用给定的数据、模型和目标，Tuner 可以调节超参数并发出最佳结果。

## 说明

Tuner 需要具有以下签名的用户模块函数 `tuner_fn`：

```python
...
from kerastuner.engine import base_tuner

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
  """Build the tuner using the KerasTuner API.
  Args:
    fn_args: Holds args as name/value pairs.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.
  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """
  ...
```

在此函数中，您可以定义模型和超参数搜索空间，并选择用于调节的目标和算法。Tuner 组件会将此模块代码作为输入，调节超参数，然后发出最佳结果。

Trainer 可以将 Tuner 的输出超参数作为输入，并在其用户模块代码中使用它们。流水线定义如下所示：

```python
...
tuner = Tuner(
    module_file=module_file,  # Contains `tuner_fn`.
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    train_args=trainer_pb2.TrainArgs(num_steps=20),
    eval_args=trainer_pb2.EvalArgs(num_steps=5))

trainer = Trainer(
    module_file=module_file,  # Contains `run_fn`.
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    # This will be passed to `run_fn`.
    hyperparameters=tuner.outputs['best_hyperparameters'],
    train_args=trainer_pb2.TrainArgs(num_steps=100),
    eval_args=trainer_pb2.EvalArgs(num_steps=5))
...
```

您可能不想在每次重新训练模型时都调节超参数。一旦使用 Tuner 确定一组合适的超参数，就可以从流水线中移除 Tuner，并使用 `ImporterNode` 从先前的训练运行中导入 Tuner 工件，以馈入 Trainer。

```python
hparams_importer = ImporterNode(
    instance_name='import_hparams',
    # This can be Tuner's output file or manually edited file. The file contains
    # text format of hyperparameters (kerastuner.HyperParameters.get_config())
    source_uri='path/to/best_hyperparameters.txt',
    artifact_type=HyperParameters)

trainer = Trainer(
    ...
    # An alternative is directly use the tuned hyperparameters in Trainer's user
    # module code and set hyperparameters to None here.
    hyperparameters = hparams_importer.outputs['result'])
```

## 在 Google Cloud Platform (GCP) 上调节

在 Google Cloud Platform (GCP) 上运行时，Tuner 组件可以利用以下两项服务：

- [AI Platform Optimizer](https://cloud.google.com/ai-platform/optimizer/docs/overview)（通过 CloudTuner 实现）
- [AI Platform Training](https://cloud.google.com/ai-platform/training/docs)（作为分布式调节的群管理器）

### AI Platform Optimizer 作为超参数调节的后端

[AI Platform Optimizer](https://cloud.google.com/ai-platform/optimizer/docs/overview) 是一项托管服务，可基于 [Google Vizier](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/bcb15507f4b52991a0783013df4222240e942381.pdf) 技术执行黑盒优化。

[CloudTuner](https://github.com/GoogleCloudPlatform/tensorflow-gcp-tools/blob/master/python/tensorflow_enterprise_addons/cloudtuner/cloud_tuner.py) 是 [KerasTuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) 的一个实现，可与作为研究后端的 AI Platform Optimizer 服务对话。由于 CloudTuner 是 `kerastuner.Tuner` 的子类，因此它可以用作 `tuner_fn` 模块中的直接替代，并作为 TFX Tuner 组件的一部分执行。

下面是一个如何使用 `CloudTuner` 的代码段。请注意，对 `CloudTuner` 进行配置需要特定于 GCP 的条目，例如 `project_id` 和 `region`。

```python
...
from tensorflow_enterprise_addons import cloudtuner

...
def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
  """An implementation of tuner_fn that instantiates CloudTuner."""

  ...
  tuner = cloudtuner.CloudTuner(
      _build_model,
      hyperparameters=...,
      ...
      project_id=...,       # GCP Project ID
      region=...,           # GCP Region where Optimizer service is run.
      study_id=...,         # Unique ID of the tuning study
  )

  ...
  return TuneFnResult(
      tuner=tuner,
      fit_kwargs={...}
  )
```

### AI Platform Training 分布式工作进程群上的并行调节

作为 Tuner 组件的底层实现，KerasTuner 框架具有并行执行超参数搜索的能力。尽管固有 Tuner 组件不能并行执行多个搜索工作进程，但是通过使用 [Google Cloud AI Platform 扩展 Tuner 组件](https://github.com/tensorflow/tfx/blob/master/tfx/extensions/google_cloud_ai_platform/tuner/component.py)，它可以利用 AI Platform Training 作业作为分布式工作进程群管理器来运行并行调节。[TuneArgs](https://github.com/tensorflow/tfx/blob/master/tfx/proto/tuner.proto) 是为此组件指定的配置。这是固有 Tuner 组件的直接替代。

```python
from tfx.extensions.google_cloud_ai_platform.tuner.component import Tuner
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor

...
tuner = Tuner(
    ...   # Same kwargs as the above stock Tuner component.
    tune_args=tuner_pb2.TuneArgs(num_parallel_trials=3),  # 3-worker parallel
    custom_config={
        # Configures Cloud AI Platform-specific configs . For for details, see
        # https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#traininginput.
        ai_platform_trainer_executor.TRAINING_ARGS_KEY:
            {
                'project': ...,
                'region': ...,
                # Configuration of machines for each master/worker in the flock.
                'masterConfig': ...,
                'workerConfig': ...,
                ...
            }
    })
...
```

扩展 Tuner 组件的行为和输出与固有 Tuner 组件相同，只是多个超参数搜索会在不同的工作进程机器上并行执行，因此，`num_trials` 将更快地完成。当搜索算法极易并行化（例如 `RandomSearch`）时，这特别有效。但是，如果搜索算法使用来自前期试验结果（例如 AI Platform Optimizer 中实现的 Google Vizier 算法）的信息，则过度并行搜索会对搜索效率造成负面影响。

注：每个并行搜索中的每次试验都是在工作进程群中的单一机器上进行的，即模型训练没有利用多工作进程分布式训练的优势。我们计划在 CloudTuner 中添加相应的支持，即在每次试验中并行执行 Tuner 的情况下进行多工作进程分布式训练。

注：`CloudTuner` 和 Google Cloud AI Platform 扩展 Tuner 组件可以一起使用，在这种情况下，它允许由 AI Platform Optimizer 的超参数搜索算法提供支持的分布式并行调节。不过，为了做到这一点，Cloud AI Platform 作业必须获得访问 AI Platform Optimizer 服务的权限。

## 链接

[E2E 示例](https://github.com/tensorflow/tfx/blob/master/tfx/examples/iris/iris_pipeline_native_keras.py)

[KerasTuner 教程](https://www.tensorflow.org/tutorials/keras/keras_tuner)

[CloudTuner 教程](https://github.com/GoogleCloudPlatform/ai-platform-samples/blob/master/notebooks/samples/optimizer/ai_platform_optimizer_tuner.ipynb)

[提案](https://github.com/tensorflow/community/blob/master/rfcs/20200420-tfx-tuner-component.md)
