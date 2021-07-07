## 使用存根执行器测试流水线

### 简介

**您应该先将 [template.ipynb](https://github.com/tensorflow/tfx/blob/master/docs/tutorials/tfx/template.ipynb) 教程完成到*第 6 步*，以继续本教程。**

本文档将提供使用 `BaseStubExecuctor` 测试 TensorFlow Extended (TFX) 流水线的说明，它会使用黄金测试数据生成伪工件。这是为了让用户替换他们不想测试的执行器，以便节省运行实际执行器的时间。存根执行器随 TFX Python 软件包一起提供，位于 `tfx.experimental.pipeline_testing.base_stub_executor` 下。

本教程是 `template.ipynb` 教程的扩展，因此，您仍将使用芝加哥市发布的 [Taxi Trips 数据集](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)。我们强烈建议您在使用存根执行器之前尝试修改组件。

### 1. 在 Google Cloud Storage 中记录流水线输出

我们首先需要记录流水线输出，以便存根执行器可以从记录的输出中复制工件。

由于本教程假定您已将 `template.ipynb` 完成到了第 6 步，因此一定已经在 [MLMD](https://www.tensorflow.org/tfx/guide/mlmd) 中保存了一个成功的流水线运行。MLMD 中的执行信息可以使用 gRPC 服务器访问。

打开终端并运行以下命令：

1. 生成具有相应凭据的 kubeconfig 文件：`bash gcloud container clusters get-credentials $cluster_name --zone $compute_zone --project $gcp_project_id`。其中，`$compute_zone` 是 GCP 引擎的区域，`$gcp_project_id` 是您的 GCP 项目的项目 ID。

2. 设置端口转发以连接到 MLMD：`bash nohup kubectl port-forward deployment/metadata-grpc-deployment -n $namespace $port:8080 &`。其中，`$namespace` 是集群命名空间，`$port` 是将用于端口转发的任何未使用的端口。

3. 克隆 tfx GitHub 仓库。在 tfx 目录中，运行以下命令：

```bash
python tfx/experimental/pipeline_testing/pipeline_recorder.py \
--output_dir=gs://<gcp_project_id>-kubeflowpipelines-default/testdata \
--host=$host \
--port=$port \
--pipeline_name=$pipeline_name
```

应将 `$output_dir` 设置为 Google Cloud Storage 中要记录流水线输出的路径，因此请务必将 `<gcp_project_id>` 替换为 GCP 项目 ID。

`$host` 和 `$port` 是用于连接到 MLMD 的元数据 GRPC 服务器的主机名和端口。应将 `$port` 设置为您用于端口转发的端口号，而主机名可以设置为“localhost”。

在 `template.ipynb` 教程中，流水线名称默认设置为“my_pipeline”，因此设置 `pipeline_name="my_pipeline"`。如果您在运行模板教程时修改了流水线名称，则应相应地修改 `--pipeline_name`。

### 2. 在 Kubeflow DAG 运行程序中启用存根执行器

首先，请确保已使用 `tfx template copy` CLI 命令将预定义模板复制到了您的项目目录中。需要对复制的源文件中的以下两个文件进行编辑。

1. 在 kubeflow_dag_runner.py 所在目录下创建名为 `stub_component_launcher.py` 的文件，并将以下内容放入其中。

    ```python
    from tfx.experimental.pipeline_testing import base_stub_component_launcher
    from pipeline import configs

    class StubComponentLauncher(
        base_stub_component_launcher.BaseStubComponentLauncher):
      pass

    # GCS directory where KFP outputs are recorded
    test_data_dir = "gs://{}/testdata".format(configs.GCS_BUCKET_NAME)
    # TODO: customize self.test_component_ids to test components, replacing other
    # component executors with a BaseStubExecutor.
    test_component_ids = ['Trainer']
    StubComponentLauncher.initialize(
        test_data_dir=test_data_dir,
        test_component_ids=test_component_ids)
    ```

    注：此存根组件启动器不能在 `kubeflow_dag_runner.py` 中进行定义，因为启动器类是通过模块路径导入的。

2. 将组件 ID 设置为要测试的组件 ID 的列表（换句话说，其他组件的执行器将被替换为 BaseStubExecutor）。

3. 打开 `kubeflow_dag_runner.py`。在顶部添加以下导入语句，以使用我们刚才添加的 `StubComponentLauncher` 类。

    ```python
    import stub_component_launcher
    ```

4. 在 `kubeflow_dag_runner.py` 中，将 `StubComponentLauncher` 类添加到 `KubeflowDagRunnerConfig` 的 `supported_launcher_class`，以使存根执行器可以启动：

    ```python
    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        supported_launcher_classes=[
            stub_component_launcher.StubComponentLauncher
        ],
    ```

### 3. 使用存根执行器更新并运行流水线

使用带有存根执行器的修改后的流水线定义更新现有流水线。

```bash
tfx pipeline update --pipeline-path=kubeflow_dag_runner.py \
  --endpoint=$endpoint --engine=kubeflow
```

应将 `$endpoint` 设置为您的 KFP 集群端点。

运行以下命令，为更新后的流水线创建新的执行运行。

```bash
tfx run create --pipeline-name $pipeline_name --endpoint=$endpoint \
  --engine=kubeflow
```

## 清理

使用命令 `fg` 在后台访问端口转发，然后按 Ctrl+C 终止。您可以使用 `gsutil -m rm -R $output_dir` 删除包含记录了流水线输出的目录。

要清理此项目中使用的所有 Google Cloud 资源，您可以[删除用于本教程的 Google Cloud 项目](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects)。

或者，您可以通过访问每个控制台来清理各个资源：- [Google Cloud Storage](https://console.cloud.google.com/storage) - [Google Container Registry](https://console.cloud.google.com/gcr) - [Google Kubernetes Engine](https://console.cloud.google.com/kubernetes)
