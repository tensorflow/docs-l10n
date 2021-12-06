# 在 GCP 上运行 TFF 模拟

本教程介绍如何在 GCP 上运行 TFF 模拟。

## 在单个运行时容器上运行模拟

### 1. [安装并初始化 Cloud SDK](https://cloud.google.com/sdk/docs/quickstarts)。

### 2. 克隆 TensorFlow Federated 仓库。

```shell
$ git clone https://github.com/tensorflow/federated.git
$ cd "federated"
```

### 3. 运行单个运行时容器。

1. 构建运行时容器。

    ```shell
    $ docker build \
        --network=host \
        --tag "<registry>/tff-runtime" \
        --file "tensorflow_federated/tools/runtime/container/latest.Dockerfile" \
        .
    ```

2. 发布运行时容器。

    ```shell
    $ docker push <registry>/tff-runtime
    ```

3. 创建 Compute Engine 实例

    1. 在 Cloud Console 中，转到 [VM Instances](https://console.cloud.google.com/compute/instances) 页面。

    2. 点击 **Create instance**。

    3. 在 **Firewall** 部分，选择 **Allow HTTP traffic** 和 **Allow HTTPS traffic**。

    4. 点击 **Create** 以创建实例。

4. 使用 `ssh` 连接到实例。

    ```shell
    $ gcloud compute ssh <instance>
    ```

5. 在后台运行运行时容器。

    ```shell
    $ docker run \
        --detach \
        --name=tff-runtime \
        --publish=8000:8000 \
        <registry>/tff-runtime
    ```

6. 退出实例。

    ```shell
    $ exit
    ```

7. 获取实例的内部  <strong>IP 地址</strong>。

    此信息稍后将用作我们测试脚本的参数。

    ```shell
    $ gcloud compute instances describe <instance> \
        --format='get(networkInterfaces[0].networkIP)'
    ```

### 4. 在客户端容器上运行模拟。

1. 构建客户端容器。

    ```shell
    $ docker build \
        --network=host \
        --tag "<registry>/tff-client" \
        --file "tensorflow_federated/tools/client/latest.Dockerfile" \
        .
    ```

2. 发布客户端容器。

    ```shell
    $ docker push <registry>/tff-client
    ```

3. 创建 Compute Engine 实例

    1. 在 Cloud Console 中，转到 [VM Instances](https://console.cloud.google.com/compute/instances) 页面。

    2. 点击 **Create instance**。

    3. 在 **Firewall** 部分，选择 **Allow HTTP traffic** 和 **Allow HTTPS traffic**。

    4. 点击 **Create** 以创建实例。

4. 将实验复制到 Compute Engine 实例中。

    ```shell
    $ gcloud compute scp \
        "tensorflow_federated/tools/client/test.py" \
        <instance>:~
    ```

5. 使用 `ssh` 连接到实例。

    ```shell
    $ gcloud compute ssh <instance>
    ```

6. 以交互方式运行客户端容器。

    随即会在终端打印出字符串“Hello World”。

    ```shell
    $ docker run \
        --interactive \
        --tty \
        --name=tff-client \
        --volume ~/:/simulation \
        --workdir /simulation \
        <registry>/tff-client \
        bash
    ```

7. 运行 Python 脚本。

    使用运行运行时容器的实例的内部 <strong>IP 地址</strong>。

    ```shell
    $ python3 test.py --host '<IP address>'
    ```

8. 退出容器。

    ```shell
    $ exit
    ```

9. 退出实例。

    ```shell
    $ exit
    ```
