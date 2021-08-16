# GCP での TFF シミュレーション

このチュートリアルでは、GCP で TFF シミュレーションを実行する方法を説明します。

## 単一のランタイムコンテナでシミュレーションを実行する

### 1. <a>Cloud SDK をインストールして初期化します</a>。

### 2. TensorFlow Federated リポジトリを複製します。

```shell
$ git clone https://github.com/tensorflow/federated.git
$ cd "federated"
```

### 3. 単一のランタイムコンテナを実行します。

1. ランタイムコンテナを構築します。

    ```shell
    $ docker build \
        --network=host \
        --tag "<registry>/tff-runtime" \
        --file "tensorflow_federated/tools/runtime/container/latest.Dockerfile" \
        .
    ```

2. ランタイムコンテナを公開します。

    ```shell
    $ docker push <registry>/tff-runtime
    ```

3. Compute Engine インスタンスを作成します。

    1. Cloud Console で、[VM Instances](https://console.cloud.google.com/compute/instances) ページに移動します。

    2. **Create instance** をクリックします。

    3. **Firewall** セクションで、**Allow HTTP traffic** と **Allow HTTPS traffic** を選択します。

    4. **Create** をクリックして、インスタンスを作成します。

4. `ssh` でインスタンスに移動します。

    ```shell
    $ gcloud compute ssh <instance>
    ```

5. バックグラウンドでランタイムコンテナを実行します。

    ```shell
    $ docker run \
        --detach \
        --name=tff-runtime \
        --publish=8000:8000 \
        <registry>/tff-runtime
    ```

6. インスタンスを終了します。

    ```shell
    $ exit
    ```

7. インスタンスの内部 **IP アドレス**を取得します。

    これは、テストスクリプトのパラメータとして後で使用します。

    ```shell
    $ gcloud compute instances describe <instance> \
        --format='get(networkInterfaces[0].networkIP)'
    ```

### 4. クライアントコンテナでシミュレーションを実行します。

1. クライアントコンテナを構築します。

    ```shell
    $ docker build \
        --network=host \
        --tag "<registry>/tff-client" \
        --file "tensorflow_federated/tools/client/latest.Dockerfile" \
        .
    ```

2. クライアントコンテナを公開します。

    ```shell
    $ docker push <registry>/tff-client
    ```

3. Compute Engine インスタンスを作成します。

    1. Cloud Console で、[VM Instances](https://console.cloud.google.com/compute/instances) ページに移動します。

    2. **Create instance** をクリックします。

    3. **Firewall** セクションで、**Allow HTTP traffic** と **Allow HTTPS traffic** を選択します。

    4. **Create** をクリックして、インスタンスを作成します。

4. Compute Engine インスタンスに実験をコピーします。

    ```shell
    $ gcloud compute scp \
        "tensorflow_federated/tools/client/test.py" \
        <instance>:~
    ```

5. `ssh` でインスタンスに移動します。

    ```shell
    $ gcloud compute ssh <instance>
    ```

6. クライアントコンテナをインタラクティブに実行します。

    ターミナルに "Hello World" という文字列が出力されます。

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

7. Python スクリプトを実行します。

    ランタイムコンテナを実行しているインスタンスの内部 **IP アドレス**を使用します。

    ```shell
    $ python3 test.py --host '<IP address>'
    ```

8. コンテナを終了します。

    ```shell
    $ exit
    ```

9. インスタンスを終了します。

    ```shell
    $ exit
    ```
