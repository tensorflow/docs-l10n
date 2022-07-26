# GCP의 TFF 시뮬레이션

이 가이드에서는 GCP에서 TFF 시뮬레이션을 실행하는 방법을 설명합니다.

## 단일 런타임 컨테이너에서 시뮬레이션 실행하기

### 1. [Cloud SDK를 설치하고 초기화합니다.](https://cloud.google.com/sdk/docs/quickstarts)

### 2. TensorFlow Federated 리포지토리를 복제합니다.

```shell
$ git clone https://github.com/tensorflow/federated.git
$ cd "federated"
```

### 3. 단일 런타임 컨테이너를 실행합니다.

1. 런타임 컨테이너를 빌드합니다.

    ```shell
    $ docker build \
        --network=host \
        --tag "<registry>/tff-runtime" \
        --file "tensorflow_federated/tools/runtime/container/latest.Dockerfile" \
        .
    ```

2. 런타임 컨테이너를 게시합니다.

    ```shell
    $ docker push <registry>/tff-runtime
    ```

3. Compute Engine 인스턴스를 생성합니다.

    1. Cloud Console에서 [VM 인스턴스](https://console.cloud.google.com/compute/instances) 페이지로 이동합니다.

    2. **Create instance**를 클릭합니다.

    3. **Firewall** 섹션에서, **Allow HTTP traffic**과 **Allow HTTPS traffic**을 선택합니다.

    4. **Create**를 클릭하여 인스턴스를 생성합니다.

4. `ssh`를 사용하여 인스턴스에 연결합니다.

    ```shell
    $ gcloud compute ssh <instance>
    ```

5. 백그라운드에서 런타임 컨테이너를 실행합니다.

    ```shell
    $ docker run \
        --detach \
        --name=tff-runtime \
        --publish=8000:8000 \
        <registry>/tff-runtime
    ```

6. 인스턴스를 종료합니다.

    ```shell
    $ exit
    ```

7. 인스턴스의 내부 <strong>IP 주소</strong>를 가져옵니다.

    이 주소는 나중에 테스트 스크립트의 매개변수로 사용됩니다.

    ```shell
    $ gcloud compute instances describe <instance> \
        --format='get(networkInterfaces[0].networkIP)'
    ```

### 4. 클라이언트 컨테이너에서 시뮬레이션을 실행합니다.

1. 클라이언트 컨테이너를 빌드합니다.

    ```shell
    $ docker build \
        --network=host \
        --tag "<registry>/tff-client" \
        --file "tensorflow_federated/tools/client/latest.Dockerfile" \
        .
    ```

2. 클라이언트 컨테이너를 게시합니다.

    ```shell
    $ docker push <registry>/tff-client
    ```

3. Compute Engine 인스턴스를 생성합니다.

    1. Cloud Console에서 [VM 인스턴스](https://console.cloud.google.com/compute/instances) 페이지로 이동합니다.

    2. **Create instance**를 클릭합니다.

    3. **Firewall** 섹션에서, **Allow HTTP traffic**과 **Allow HTTPS traffic**을 선택합니다.

    4. **Create**를 클릭하여 인스턴스를 생성합니다.

4. 실험을 Compute Engine 인스턴스에 복사합니다.

    ```shell
    $ gcloud compute scp \
        "tensorflow_federated/tools/client/test.py" \
        <instance>:~
    ```

5. `ssh`를 사용하여 인스턴스에 연결합니다.

    ```shell
    $ gcloud compute ssh <instance>
    ```

6. 클라이언트 컨테이너를 대화형으로 실행합니다.

    문자열 "Hello World"이 터미널에 인쇄되어야 합니다.

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

7. Python 스크립트를 실행합니다.

    런타임 컨테이너를 실행하는 인스턴스의 <strong>IP 주소</strong>를 사용합니다.

    ```shell
    $ python3 test.py --host '<IP address>'
    ```

8. 컨테이너를 종료합니다.

    ```shell
    $ exit
    ```

9. 인스턴스를 종료합니다.

    ```shell
    $ exit
    ```
