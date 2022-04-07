# tfds 및 Google 클라우드 스토리지

GCS(Google Cloud Storage)는 여러 가지 이유로 tfd와 함께 사용할 수 있습니다.

- 사전 처리된 데이터 저장하기
- GCS에 저장된 데이터가 있는 데이터세트에 액세스하기

## TFDS GCS 버킷을 통한 액세스

Some datasets are available directly in our GCS bucket [`gs://tfds-data/datasets/`](https://console.cloud.google.com/storage/browser/tfds-data) without any authentication:

- `tfds.load(..., try_gcs=False)`(기본값)인 경우, 데이터세트는 `download_and_prepare` 중에 `~/tensorflow_datasets`에서 로컬로 복사됩니다.
- `tfds.load(..., try_gcs=True)`인 경우, 데이터세트는 GCS에서 직접 스트리밍됩니다(`download_and_prepare`는 건너뜀).

`tfds.is_dataset_on_gcs('mnist')`를 이용해 데이터세트가 공용 버킷에서 호스팅되는지 여부를 확인할 수 있습니다.

## 인증

시작하기 전에 인증 방법을 결정해야 합니다. 3가지 옵션이 있습니다.

- 인증 없음(일명 익명 액세스)
- Google 계정 사용하기
- 서비스 계정 사용하기(팀의 다른 사용자와 쉽게 공유할 수 있음)

You can find detailed information in [Google Cloud documentation](https://cloud.google.com/docs/authentication/getting-started)

### 단순화된 지침

colab에서 실행하는 경우 계정으로 인증할 수 있지만, 다음을 실행하세요.

```python
from google.colab import auth
auth.authenticate_user()
```

로컬 머신(또는 VM)에서 실행하는 경우, 다음을 실행하여 계정으로 인증할 수 있습니다.

```shell
gcloud login application-default
```

서비스 계정으로 로그인하려면, JSON 파일 키를 다운로드하고 설정하세요.

```shell
export GOOGLE_APPLICATION_CREDENTIALS=<JSON_FILE_PATH>
```

## Google Cloud Storage를 사용하여 사전 처리된 데이터 저장하기

일반적으로 TensorFlow Datasets를 사용하면 다운로드 및 준비된 데이터가 로컬 디렉토리(기본적으로 `~/tensorflow_datasets`)에 캐시됩니다.

로컬 디스크가 일시적이거나(임시 클라우드 서버 또는 [Colab 노트북](https://colab.research.google.com)) 여러 머신에서 데이터에 액세스해야 하는 일부 환경에서는 `data_dir`을 GCS(Google Cloud Storage) 버킷과 같은 클라우드 스토리지 시스템으로 설정하는 것이 유용합니다.

### 어떻게?

[GCS 버킷을 생성하고](https://cloud.google.com/storage/docs/creating-buckets) 사용자(또는 서비스 계정)가 버킷에 대한 읽기/쓰기 권한을 갖도록 합니다(위의 인증 지침 참조).

`tfds`를 사용할 때, `data_dir`를  `"gs://YOUR_BUCKET_NAME"`로 설정할 수 있습니다.

```python
ds_train, ds_test = tfds.load(name="mnist", split=["train", "test"], data_dir="gs://YOUR_BUCKET_NAME")
```

### 주의 사항:

- 이 접근 방식은 데이터 액세스에 `tf.io.gfile`만 사용하는 데이터세트에 적용됩니다. 이 방식은 대부분의 데이터세트에 적용되지만, 전부는 아닙니다.
- GCS에 액세스하면 원격 서버에 액세스하고 데이터를 스트리밍하므로 네트워크 비용이 발생할 수 있습니다.

## GCS에 저장된 데이터세트에 액세스하기

데이터세트 소유자가 익명 액세스를 허용한 경우, tfds.load 코드를 실행할 수 있으며 정상적인 인터넷 다운로드처럼 동작합니다.

데이터세트에 인증이 필요한 경우, 위의 지침을 사용하여 원하는 옵션(소유자 계정 vs 서비스 계정)을 결정하고 계정 이름(일명 이메일)을 데이터세트 소유자에게 전달하세요. GCS 디렉토리에 액세스할 수 있게 되면, tfds 다운로드 코드를 실행할 수 있습니다.
