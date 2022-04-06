<!--* freshness: { owner: 'maringeo' reviewed: '2021-12-13' review_interval: '6 months'} *-->

# 모델 호스팅 프로토콜

이 문서는 [thub.dev](https://tfhub.dev) - TFJS, TF Lite 및 TensorFlow 모델에서 모든 모델 유형을 호스팅할 때 사용되는 URL 규칙을 설명합니다. 또한, <a>thub.dev</a>의 TensorFlow 모델과 호환 가능한 서비스를 TensorFlow 프로그램에 로드하기 위해 <code>tensorflow_hub</code> 라이브러리에서 구현된 HTTP(S) 기반 프로토콜에 대해서도 설명합니다.

주요 기능은 코드에서 같은 URL을 사용하여 모델을 로드하고 브라우저에서 모델 설명서를 보는 것입니다.

## 일반 URL 규칙

[thub.dev](https://tfhub.dev)는 다음 URL 형식을 지원합니다.

- TF Hub 게시자는 `https://tfhub.dev/<publisher>`를 따릅니다.
- TF Hub 모음은 `https://tfhub.dev/<publisher>/collection/<collection_name>`를 따릅니다.
- TF Hub 모델에는 버전이 지정된 URL `https://tfhub.dev/<publisher>/<model_name>/<version>` 및 최신 버전으로 확인되는 버전 없는 URL `https://tfhub.dev/<publisher>/<model_name>`이 있습니다.

TF Hub 모델은 [thub.dev](https://tfhub.dev) 모델 URL에 URL 매개변수를 추가하여 압축된 자산으로 다운로드할 수 있습니다. 그러나 이를 달성하는 데 필요한 URL 매개변수는 모델 유형에 따라 다릅니다.

- TensorFlow 모델(SavedModel 및 TF1 Hub 형식 모두): `?tf-hub-format=compressed`를 TensorFlow 모델 URL에 추가합니다.
- TFJS 모델: `?tfjs-format=compressed`를 TFJS 모델 URL에 추가하여 압축된 또는 `/model.json?tfjs-format=file`을 다운로드하고 원격 스토리지에서 읽습니다.
- TF Lite 모델: `?lite-format=tflite`를 TF Lite 모델 URL에 추가합니다.

예를 들면 다음과 같습니다.

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 10%">
    <col style="width: 20%">
    <col style="width: 15%">
    <col style="width: 30%">
    <col style="width: 25%">
    <td style="text-align: center; background-color: #D0D0D0">유형</td>
    <td style="text-align: center; background-color: #D0D0D0">모델 URL</td>
    <td style="text-align: center; background-color: #D0D0D0">다운로드 유형</td>
    <td style="text-align: center; background-color: #D0D0D0">URL 매개변수</td>
    <td style="text-align: center; background-color: #D0D0D0">다운로드 URL</td>
  </tr>
  <tr>
    <td>TensorFlow (SavedModel, TF1 Hub 형식)</td>
    <td>https://tfhub.dev/google/spice/2</td>
    <td>.tar.gz</td>
    <td>?tf-hub-format=compressed</td>
    <td>https://tfhub.dev/google/spice/2?tf-hub-format=compressed</td>
  </tr>
  <tr>
    <td>TF Lite</td>
    <td>https://tfhub.dev/google/lite-model/spice/1</td>
    <td>.tflite</td>
    <td>?lite-format=tflite</td>
    <td>https://tfhub.dev/google/lite-model/spice/1?lite-format=tflite</td>
  </tr>
  <tr>
    <td>TF.js</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1</td>
    <td>.tar.gz</td>
    <td>?tfjs-format=compressed</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1?tfjs-format=compressed</td>
  </tr>
</table>

또한, 일부 모델은 다운로드하지 않고도 원격 스토리지에서 직접 읽을 수 있는 형식으로 호스팅됩니다. 이는 브라우저에서 TF.js 모델을 실행하거나 [Colab](https://colab.research.google.com/)에서 SavedModel을 로드하는 등 사용 가능한 로컬 스토리지가 없는 경우에 특히 유용합니다. 로컬로 다운로드하지 않고 원격으로 호스팅되는 모델을 읽으면 대기 시간이 늘어날 수 있습니다.

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 10%">
    <col style="width: 20%">
    <col style="width: 15%">
    <col style="width: 30%">
    <col style="width: 25%">
    <td style="text-align: center; background-color: #D0D0D0">유형</td>
    <td style="text-align: center; background-color: #D0D0D0">모델 URL</td>
    <td style="text-align: center; background-color: #D0D0D0">응답 유형</td>
    <td style="text-align: center; background-color: #D0D0D0">URL 매개변수</td>
    <td style="text-align: center; background-color: #D0D0D0">요청 URL</td>
  </tr>
  <tr>
    <td>TensorFlow (SavedModel, TF1 Hub 형식)</td>
    <td>https://tfhub.dev/google/spice/2</td>
    <td>문자열(압축되지 않은 모델이 저장되는 GCS 폴더 경로)</td>
    <td>?tf-hub-format=uncompressed</td>
    <td>https://tfhub.dev/google/spice/2?tf-hub-format=uncompressed</td>
  </tr>
  <tr>
    <td>TF.js</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1</td>
    <td>.json</td>
    <td>?tfjs-format=file</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1/model.json?tfjs-format=file</td>
  </tr>
</table>

## tensorflow_hub 라이브러리 프로토콜

이 섹션에서는 tensorflow_hub 라이브러리와 함께 사용하기 위해 [thub.dev](https://tfhub.dev)에서 모델을 호스팅하는 방법을 설명합니다. tensorflow_hub 라이브러리와 함께 동작하도록 자체 모델 리포지토리를 호스팅하려면 HTTP(s) 배포 서비스에서 이 프로토콜의 구현을 제공해야 합니다.

이 섹션에서는 TF Lite 및 TFJS 모델이 `tensorflow_hub` 라이브러리를 통해 다운로드되지 않으므로 호스팅에 대해서는 다루지 않습니다. 이들 모델 유형의 호스팅에 대한 자세한 내용은 [위에서](#general-url-conventions) 확인하세요.

### 압축된 호스팅

모델은 압축된 tar.gz 파일로 [thub.dev](https://tfhub.dev)에 저장됩니다. tensorflow_hub 라이브러리는 압축된 모델을 자동으로 다운로드합니다. 또는, 모델 URL에 `?tf-hub-format=compressed`를 추가하여 수동으로 다운로드할 수도 있습니다. 예를 들면 다음과 같습니다.

```shell
wget https://tfhub.dev/tensorflow/albert_en_xxlarge/1?tf-hub-format=compressed
```

아카이브의 루트는 모델 디렉토리의 루트이며 다음 예제와 같이 SavedModel을 포함해야 합니다.

```shell
# Create a compressed model from a SavedModel directory.
$ tar -cz -f model.tar.gz --owner=0 --group=0 -C /tmp/export-model/ .

# Inspect files inside a compressed model
$ tar -tf model.tar.gz
./
./variables/
./variables/variables.data-00000-of-00001
./variables/variables.index
./assets/
./saved_model.pb
```

레거시 [TF1 Hub 형식](https://www.tensorflow.org/hub/tf1_hub_module)과 함께 사용하기 위한 Tarball에는 `./tfhub_module.pb` 파일도 포함됩니다.

`tensorflow_hub` 라이브러리 모델 로딩 API 중 하나가 호출되면([hub.KerasLayer](https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer), [hub.load](https://www.tensorflow.org/hub/api_docs/python/hub/load) 등), 라이브러리는 모델을 다운로드하고 모델의 압축을 풀고 로컬로 캐싱합니다. `tensorflow_hub` 라이브러리는 모델 URL의 버전이 지정되고 지정된 버전의 모델 콘텐츠가 변경 불가능하여 무기한 캐싱될 수 있다고 예상합니다. [캐싱 모델](caching.md)에 대해 자세히 알아보세요.

![](https://raw.githubusercontent.com/tensorflow/hub/master/docs/images/library_download_cache.png)

### 압축되지 않은 호스팅

환경 변수 `TFHUB_MODEL_LOAD_FORMAT` 또는 명령줄 플래그 `--tfhub_model_load_format`가 `UNCOMPRESSED`로 설정된 경우, 모델은 로컬로 다운로드 및 압축 해제되는 대신 원격 저장소(GCS)에서 직접 읽어옵니다. 이 동작이 활성화되면 라이브러리는 `?tf-hub-format=uncompressed`를 모델 URL에 추가합니다. 이 요청은 압축되지 않은 모델 파일이 포함된 GCS의 폴더 경로를 반환합니다. 예를 들어, <br>`https://tfhub.dev/google/spice/2?tf-hub-format=uncompressed`<br>는 303 응답 본문에 `gs://tfhub-modules/google/spice/2/uncompressed`<br>를 반환합니다. 그런 다음 이 라이브러리는 해당 GCS 대상에서 모델을 읽습니다.
