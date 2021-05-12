<!--* freshness: { owner: 'maringeo' reviewed: '2021-03-15' review_interval: '3 months'} *-->

# 모델 호스팅 프로토콜

This document describes the URL coventions used when hosting all model types on [tfhub.dev](https://tfhub.dev) - TFJS, TF Lite and TensorFlow models. It also describes the HTTP(S)-based protocol implemented by the `tensorflow_hub` library in order to load TensorFlow models from [tfhub.dev](https://tfhub.dev) and compatibe services into TensorFlow programs.

주요 기능은 코드에서 같은 URL을 사용하여 모델을 로드하고 브라우저에서 모델 설명서를 보는 것입니다.

## 일반 URL 규칙

[tfhub.dev](https://tfhub.dev) supports the following URL formats:

- TF Hub 게시자는 `https://tfhub.dev/<publisher>`를 따릅니다.
- TF Hub 모음은 `https://tfhub.dev/<publisher>/collection/<collection_name>`를 따릅니다.
- TF Hub 모델에는 버전이 지정된 URL `https://tfhub.dev/<publisher>/<model_name>/<version>` 및 최신 버전으로 확인되는 버전 없는 URL `https://tfhub.dev/<publisher>/<model_name>`이 있습니다.

TF Hub models can be downloaded as compressed assets by appending URL parameters to the [tfhub.dev](https://tfhub.dev) model URL. However, the URL paramters required to achieve that depend on the model type:

- TensorFlow models (both SavedModel and TF1 Hub formats): append `?tf-hub-format=compressed` to the TensorFlow model url.
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

Additionally, some models also are hosted in a format that can be read directly from remote storage without being downloaded. This is especially useful if there is no local storage available, such as running a TF.js model in the browser or loading a SavedModel on [Colab](https://colab.research.google.com/). Be conscious that reading models that are hosted remotely without being downloaded locally may increase latency.

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 10%">
    <col style="width: 20%">
    <col style="width: 15%">
    <col style="width: 30%">
    <col style="width: 25%">
    <td style="text-align: center; background-color: #D0D0D0">유형</td>
    <td style="text-align: center; background-color: #D0D0D0">모델 URL</td>
    <td style="text-align: center; background-color: #D0D0D0">Response type</td>
    <td style="text-align: center; background-color: #D0D0D0">URL 매개변수</td>
    <td style="text-align: center; background-color: #D0D0D0">Request URL</td>
  </tr>
  <tr>
    <td>TensorFlow (SavedModel, TF1 Hub format)</td>
    <td>https://tfhub.dev/google/spice/2</td>
    <td>String (Path to GCS folder where the uncompressed model is stored)</td>
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

This section describes how we host models on [tfhub.dev](https://tfhub.dev) for use with the tensorflow_hub library. If you want to host your own model repository to work with the tensorflow_hub library, your HTTP(s) distribution service should provide an implementation of this protocol.

이 섹션에서는 TF Lite 및 TFJS 모델이 `tensorflow_hub` 라이브러리를 통해 다운로드되지 않으므로 호스팅에 대해서는 다루지 않습니다. 이들 모델 유형의 호스팅에 대한 자세한 내용은 [위에서](#general-url-conventions) 확인하세요.

### Compressed Hosting

Models are stored on [tfhub.dev](https://tfhub.dev) as compressed tar.gz files. By default, the tensorflow_hub library automatically downloads the compressed model. They can also be manually downloaded by appending the `?tf-hub-format=compressed` to the model url, for example:

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

When one of `tensorflow_hub` library model loading APIs is invoked ([hub.KerasLayer](https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer), [hub.load](https://www.tensorflow.org/hub/api_docs/python/hub/load), etc) the library downloads the model, uncompresses the model and caches it locally. The `tensorflow_hub` library expects that model URLs are versioned and that the model content of a given version is immutable, so that it can be cached indefinitely. Learn more about [caching models](caching.md).

![](https://raw.githubusercontent.com/tensorflow/hub/master/docs/images/library_download_cache.png)

### Uncompressed Hosting

When the environment variable `TFHUB_MODEL_LOAD_FORMAT` or the command-line flag `--tfhub_model_load_format` is set to `UNCOMPRESSED`, the model is read directly from remote storage (GCS) instead of being downloaded and uncompressed locally. When this behavior is enabled the library appends `?tf-hub-format=uncompressed` to the model URL. That request returns the path to the folder on GCS that contains the uncompressed model files. As an example,
 `https://tfhub.dev/google/spice/2?tf-hub-format=uncompressed`
 returns
 `gs://tfhub-modules/google/spice/2/uncompressed` in the body of the 303 response. The library then reads the model from that GCS destination.
