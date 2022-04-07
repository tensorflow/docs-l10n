# Android용 TensorFlow Lite 빌드하기

이 문서는 TensorFlow Lite Android 라이브러리를 직접 빌드하는 방법을 설명합니다. 일반적으로 TensorFlow Lite Android 라이브러리를 로컬로 빌드할 필요는 없지만 사용하는 가장 쉬운 방법은 [MavenCentral에서 호스팅되는 TensorFlow Lite AAR](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite)을 사용하는 것입니다. Android 프로젝트에서 사용하는 방법에 대한 자세한 내용은 [Android 빠른 시작](../guide/android.md)을 참조하세요.

## 야간 스냅샷 사용하기

야간 스냅샷을 사용하려면 루트 Gradle 빌드 구성에 다음 저장소를 추가하세요.

```build
allprojects {
    repositories {      // should be already there
        mavenCentral()  // should be already there
        maven {         // add this repo to use snapshots
          name 'ossrh-snapshot'
          url 'https://oss.sonatype.org/content/repositories/snapshots'
        }
    }
}
```

## 로컬에서 TensorFlow Lite 빌드하기

경우에 따라 TensorFlow Lite의 로컬 빌드를 사용해야 할 수 있습니다. 예를 들어, [TensorFlow에서 선택한 연산](https://www.tensorflow.org/lite/guide/ops_select)을 포함하는 사용자 정의 바이너리를 빌드하거나 TensorFlow Lite를 로컬에서 변경하고자 할 수 있습니다.

### Docker를 사용하여 빌드 환경 설정하기

- Docker 파일을 다운로드합니다. Docker 파일을 다운로드하면 다음 서비스 약관이 파일 사용에 적용된다는 데 동의하는 것입니다.

*동의를 클릭하면 Android Studio 및 Android Native Development Kit의 모든 사용에 https://developer.android.com/studio/terms에서 제공하는 Android 소프트웨어 개발 키트 라이선스 계약(이 URL은 Google에서 수시로 업데이트하거나 변경할 수 있음)이 적용된다는 데 동의하는 것입니다.*

<!-- mdformat off(devsite fails if there are line-breaks in templates) -->

{% dynamic if 'tflite-android-tos' in user.acknowledged_walls and request.tld != 'cn' %} <a href="https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/dockerfiles/tflite-android.Dockerfile">여기</a>에서 Docker 파일을 다운로드할 수 있습니다. {% dynamic else %} 파일을 다운로드하려면 서비스 약관에 동의해야 합니다. <a class="button button-blue devsite-acknowledgement-link" data-globally-unique-wall-id="tflite-android-tos">동의</a> {% dynamic endif %}

<!-- mdformat on -->

- 선택적으로, Android SDK 또는 NDK 버전을 변경할 수 있습니다. 다운로드한 Docker 파일을 빈 폴더에 놓고 다음을 실행하여 Docker 이미지를 빌드합니다.

```shell
docker build . -t tflite-builder -f tflite-android.Dockerfile
```

- 컨테이너 내부의 /host_dir에 현재 폴더를 마운트하여 대화형으로 docker 컨테이너를 시작합니다(/tensorflow_src는 컨테이너 내부의 TensorFlow 저장소입니다).

```shell
docker run -it -v $PWD:/host_dir tflite-builder bash
```

Windows에서 PowerShell을 사용하는 경우 "$PWD"를 "pwd"로 바꿉니다.

호스트에서 TensorFlow 리포지토리를 사용하려면, 대신 해당 호스트 디렉터리를 마운트합니다(-v hostDir:/host_dir).

- 컨테이너 내부에 들어왔으면 다음을 실행하여 추가 Android 도구와 라이브러리를 다운로드할 수 있습니다(라이선스에 동의해야 할 수 있음).

```shell
sdkmanager \
  "build-tools;${ANDROID_BUILD_TOOLS_VERSION}" \
  "platform-tools" \
  "platforms;android-${ANDROID_API_LEVEL}"
```

이제 [WORKSPACE 및 .bazelrc 구성](#configure_workspace_and_bazelrc) 섹션으로 진행하여 빌드 설정을 구성해야 합니다.

라이브러리 빌드를 완료한 후, 호스트에서 액세스할 수 있도록 컨테이너 내부의 /host_dir에 라이브러리를 복사할 수 있습니다.

### Docker 없이 빌드 환경 설정하기

#### Bazel 및 Android 필수 구성 요소 설치하기

Bazel은 TensorFlow의 기본 빌드 시스템입니다. Bazel을 사용하여 빌드하려면 시스템에 Android NDK 및 SDK가 설치되어 있어야 합니다.

1. 최신 버전의 [Bazel 빌드 시스템](https://bazel.build/versions/master/docs/install.html)을 설치합니다.
2. 네이티브(C/C++) TensorFlow Lite 코드를 빌드하려면 Android NDK가 필요합니다. 현재 권장되는 버전은 19c이며 [여기](https://developer.android.com/ndk/downloads/older_releases.html#ndk-19c-downloads)에서 찾을 수 있습니다.
3. Android SDK 및 빌드 도구는 [여기](https://developer.android.com/tools/revisions/build-tools.html)에서 얻거나, [Android Studio](https://developer.android.com/studio/index.html)의 일부로 얻을 수도 있습니다. TensorFlow Lite 빌드에 권장되는 버전은 Build tools API &gt;= 23입니다.

### WORKSPACE 및 .bazelrc 구성하기

루트 TensorFlow 체크아웃 디렉토리에서 `./configure` 스크립트를 실행하고 스크립트가 Android 빌드용 `./WORKSPACE`를 대화식으로 구성할 것인지 물으면 "Yes"를 선택합니다. 스크립트는 다음 환경 변수를 사용하여 설정 구성을 시도합니다.

- `ANDROID_SDK_HOME`
- `ANDROID_SDK_API_LEVEL`
- `ANDROID_NDK_HOME`
- `ANDROID_NDK_API_LEVEL`

이들 변수가 설정되지 않은 경우, 스크립트 프롬프트에서 대화식으로 제공해야 합니다. 성공적으로 구성되면 루트 폴더의 `.tf_configure.bazelrc` 파일에 다음과 같은 항목이 생깁니다.

```shell
build --action_env ANDROID_NDK_HOME="/usr/local/android/android-ndk-r17c"
build --action_env ANDROID_NDK_API_LEVEL="21"
build --action_env ANDROID_BUILD_TOOLS_VERSION="28.0.3"
build --action_env ANDROID_SDK_API_LEVEL="23"
build --action_env ANDROID_SDK_HOME="/usr/local/android/android-sdk-linux"
```

### 빌드 및 설치하기

Bazel이 올바르게 구성되면 다음과 같이 루트 체크아웃 디렉토리에서 TensorFlow Lite AAR을 빌드할 수 있습니다.

```sh
bazel build -c opt --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  //tensorflow/lite/java:tensorflow-lite
```

그러면 `bazel-bin/tensorflow/lite/java/`에 AAR 파일이 생성됩니다. 그 결과로 몇 가지 아키텍처를 포함한 "뚱뚱한" AAR이 구축된다는 점에 유의하세요. 모두 필요하지 않은 경우 배포 환경에 적합하게 일부만 사용합니다.

다음과 같이 모델의 일부만 대상으로 하는 더 작은 AAR 파일을 빌드할 수 있습니다.

```sh
bash tensorflow/lite/tools/build_aar.sh \
  --input_models=model1,model2 \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

위 스크립트는 모델 중 하나가 Tensorflow 연산을 사용하는 경우 `tensorflow-lite.aar` 파일, 및 선택적으로 `tensorflow-lite-select-tf-ops.aar` 파일을 생성합니다. 자세한 내용은 [TensorFlow Lite 바이너리 크기 줄이기](../guide/reduce_binary_size.md) 섹션을 참조하세요.

#### 프로젝트에 직접 AAR 추가하기

`tensorflow-lite.aar` 파일을 프로젝트의 `libs`라고 하는 디렉토리로 이동합니다. 새 디렉토리를 참조하도록 앱의 `build.gradle` 파일을 수정하고 기존 TensorFlow Lite 종속성을 새 로컬 라이브러리로 바꿉니다. 예를 들면 다음과 같습니다.

```
allprojects {
    repositories {
        mavenCentral()
        maven {  // Only for snapshot artifacts
            name 'ossrh-snapshot'
            url 'https://oss.sonatype.org/content/repositories/snapshots'
        }
        flatDir {
            dirs 'libs'
        }
    }
}

dependencies {
    compile(name:'tensorflow-lite', ext:'aar')
}
```

#### 로컬 Maven 리포지토리에 AAR 설치하기

루트 체크아웃 디렉토리에서 다음 명령을 실행합니다.

```sh
mvn install:install-file \
  -Dfile=bazel-bin/tensorflow/lite/java/tensorflow-lite.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite -Dversion=0.1.100 -Dpackaging=aar
```

앱의 `build.gradle`에서 `mavenLocal()` 종속성이 있는지 확인하고 표준 TensorFlow Lite 종속성을 선택한 TensorFlow 연산을 지원하는 종속성으로 바꿉니다.

```
allprojects {
    repositories {
        mavenCentral()
        maven {  // Only for snapshot artifacts
            name 'ossrh-snapshot'
            url 'https://oss.sonatype.org/content/repositories/snapshots'
        }
        mavenLocal()
    }
}

dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.1.100'
}
```

여기서 `0.1.100` 버전은 순전히 테스트/개발을 위한 것입니다. 로컬 AAR이 설치되면 앱 코드에서 표준 [TensorFlow Lite Java 추론 API](../guide/inference.md)를 사용할 수 있습니다.
