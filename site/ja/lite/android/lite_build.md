# Android 用の TensorFlow Lite を構築する

This document describes how to build TensorFlow Lite Android library on your own. Normally, you do not need to locally build TensorFlow Lite Android library. If you just want to use it, the easiest way is using the [TensorFlow Lite AAR hosted at MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite). See [Android quickstart](../guide/android.md) for more details on how to use them in your Android projects.

## Use Nightly Snapshots

To use nightly snapshots, add the following repo to your root Gradle build config.

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

## ローカルで TensorFlow Lite を構築する

In some cases, you might wish to use a local build of TensorFlow Lite. For example, you may be building a custom binary that includes [operations selected from TensorFlow](https://www.tensorflow.org/lite/guide/ops_select), or you may wish to make local changes to TensorFlow Lite.

### Docker を使用して構築環境をセットアップする

- Docker ファイルをダウンロードします。Docker ファイルをダウンロードすることで、以下の利用規約に同意したことになります。

*同意をクリックすることで、すべての Android Studio および Android Native Development Kit の使用は、https://developer.android.com/studio/terms から入手可能な Android Software Development Kit License Agreement に準拠することに同意したことになります（このような URL は Google により随時更新または変更される場合があります）。*

<!-- mdformat off(devsite fails if there are line-breaks in templates) -->

{% dynamic if 'tflite-android-tos' in user.acknowledged_walls and request.tld != 'cn' %} Docker ファイルは<a href="https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/dockerfiles/tflite-android.Dockerfile">こちら</a>からダウンロードできます。{% dynamic else %}ファイルをダウンロードするには、利用規約に同意する必要があります。<a class="button button-blue devsite-acknowledgement-link" data-globally-unique-wall-id="tflite-android-tos">同意する</a> {% dynamic endif %}

<!-- mdformat on -->

- オプションで Android の SDK や NDK のバージョンを変更することができます。ダウンロードした Docker ファイルを空のフォルダに入れ、実行して Docker イメージを構築します。

```shell
docker build . -t tflite-builder -f tflite-android.Dockerfile
```

- Start the docker container interactively by mounting your current folder to /host_dir inside the container (note that /tensorflow_src is the TensorFlow repository inside the container):

```shell
docker run -it -v $PWD:/host_dir tflite-builder bash
```

Windows で PowerShell を使用する場合は、"$PWD" を "pwd" に置き換えます。

If you would like to use a TensorFlow repository on the host, mount that host directory instead (-v hostDir:/host_dir).

- コンテナの中に入ると、以下を実行して追加の Android ツールやライブラリをダウンロードすることができます（ライセンスに同意する必要があるかもしれないので注意してください）。

```shell
sdkmanager \
  "build-tools;${ANDROID_BUILD_TOOLS_VERSION}" \
  "platform-tools" \
  "platforms;android-${ANDROID_API_LEVEL}"
```

Now you should proceed to the [Configure WORKSPACE and .bazelrc](#configure_workspace_and_bazelrc) section to configure the build settings.

After you finish building the libraries, you can copy them to /host_dir inside the container so that you can access them on the host.

### Docker を使用せずに構築環境をセットアップする

#### Bazel と Android の前提条件をインストールする

Bazel is the primary build system for TensorFlow. To build with it, you must have it and the Android NDK and SDK installed on your system.

1. Install the latest version of the [Bazel build system](https://bazel.build/versions/master/docs/install.html).
2. The Android NDK is required to build the native (C/C++) TensorFlow Lite code. The current recommended version is 19c, which may be found [here](https://developer.android.com/ndk/downloads/older_releases.html#ndk-19c-downloads).
3. The Android SDK and build tools may be obtained [here](https://developer.android.com/tools/revisions/build-tools.html), or alternatively as part of [Android Studio](https://developer.android.com/studio/index.html). Build tools API &gt;= 23 is the recommended version for building TensorFlow Lite.

### WORKSPACE と .bazelrc の構成

This is a one-time configuration step that is required to build the TF Lite libraries. Run the `./configure` script in the root TensorFlow checkout directory, and answer "Yes" when the script asks to interactively configure the `./WORKSPACE` for Android builds. The script will attempt to configure settings using the following environment variables:

- `ANDROID_SDK_HOME`
- `ANDROID_SDK_API_LEVEL`
- `ANDROID_NDK_HOME`
- `ANDROID_NDK_API_LEVEL`

If these variables aren't set, they must be provided interactively in the script prompt. Successful configuration should yield entries similar to the following in the `.tf_configure.bazelrc` file in the root folder:

```shell
build --action_env ANDROID_NDK_HOME="/usr/local/android/android-ndk-r19c"
build --action_env ANDROID_NDK_API_LEVEL="21"
build --action_env ANDROID_BUILD_TOOLS_VERSION="28.0.3"
build --action_env ANDROID_SDK_API_LEVEL="23"
build --action_env ANDROID_SDK_HOME="/usr/local/android/android-sdk-linux"
```

### 構築してインストールする

Bazel が正しく設定されたら、以下のようにしてルートチェックアウトディレクトリから TensorFlow Lite AAR を構築します。

```sh
bazel build -c opt --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  //tensorflow/lite/java:tensorflow-lite
```

これにより、`bazel-bin/tensorflow/lite/java/` に AAR ファイルが生成されます。これは、複数の異なるアーキテクチャの「大きな」AAR を構築するので注意してください。その全部が必要ではない場合は、デプロイ環境に適切なサブセットを使用します。

You can build smaller AAR files targeting only a set of models as follows:

```sh
bash tensorflow/lite/tools/build_aar.sh \
  --input_models=model1,model2 \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

上記のスクリプトは、`tensorflow-lite.aar` ファイルを生成し、モデルのいずれかが Tensorflow 演算子を使用している場合には、オプションで `tensorflow-lite-select-tf-ops.aar` ファイルを生成します。詳細については、[TensorFlow Lite バイナリサイズの削減](../guide/reduce_binary_size.md)の項目をご覧ください。

#### プロジェクトに AAR を直接追加する

プロジェクト内の `tensorflow-lite.aar` ファイルを `libs` というディレクトリに移動します。アプリの `build.gradle` ファイルを新しいディレクトリを参照するように変更し、既存の TensorFlow Lite の依存関係を新しいローカルライブラリに置き換えます。例を示します。

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

#### ローカルの Maven リポジトリに AAR をインストールする

ルートチェックアウト ディレクトリから以下のコマンドを実行します。

```sh
mvn install:install-file \
  -Dfile=bazel-bin/tensorflow/lite/java/tensorflow-lite.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite -Dversion=0.1.100 -Dpackaging=aar
```

アプリの `build.gradle` で、`mavenLocal()` 依存関係があることを確認し、標準の TensorFlow Lite 依存関係を、セレクト TensorFlow 演算子をサポートする依存関係に置き換えます。

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

ここで示している `0.1.100` バージョンは、純粋にテストと開発目的のバージョンなので注意してください。ローカルに AAR がインストールされていると、アプリコード内で標準の [TensorFlow Lite Java 推論 API](../guide/inference.md) を使用することができます。
